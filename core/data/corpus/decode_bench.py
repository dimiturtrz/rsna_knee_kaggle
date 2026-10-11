"""Is decode I/O-bound or GIL-bound? `python -m core.data.corpus.decode_bench --studies 96`.

`Submission.decode` runs eight threads and says why: "above the four available cores on purpose:
decoding is dominated by waiting on the mount." That is an assumption, it was never measured, and it is
wrong — 96 real studies, warm, returning the actual bags:

    serial         25.7s   0.27s/study   1.00x
    threads(4)     19.0s   0.20s/study   1.35x
    threads(8)     26.1s   0.27s/study   0.98x   <- the shipped setting, no better than no pool
    processes(4)   11.2s   0.12s/study   2.29x
    processes(8)   12.8s   0.13s/study   2.01x

If decode were mount-bound, threads would scale and processes would not beat them. pydicom parses
headers and tags in Python, so decode is GIL-bound, and eight threads contend badly enough to lose to
running serially.

WHY IT IS COMMITTED RATHER THAN RUN ONCE. The out-of-fold oracle was a scratchpad script, was never
committed, and is gone — and two beads then argued for three days about a number nobody could
recompute. A performance claim is exactly the kind that goes stale silently: the numbers above are true
of this machine on this day, and the only thing that keeps them honest is that anyone can re-run them.

MEASURE IT ON THE HOST THAT MATTERS. This is a 16-core box on local NVMe; Kaggle is four cores on a
slower mount, so the RATIO will differ. What transfers is the GIL-boundedness, which is a property of
the code rather than the disk.
"""

import logging
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from core import cli

if TYPE_CHECKING:
    from core.data.corpus.study import Study

logger = logging.getLogger(__name__)

STUDIES = 96
"""Enough that a pool's startup is not the measurement. At 24 the Windows spawn cost alone flattered
threads and hid the process pool's win entirely."""

SLICES = "DECODE_BENCH_SLICES"
"""How the depth reaches a worker. A process pool SPAWNS here, so a module global set in the parent is
not inherited — but the environment is, and `os.environ` is the one channel that survives a spawn
without being sent per item (which would time pickling, the same trap `PLANES` avoids)."""

DEFAULT_SLICES = 12
"""The depth every shipped member reads at except the 20-slice group. The bench's own default, so a
run with no flag prices what the champion actually serves."""

type ReadGroup = tuple[int, tuple[float, float]]
"""One decode of every study: how many slices, read inside which (low, high) band of the series."""

CHAMPION_GROUPS: tuple[ReadGroup, ...] = ((20, (0.1, 0.9)), (12, (0.1, 0.9)), (12, (0.2, 0.8)))
"""The three read groups `raptorweight_v1` serves — three seats, one seat, two seats. Read off the
members' own recipes on 2026-09-06, and the reason its Kaggle log decodes the same three visible
studies three times over (`decode 4s`, `2s`, `2s`) and projects 7.6 h of a 9 h cap."""

PLANES: dict[str, str] = {}
"""Filled per worker on first use. A process pool re-imports this module rather than inheriting state,
and the alternative — sending the table with every study — would measure pickling, not decoding."""


class DecodeBench:
    """One decode workload, timed under each way of running it."""

    def timings[T](self, work: Callable[[T], object], items: Sequence[T]) -> dict[str, float]:
        """Seconds for each strategy, warmed first so the earliest one is not charged for cold cache.

        `work` is injected rather than reached for, so the arithmetic here can be exercised without a
        corpus — and so the same harness can time a different stage later without being rewritten.
        """
        with ThreadPoolExecutor(max_workers=8) as warm:
            list(warm.map(work, items))

        measured: dict[str, float] = {}
        for label, runner, workers in (
            ("serial", None, 1),
            ("threads(4)", ThreadPoolExecutor, 4),
            ("threads(8)", ThreadPoolExecutor, 8),
            ("processes(4)", ProcessPoolExecutor, 4),
            ("processes(8)", ProcessPoolExecutor, 8),
        ):
            started = time.time()
            if runner is None:
                for item in items:
                    work(item)
            else:
                with runner(max_workers=workers) as pool:
                    list(pool.map(work, items))
            measured[label] = time.time() - started
        return measured

    @staticmethod
    def _union_all(studies: Sequence[str], groups: Sequence[ReadGroup]) -> tuple[int, int]:
        """Distinct and summed opens over a corpus slice for ANY set of read groups.

        Taking the groups rather than reading the constant is what prices a CANDIDATE group: the memo
        makes an entry's decode bill the size of the UNION, so what a new reading costs is
        `union(champion + candidate) - union(champion)`, and that marginal is nothing like the group's
        own size — a band nested inside one already served costs almost nothing, a shifted band costs
        nearly all of itself.
        """
        counted = [BenchStudy(study).opens(groups) for study in studies]
        return sum(pair[0] for pair in counted), sum(pair[1] for pair in counted)

    def report(self, measured: dict[str, float], studies: int) -> list[str]:
        """Each strategy against SERIAL, because the question is whether a pool earns its complexity."""
        baseline = measured.get("serial") or 1.0
        return [
            f"{label:14s} {seconds:6.1f}s   {seconds / max(studies, 1):.2f}s/study   {baseline / seconds:.2f}x"
            for label, seconds in measured.items()
        ]


@dataclass(frozen=True)
class BenchStudy:
    """One study directory, read the three ways the bench prices: decoded, headers only, file opens.

    The directory is the instance because all three readings are of one study, and each used to thread
    it as its only argument while re-spelling the same planes lookup and `Study` construction. Frozen and
    holding one string, so a process pool pickles it as cheaply as the bare path it replaces; the
    readings are reached by qualified name (`BenchStudy.decoded`), which is what the child process needs.

    None is unit-tested: every reading opens the configured corpus, which `hygiene.small` forbids a unit
    test to reach. The harness that times them, `DecodeBench`, is.
    """

    directory: str

    def decoded(self) -> object:  # devtools-ignore: test-mirror
        """One study through the real reader."""
        return self._study().bag(int(os.environ.get(SLICES, DEFAULT_SLICES)))

    def headers(self) -> object:  # devtools-ignore: test-mirror
        """The same study with the PIXELS left unread — every header parsed, nothing decoded.

        This is the term no depth can cut. `Acquisition.ordered_files` header-parses EVERY file in
        every series to sort them, and only then does `SlotReader.cropped` decode the `slices` files
        the band selects. So decode time is `headers(all files) + pixels(slices)`, and halving the
        depth halves the second term only. Timed separately because a lever that moves the smaller of
        the two is a lever priced wrong — and if headers dominate, ordering by filename or parsing one
        header per series is worth more than any slice count.
        """
        return [acquisition.ordered_files() for acquisition in self._study().acquisitions]

    def opens(self, groups: Sequence[ReadGroup]) -> tuple[int, int]:  # devtools-ignore: test-mirror
        """How many DISTINCT files the read groups open in this study, against their sum.

        A group is a decode of every study, but two groups at the same band and different depths open
        overlapping files, and the historic band 0.2-0.8 sits INSIDE 0.1-0.9. So the floor a file-level
        memo reaches is the size of the union, not the size of one group — and unlike unifying the
        groups, a memo changes no pixels any member sees, so it costs 0 AUC and can ride in final A.

        Returns `(distinct, total)` per study; the caller sums both. Reads headers only — the count is a
        property of `band_indices` and the file list, so no pixel is decoded to answer it.
        """
        from core.data.corpus.slot_reader import Band, SlotReader  # noqa: PLC0415 - CLI-only

        study = self._study()
        distinct: set[tuple[str, int]] = set()
        total = 0
        # ROUTED acquisitions only. A study holds many series and the pipeline decodes one per slot;
        # counting every series would price a decode nothing performs and inflate the memo's win.
        for acquisition in study.pipeline.route(study.acquisitions, PLANES).values():
            available = len(acquisition.ordered_files())
            for slices, band in groups:
                opened = SlotReader(band=Band(*band)).band_indices(available, slices)
                distinct.update((acquisition.series, index) for index in opened)
                total += len(opened)
        return len(distinct), total

    def _study(self) -> "Study":
        global PLANES
        from core.data.corpus.study import SlotPipeline, Study  # noqa: PLC0415 - CLI-only
        from core.data.corpus.study_cache import CorpusSplit  # noqa: PLC0415 - CLI-only
        from core.paths import DataRoot  # noqa: PLC0415 - CLI-only

        if not PLANES:
            PLANES = CorpusSplit(DataRoot.from_config(Path("paths.yaml")).raw()).planes()
        return Study.from_directory(Path(self.directory), PLANES, SlotPipeline())


def main() -> None:
    """Time the real decode under each strategy and print what each buys over running serially."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--studies", type=int, default=STUDIES)
    command.parser.add_argument("--slices", type=int, default=DEFAULT_SLICES, help="depth to decode at")
    command.parser.add_argument(
        "--headers-only", action="store_true", help="time the header parse alone, the term no depth cuts"
    )
    command.parser.add_argument(
        "--union", action="store_true", help="count distinct vs opened files across the champion's read groups"
    )
    command.parser.add_argument(
        "--extra",
        action="append",
        default=[],
        metavar="SLICES:LOW:HIGH",
        help="price a CANDIDATE read group's marginal decode over the champion's union, one flag each",
    )
    arguments = command.parsed()

    from core.data.corpus.study_cache import CorpusSplit  # noqa: PLC0415 - CLI-only

    source = CorpusSplit(command.rooted(arguments).raw()).source
    studies = [str(path) for path in sorted(source.iterdir()) if path.is_dir()]
    chosen = studies[: arguments.studies]

    if arguments.union:
        distinct, total = DecodeBench._union_all(chosen, CHAMPION_GROUPS)  # noqa: SLF001 - its own CLI
        lines = [
            f"{len(chosen)} studies: {total / len(chosen):.1f} files opened per study across "
            f"{len(CHAMPION_GROUPS)} groups, {distinct / len(chosen):.1f} distinct "
            f"({1 - distinct / total:.1%} of the decode is the same file read twice)"
        ]
        priced = []
        for spec in arguments.extra:
            slices, low, high = spec.split(":")
            candidate = (int(slices), (float(low), float(high)))
            joined, _ = DecodeBench._union_all(chosen, (*CHAMPION_GROUPS, candidate))  # noqa: SLF001 - its own CLI
            priced.append((joined - distinct, spec))
        for marginal, spec in sorted(priced):
            lines.append(
                f"candidate {spec:<16s} + {marginal / len(chosen):5.1f} files/study "
                f"({marginal / max(distinct, 1):+.1%} over the union)"
            )
        command.report(lines)
        return

    os.environ[SLICES] = str(arguments.slices)
    bench = DecodeBench()
    work = BenchStudy.headers if arguments.headers_only else BenchStudy.decoded
    measured = bench.timings(work, [BenchStudy(directory) for directory in chosen])
    command.report(["headers only, no pixels" if arguments.headers_only else f"{arguments.slices} slices"])
    command.report(bench.report(measured, len(chosen)))


if __name__ == "__main__":
    main()
