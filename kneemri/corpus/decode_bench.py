"""Is decode I/O-bound or GIL-bound? `python -m kneemri.corpus.decode_bench --studies 96`.

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

import argparse
import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger(__name__)

STUDIES = 96
"""Enough that a pool's startup is not the measurement. At 24 the Windows spawn cost alone flattered
threads and hid the process pool's win entirely."""

PLANES: dict[str, str] = {}
"""Filled per worker on first use. A process pool re-imports this module rather than inheriting state,
and the alternative — sending the table with every study — would measure pickling, not decoding."""


class DecodeBench:
    """One decode workload, timed under each way of running it."""

    def timings(self, work: Callable[[str], object], items: Sequence[str]) -> dict[str, float]:
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
    def _decode_one(directory: str) -> object:
        """One study through the real reader.

        A staticmethod rather than a module-level function because the house rule wants behaviour on
        the class that owns it — and a staticmethod still pickles by qualified name, which is what the
        child process needs. Same shape as `SlotTies._contested_of`, for the same two reasons.
        """
        global PLANES
        from core.paths import DataRoot  # noqa: PLC0415 - CLI-only
        from kneemri.corpus.study import SlotPipeline, Study  # noqa: PLC0415 - CLI-only
        from kneemri.corpus.study_cache import CorpusSplit  # noqa: PLC0415 - CLI-only

        if not PLANES:
            PLANES = CorpusSplit(DataRoot.from_config(Path("paths.yaml")).raw()).planes()
        return Study.from_directory(Path(directory), PLANES, SlotPipeline()).bag(12)

    def report(self, measured: dict[str, float], studies: int) -> list[str]:
        """Each strategy against SERIAL, because the question is whether a pool earns its complexity."""
        baseline = measured.get("serial") or 1.0
        return [
            f"{label:14s} {seconds:6.1f}s   {seconds / max(studies, 1):.2f}s/study   {baseline / seconds:.2f}x"
            for label, seconds in measured.items()
        ]


def main() -> None:
    """Time the real decode under each strategy and print what each buys over running serially."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=int, default=STUDIES)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from core.paths import DataRoot  # noqa: PLC0415 - CLI-only
    from kneemri.corpus.study_cache import CorpusSplit  # noqa: PLC0415 - CLI-only

    source = CorpusSplit(DataRoot.from_config(arguments.paths).raw()).source
    studies = [str(path) for path in sorted(source.iterdir()) if path.is_dir()]
    chosen = studies[: arguments.studies]

    bench = DecodeBench()
    measured = bench.timings(DecodeBench._decode_one, chosen)  # noqa: SLF001 - its own CLI
    for line in bench.report(measured, len(chosen)):
        logger.info(line)


if __name__ == "__main__":
    main()
