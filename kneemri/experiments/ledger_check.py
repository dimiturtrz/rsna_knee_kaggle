"""Every run that finished should appear in the ledger: `python -m kneemri.experiments.ledger_check`.

The gates catch defects in code. NOTHING CHECKS A DOCUMENT, and on 2026-08-13 that turned out to be the
larger exposure: five files were carrying conclusions that later measurement had overturned, and the
ledger itself had five arms sitting as "queued" that had run two days earlier — `slice_attention` at
0.8473, `extended_slots` at 0.8459, `intensity_augment`, `dinov3`, `calibrated_targets`. An arm that
runs and is never written down is an arm paid for and thrown away, and the next planning pass reads its
row as unstarted work.

WHAT IT CHECKS, and deliberately only this: every `*.metrics.jsonl` on disk has its best score quoted
somewhere in the ledger. It matches on the NUMBER rather than the arm name, because the table's names
are prose (`whole_stack`, `resolution`) while the files are outputs (`wholestack_s2026`,
`res392_ws_s2026`) and a mapping between them is one more thing to get wrong.

WHAT IT CANNOT CHECK: whether the surrounding sentence is true. A number can be present and its
interpretation stale — which is the other half of what went wrong that day, and needs a reader.

WHAT COUNTS AS A FINISHED RUN: a checkpoint, not a metrics file. Scanning metrics files alone reported
"15 finished runs, 0 unrecorded" while 69 checkpoints sat beside them — a pass computed over 15 of 69,
which is the same wrong-denominator mistake in the check that exists to catch it.

WHAT IS NOT A FINISHED RUN: a smoke. It is recognised by its RECIPE (`--studies` capped) and not only by
its name, because the smoke this project documents takes no `--out` and lands on the default
`member_fold0.pt` — so a routine pre-flight left this check exiting 1 on a clean repo, and a check that
fails on correct behaviour is one the reader learns to skip.
"""

import argparse
import logging
from pathlib import Path

import torch

from kneemri.denominator import Denominator
from kneemri.runs.recorded_run import RecordedRun
from kneemri.runs.run_metrics import METRICS_SUFFIX, RunMetrics

logger = logging.getLogger(__name__)

LEDGER = Path("experiments/README.md")
INTERPRETATIONS = Path("interpretations")
UNQUOTABLE = ("throughput", "smoke")
"""Name fragments of runs whose scores no ledger should quote.

Speed probes measure wall-clock, not accuracy — and a run cannot state in its recipe that it was one,
so this list is the only place that knows. A smoke names itself here too when it was given an `--out`
that says so, which is where the whole exemption used to live; `truncated` is the half that catches the
ones whose name says nothing.
"""


class LedgerCheck:
    """Runs on disk, against the scores the written record quotes.

    The record is the ledger AND the interpretations, because the question is whether a run was written
    down anywhere findable, not whether it reached one particular table. Twenty-four runs live only in
    an interpretation; reporting those as missing would bury the eleven that are written down nowhere.
    """

    def __init__(self, *documents: Path) -> None:
        self.text = "\n".join(d.read_text(encoding="utf-8") for d in documents if d.exists())

    def best(self, metrics: Path) -> float | None:
        """The best holdout mean AUC a run recorded, or None if it never scored an epoch.

        Through `RunMetrics` rather than parsing the JSONL again, because that file now holds two row
        shapes — epochs and the recipe line — and a second reader is a second place to learn that. It
        also drops NaN, which a hand-rolled `is not None` does not: a final fit (`--fold -1`) records
        NaN by design, and `max()` over it can return NaN, quoted as "nan" and reported UNRECORDED for
        as long as the run exists.
        """
        best = RunMetrics(metrics).best()
        return best["auc"] if best else None

    def finished(self, processed: Path) -> dict[str, float]:
        """Every run on disk that scored, by name, from its metrics file or else its checkpoint.

        A metrics file wins where both exist: it holds every epoch, so it can say which was best, while
        a checkpoint holds the one score belonging to the weights that were kept. A run that finished
        before the metrics recorder existed, or whose file was never written, has only the checkpoint —
        and 50 of the 69 runs here are in exactly that state.
        """
        scores: dict[str, float] = {}
        for checkpoint in sorted(processed.glob("*.pt")):
            holdout = torch.load(checkpoint, map_location="cpu", weights_only=True).get("holdout")
            if holdout is not None and float(holdout) == float(holdout):
                scores[checkpoint.stem] = float(holdout)
        for metrics in sorted(processed.glob("*.metrics.jsonl")):
            score = self.best(metrics)
            if score is not None:
                scores[metrics.name.removesuffix(".metrics.jsonl")] = score
        return {
            name: score
            for name, score in scores.items()
            if not any(fragment in name for fragment in UNQUOTABLE)
            and not self.truncated(RunMetrics(processed / f"{name}.pt"))
        }

    def truncated(self, run: RunMetrics) -> bool:
        """Whether the run capped the corpus it touched — a SMOKE BY CONSTRUCTION, not by name.

        The name test alone fails on exactly the smoke this project documents: `--studies 60 --epochs 1`
        with no `--out` writes the default `member_fold0.pt`, whose name carries no fragment to match, so
        every routine pre-flight left the ledger check exiting 1 on a clean repo. A check that fails on
        its own documented correct behaviour teaches the reader to skip its output — which is the exact
        failure it exists to prevent, and it had already started happening.

        Read through `RecordedRun` because the field can be in either record, and because a run predating
        the recipe records no `studies` and must count as a REAL run: an unrecorded recipe is not grounds
        to drop a score from the ledger's denominator.
        """
        recipe = RecordedRun.of(run).recipe
        return recipe is not None and recipe.get("studies", "0") not in ("0", "")

    def reran(self, processed: Path, name: str) -> int:
        """How many EARLIER runs wrote to this name, counted from the rotations left beside it.

        A re-run to the same `--out` rotates the metrics file (`RunMetrics.begin`, for the `slices16`
        mix-up), and since 2026-08-19 rotates the checkpoint and the `.holdout.npz` with it under
        `<run>_run<n>`. Before that fix nothing moved the artefacts, so the epochs of both runs survived
        and the weights and per-study predictions of only the LAST one did, and a ledger row quoting the
        first run's best had no file behind it any more.

        That is not hypothetical: on 2026-08-19 `depth_regions_span` re-entered `run_train` mid-flight
        and trained a second time under its own name. Its row quotes 0.8505 from run 1; the checkpoint
        and predictions on disk are run 2's. Nothing on disk said so, and the shape it presents to this
        check is a run reported UNRECORDED for a score the ledger plainly contains — a mystery rather
        than a cause. The rotation is still worth reporting for the runs written before the fix, where a
        rotated `.metrics.<n>.jsonl` with no `_run<n>.pt` beside it names exactly what was lost.

        Counted rather than scored, deliberately. Reading the rotated file's own best would mean
        teaching a second reader the rotation naming, and the actionable fact needs no number: this
        name trained more than once, so the artefacts belong to the last of them.
        """
        return len(list(processed.glob(f"{name}{METRICS_SUFFIX}.*.jsonl")))

    def unrecorded(self, runs: dict[str, float]) -> dict[str, float]:
        """The runs whose best score appears nowhere in the written record.

        Rounded to four places because that is how the record quotes them, and a run whose score is
        written to more digits elsewhere is still recorded.
        """
        return {name: score for name, score in runs.items() if f"{score:.4f}" not in self.text}


def main() -> None:
    """Report finished runs the written record does not mention, and exit non-zero if any."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    parser.add_argument("--interpretations", type=Path, default=INTERPRETATIONS)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from core.paths import DataRoot  # noqa: PLC0415 - CLI-only, and this module is imported as a library

    check = LedgerCheck(arguments.ledger, *sorted(arguments.interpretations.rglob("*.md")))
    processed = DataRoot.from_config(arguments.paths).processed()
    runs = check.finished(processed)
    # The data root is machine-local and gitignored, so an unmounted drive or a paths.yaml pointing at
    # the wrong dataset produces "0 finished runs, 0 unrecorded" — the sentence a fully recorded ledger
    # produces, from a check that read no run at all.
    Denominator(len(runs), f"no finished run under {processed}").refuse_empty()

    missing = check.unrecorded(runs)
    for name, score in missing.items():
        logger.info(
            "UNRECORDED %s: best %.4f appears in neither %s nor %s",
            name,
            score,
            arguments.ledger,
            arguments.interpretations,
        )
        # A re-run is the likeliest innocent cause, and it is the one the reader cannot see from disk.
        # Said here rather than as its own failure because running a recipe twice is legitimate — what is
        # not legitimate is the number changing without a reason.
        earlier = check.reran(processed, name)
        if earlier:
            logger.info(
                "  ... and it trained %d time(s) before under this name: look for its weights and "
                "predictions as %s_run<n>, and if they are absent this run predates the artefact "
                "rotation and they were overwritten",
                earlier,
                name,
            )
    logger.info("%d finished runs, %d unrecorded", len(runs), len(missing))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
