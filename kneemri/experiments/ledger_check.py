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
"""

import argparse
import logging
from pathlib import Path

import torch

from kneemri.runs.run_metrics import RunMetrics

logger = logging.getLogger(__name__)

LEDGER = Path("experiments/README.md")
INTERPRETATIONS = Path("interpretations")
UNQUOTABLE = ("throughput", "smoke")
"""Name fragments of runs whose scores no ledger should quote.

Speed probes measure wall-clock, not accuracy. A smoke is the arm's own pre-flight — `--studies 60
--epochs 1`, 48 train / 12 holdout — so its score is a number produced by a deliberately broken
denominator, and `smoke_v3base` at 0.6562 was reported UNRECORDED for as long as it sat on disk. The
only ways to clear that report are to delete the smoke, which throws away the evidence that the arm
was smoked, or to quote 0.6562 in the ledger, which puts a 60-study number in a table of 3,000-study
ones. Both are worse than naming the class here.
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
        return {name: score for name, score in scores.items() if not any(fragment in name for fragment in UNQUOTABLE)}

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
    runs = check.finished(DataRoot.from_config(arguments.paths).processed())

    missing = check.unrecorded(runs)
    for name, score in missing.items():
        logger.info(
            "UNRECORDED %s: best %.4f appears in neither %s nor %s",
            name,
            score,
            arguments.ledger,
            arguments.interpretations,
        )
    logger.info("%d finished runs, %d unrecorded", len(runs), len(missing))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
