"""Give a metrics file written before recipes existed the recipe its own checkpoint already carries.

Every run since `member_record` landed stamps its options into the checkpoint, and every run since the
recorder landed writes its scores to a JSONL beside it — but until now the two halves never met, so a
per-label comparison could not say whether its control was matched. The knowledge exists on disk; this
moves it to where the comparison reads it, for the runs that predate `RunMetrics.describe`.

`python -m kneemri.runs.metrics_recipe` fills in what it can and names what it cannot; `--check` writes
nothing and reports.
"""

import argparse
import logging
from pathlib import Path

from core.paths import DataRoot
from kneemri.runs.recorded_run import RecordedRun
from kneemri.runs.run_metrics import RunMetrics

logger = logging.getLogger(__name__)

METRICS = "*.metrics.jsonl"


class MetricsRecipe:
    """The recipe a run recorded in its checkpoint, moved beside the scores it produced."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def backfill(self, *, write: bool) -> tuple[list[str], list[str]]:
        """Every metrics file with no recipe, split into the ones recovered and the ones that cannot be.

        THIS COMMAND IS NO LONGER WHAT MAKES A RUN READABLE — `RecordedRun` answers from the checkpoint
        stamp directly, so a run whose recipe lives only there compares as matched without being touched.
        What it still buys is a metrics file that states its own recipe: greppable, diffable, and free of
        a `torch.load` per read. It reaches only runs that HAVE a metrics file, which is why it could
        never have fixed `wsfold0_e6` — that run has none, and the fix had to be in the reader.
        """
        filled: list[str] = []
        orphaned: list[str] = []
        for path in sorted(self.root.glob(METRICS)):
            metrics = RunMetrics(path)
            if metrics.recipe() is not None:
                continue
            recipe = RecordedRun.of(metrics).recipe
            if recipe is None:
                orphaned.append(path.name)
                continue
            filled.append(path.name)
            if write:
                metrics.describe(recipe)
        return filled, orphaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--check", action="store_true", help="write nothing; report what is missing")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = DataRoot.from_config(arguments.paths).processed()
    filled, orphaned = MetricsRecipe(root).backfill(write=not arguments.check)
    logger.info("%d recipe(s) %s: %s", len(filled), "recoverable" if arguments.check else "recovered", filled)
    if orphaned:
        # Named rather than counted: a run whose checkpoint is gone can never be a matched control
        # again, and that is a fact about the ledger rather than a failure of this tool.
        logger.info("%d run(s) with no checkpoint to read a recipe from: %s", len(orphaned), orphaned)


if __name__ == "__main__":
    main()
