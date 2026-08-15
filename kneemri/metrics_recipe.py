"""Give a metrics file written before recipes existed the recipe its own checkpoint already carries.

Every run since `member_record` landed stamps its options into the checkpoint, and every run since the
recorder landed writes its scores to a JSONL beside it — but until now the two halves never met, so a
per-label comparison could not say whether its control was matched. The knowledge exists on disk; this
moves it to where the comparison reads it, for the runs that predate `RunMetrics.describe`.

`python -m kneemri.metrics_recipe` fills in what it can and names what it cannot; `--check` writes
nothing and reports.
"""

import argparse
import logging
from pathlib import Path

import torch

from core.paths import DataRoot
from kneemri.run_metrics import RunMetrics

logger = logging.getLogger(__name__)

METRICS = "*.metrics.jsonl"


class MetricsRecipe:
    """The recipe a run recorded in its checkpoint, moved beside the scores it produced."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @staticmethod
    def checkpoint_of(metrics: Path) -> Path:
        """The member these scores describe. `.metrics.jsonl` is two suffixes, so `with_suffix` cannot."""
        return metrics.with_name(metrics.name.removesuffix(".metrics.jsonl") + ".pt")

    @staticmethod
    def recipe_of(checkpoint: Path) -> dict[str, str] | None:
        """The options block the checkpoint recorded, or None when there is nothing to read it from.

        `weights_only=True` because this opens files only to read a dictionary of strings out of them,
        and the weights beside it are hundreds of megabytes nobody here wants materialised.
        """
        if not checkpoint.exists():
            return None
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        options = payload.get("options")
        return {name: str(value) for name, value in options.items()} if options else None

    def backfill(self, *, write: bool) -> tuple[list[str], list[str]]:
        """Every metrics file with no recipe, split into the ones recovered and the ones that cannot be."""
        filled: list[str] = []
        orphaned: list[str] = []
        for path in sorted(self.root.glob(METRICS)):
            metrics = RunMetrics(path)
            if metrics.recipe() is not None:
                continue
            recipe = self.recipe_of(self.checkpoint_of(path))
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
