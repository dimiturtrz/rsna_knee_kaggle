"""Every recipe and what became of it, or one recipe's resolved options: `python -m kneemri.run`.

`--list` reads each `experiments/recipes/<family>/<variant>.toml` for its `[meta]` and joins the run it
names in `experiments/runs.tsv`, so the question, its status and its number sit on one line instead of
in a shell header, a bead and a table nobody joins. `--dry <recipe>` prints every training option the
recipe resolves to beside the layer that set it — a file of the `extends` chain, or the tree's default —
which is the question a stacked recipe makes hard to answer by reading.
"""

import csv
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from core import cli
from kneemri.experiments.default_drift import DefaultDrift
from kneemri.experiments.recipe_arm import RecipeArm
from kneemri.experiments.runs_table import TABLE
from kneemri.training.recipe_layers import RecipeLayers, RecipeMeta, RecipeStatus

logger = logging.getLogger(__name__)

RECIPES = Path("experiments/recipes")
LEAF_GLOB = "*/*.toml"
SHARED_PREFIX = "_"
TRUNKS = "trunks"
UNRUN = "-"
SEPARATOR = "\t"
RUN_KEY = "name"
SCORE = "best_auc"
EPOCHS = "epochs"
LIST_HEAD = ("recipe", "status", "bead", SCORE, EPOCHS, "hypothesis")
DRY_HEAD = ("option", "value", "set by")


@dataclass(frozen=True)
class RecipeTree:
    """The recipe leaves under one directory: `<family>/<variant>.toml`, never a family file or a trunk."""

    root: Path = RECIPES

    def leaves(self) -> list[Path]:
        """Every runnable recipe, in path order."""
        found = self.root.glob(LEAF_GLOB)
        return sorted(path for path in found if not path.name.startswith(SHARED_PREFIX) and path.parent.name != TRUNKS)

    @staticmethod
    def runs(table: Path) -> dict[str, dict[str, str]]:
        """The runs table keyed by run name; empty where no table has been generated."""
        if not table.exists():
            return {}
        with table.open(encoding="utf-8", newline="") as handle:
            return {row[RUN_KEY]: row for row in csv.DictReader(handle, delimiter=SEPARATOR)}

    def rows(self, runs: Mapping[str, Mapping[str, str]], status: str | None, family: str | None) -> list[str]:
        """One line per recipe matching the filters, its meta beside the run its `out` names."""
        stated = ((leaf, RecipeLayers((leaf,)).meta()) for leaf in self.leaves())
        kept = (
            (leaf, meta)
            for leaf, meta in stated
            if status in (None, meta.status) and family in (None, leaf.parent.name)
        )
        return [SEPARATOR.join(LIST_HEAD), *(self._row(leaf, meta, runs) for leaf, meta in kept)]

    @staticmethod
    def _row(leaf: Path, meta: RecipeMeta, runs: Mapping[str, Mapping[str, str]]) -> str:
        run = runs.get(meta.out or RecipeArm.name(leaf), {})
        cells = (
            RecipeArm.name(leaf),
            meta.status,
            meta.bead,
            run.get(SCORE, UNRUN),
            run.get(EPOCHS, UNRUN),
            meta.hypothesis,
        )
        return SEPARATOR.join(cells)

    @staticmethod
    def dry(leaf: Path) -> list[str]:
        """Every option the recipe resolves to, its value, and the layer that set it last."""
        lines = (SEPARATOR.join(row) for row in DefaultDrift.provenance(RecipeLayers((leaf,))))
        return [SEPARATOR.join(DRY_HEAD), *lines]


def main() -> None:
    """List every recipe with its status and result, or resolve one recipe's options."""
    command = cli.Command(__doc__, logger)
    mode = command.parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="every recipe, its meta and its run")
    mode.add_argument("--dry", type=Path, help="one recipe's resolved options and where each was set")
    command.parser.add_argument("--status", choices=[status.value for status in RecipeStatus])
    command.parser.add_argument("--family", help="only recipes under this family directory")
    command.parser.add_argument("--recipes", type=Path, default=RECIPES, help="the recipe tree")
    command.parser.add_argument("--runs", type=Path, default=TABLE, help="the generated runs table")
    arguments = command.parsed()
    tree = RecipeTree(arguments.recipes)
    runs = tree.runs(arguments.runs)
    lines = tree.dry(arguments.dry) if arguments.dry else tree.rows(runs, arguments.status, arguments.family)
    command.report(lines)


if __name__ == "__main__":
    main()
