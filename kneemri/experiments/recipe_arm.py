"""An arm that is a one-line shim over `experiments/run.sh <recipe>.toml`, read as the arm it stands for.

Every header reader in this package reads an arm's SHELL for its `# bead:`, `# reading:`, `run_train` and
`--out` lines. A recipe arm moved all four into its TOML, so its shell names none of them, and each
reader would report the arm as declaring nothing. This is the one place that follows the shim: the text
is the shell, then the recipe, then the `run_train` line `run.sh` would execute — so a reader written
against arms keeps reading recipe arms without knowing they exist.
"""

import re
from pathlib import Path

from kneemri.training.recipe_layers import RecipeLayers

RUNNER = re.compile(r"run\.sh\"?\s+(?P<recipe>\S+\.toml)")
"""The recipe a shim hands to `experiments/run.sh`, relative to the repository root."""

ROOT_DEPTH = 2
"""An arm lives at `<root>/experiments/arms/<arm>.sh`, so its root is two directories above its own."""


class RecipeArm:
    """The text an arm stands for, following a shim into the recipe it runs."""

    @staticmethod
    def name(recipe: Path) -> str:
        """The run a recipe trains under — its file stem, as `run.sh` names it; the directories only group it."""
        return recipe.stem

    @staticmethod
    def text(arm: Path) -> str:
        """The arm's shell text, followed by each recipe it hands `run.sh` and the invocation made, in order."""
        shell = arm.read_text(encoding="utf-8", errors="replace")
        root = arm.resolve().parents[ROOT_DEPTH]
        return "\n".join([shell, *(RecipeArm._followed(root, found["recipe"]) for found in RUNNER.finditer(shell))])

    @staticmethod
    def _followed(root: Path, relative: str) -> str:
        recipe = root / relative
        name = RecipeArm.name(recipe)
        out = RecipeLayers((recipe,)).meta().out or name
        invocation = f'run_train {name} --config {relative} --out "$MODELS/{out}.pt"'
        return "\n".join((recipe.read_text(encoding="utf-8", errors="replace"), invocation))
