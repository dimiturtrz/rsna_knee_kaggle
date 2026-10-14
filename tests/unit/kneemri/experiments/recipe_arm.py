from pathlib import Path

from kneemri.experiments.recipe_arm import RecipeArm


def _shim(root: Path, recipe: str) -> Path:
    arm = root / "experiments" / "arms" / "ctl_fast.sh"
    arm.parent.mkdir(parents=True)
    arm.write_text(f'#!/usr/bin/env bash\nexec bash "$(dirname "$0")/../run.sh" {recipe} "$@"\n', encoding="utf-8")
    return arm


def test_name():
    """A recipe trains under its file stem, the name `run.sh` gives it; the directories only group it."""
    assert RecipeArm.name(Path("experiments/recipes/control/ctl/ctl_fastbb.toml")) == "ctl_fastbb"


def test_text(tmp_path: Path):
    """A shim reads as itself, its recipe, and the invocation `run.sh` makes with the recipe's `out`."""
    recipe = tmp_path / "experiments" / "recipes" / "control" / "ctl" / "ctl_fast.toml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text('# bead: knee-abc\n[meta]\nout = "ctl_fast_s2026"\n', encoding="utf-8")
    arm = _shim(tmp_path, "experiments/recipes/control/ctl/ctl_fast.toml")

    text = RecipeArm.text(arm)

    assert "# bead: knee-abc" in text
    recipe = "experiments/recipes/control/ctl/ctl_fast.toml"
    invocation = f'run_train ctl_fast --config {recipe} --out "$MODELS/ctl_fast_s2026.pt"'
    assert text.endswith(invocation)


def test_text_of_a_shell_arm_is_the_shell(tmp_path: Path):
    """An arm that names no recipe reads as exactly its own text."""
    arm = tmp_path / "plain.sh"
    arm.write_text("# bead: knee-xyz\nrun_train plain --out x.pt\n", encoding="utf-8")

    assert RecipeArm.text(arm) == "# bead: knee-xyz\nrun_train plain --out x.pt\n"


def test_text_of_a_chained_shim_follows_every_recipe(tmp_path: Path):
    """A shim running two recipes in sequence reads as both, each with its own invocation, in order."""
    recipes = tmp_path / "experiments" / "recipes" / "control" / "ctl"
    recipes.mkdir(parents=True)
    (recipes / "head.toml").write_text("# bead: knee-one\n", encoding="utf-8")
    (recipes / "full.toml").write_text('[meta]\nout = "full_s2026"\n', encoding="utf-8")
    arm = tmp_path / "experiments" / "arms" / "staged.sh"
    arm.parent.mkdir(parents=True)
    runner = 'bash "$(dirname "$0")/../run.sh"'
    arm.write_text(
        f'{runner} experiments/recipes/control/ctl/head.toml "$@" && exec {runner} '
        'experiments/recipes/control/ctl/full.toml "$@"\n',
        encoding="utf-8",
    )

    text = RecipeArm.text(arm)

    assert "# bead: knee-one" in text
    head = text.index('run_train head --config experiments/recipes/control/ctl/head.toml --out "$MODELS/head.pt"')
    full = text.index('run_train full --config experiments/recipes/control/ctl/full.toml --out "$MODELS/full_s2026.pt"')
    assert head < full, "the stages read in the order the shim runs them"


def test_text_without_an_out_names_the_run(tmp_path: Path):
    """A recipe whose `[meta]` states no `out` writes under its run name, as `run.sh` does."""
    recipe = tmp_path / "experiments" / "recipes" / "control" / "ctl" / "ctl_fast.toml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("[optim]\nepochs = 4\n", encoding="utf-8")

    text = RecipeArm.text(_shim(tmp_path, "experiments/recipes/control/ctl/ctl_fast.toml"))

    assert text.endswith('--out "$MODELS/ctl_fast.pt"')
