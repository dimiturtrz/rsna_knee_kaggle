"""The tree is the one declaration of every knob; the parser, the file reader and the flat view all read it."""

import dataclasses
import json
from pathlib import Path

import pytest

from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.crop_centre import Crop
from core.data.corpus.folds import FitPositions
from core.data.corpus.slot_reader import Band, Reading
from core.data.corpus.slots import Routing
from core.data.labels.targets import CONFIDENCE_FLOOR, Supervision
from kneemri.training.augmentation import RegularisationConfig
from kneemri.training.config_tree import NOT_STATED, TrainConfig

ARM_RECIPES = Path(__file__).parent / "fixtures" / "arm_recipes.json"
REPO = Path(__file__).parents[4]


def test_leaves():
    """Every leaf is reached exactly once, and no node is mistaken for a leaf."""
    names = [leaf.name for leaf, _ in TrainConfig.leaves()]

    assert len(names) == len(set(names)), "two nodes declare one knob"
    assert "mixup" in names
    assert "regularisation" not in names


def test_parser():
    """Every leaf is reachable from argv, and every flag lands on a leaf.

    A leaf with no flag is a knob only a test can turn; a flag with no leaf would raise on `parse`.
    """
    dests = set(vars(TrainConfig.parser().parse_args(["--labels", "labels.csv"])))

    assert dests == {leaf.name for leaf, _ in TrainConfig.leaves()}


def test_parse(monkeypatch: pytest.MonkeyPatch):
    """Flags become a typed tree, so a run's configuration is inspectable without argv."""
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--labels", "labels.csv", "--fold", "3", "--epochs", "30", "--one-cycle", "--intensity-aug"],
    )
    options = TrainConfig.parse()
    assert options.run.fold == 3
    assert options.optim.epochs == 30
    assert options.optim.one_cycle
    assert options.regularisation.intensity_aug
    assert options.run.labels == Path("labels.csv")
    assert options.run.eval_labels is None, "no ruler given means score on what you trained on"


def _frozen_keys(recipe: dict[str, str], frozen: dict[str, str]) -> dict[str, str]:
    """The recipe on the keys the fixture froze, POSIX-separated; a field added since is not its business."""
    return {name: recipe[name].replace(chr(92), "/") for name in frozen}


def test_every_arm_parses_to_its_frozen_recipe(monkeypatch: pytest.MonkeyPatch):
    """Every run_train arm's line gives the recipe the flat parser gave it before the config tree.

    Frozen once from the OLD parser (knee-feb62.6 step 2), so a rewrite of the options surface is proven
    byte-identical over the real arm corpus rather than over the handful of cases a test author thought of.
    The hazard it exists for is an argparse default that disagrees with its dataclass default.
    """
    frozen = json.loads(ARM_RECIPES.read_text(encoding="utf-8"))
    monkeypatch.chdir(REPO)

    defaults = frozen["defaults"]
    assert _frozen_keys(TrainConfig().recipe, defaults) == defaults
    for name, arm in frozen["arms"].items():
        assert _frozen_keys(TrainConfig.of(arm["argv"]).recipe, defaults) == defaults | arm["recipe"], name


def test_parse_reads_a_separate_evaluation_ruler(monkeypatch: pytest.MonkeyPatch):
    """The ruler is separable, which is what makes two label arms comparable."""
    monkeypatch.setattr("sys.argv", ["train", "--labels", "a.csv", "--eval-labels", "b.csv"])
    options = TrainConfig.parse()
    assert options.run.labels == Path("a.csv")
    assert options.run.eval_labels == Path("b.csv")


def test_parse_keeps_the_split_seed_independent_of_the_model_seed(monkeypatch: pytest.MonkeyPatch):
    """Changing the model seed must NOT move the fold split.

    This exists because it once did. `ReportFolds` was constructed with the training seed, so seed 7's
    fold-0 holdout shared 192 of 909 studies with seed 2026's — the two runs' holdout numbers were
    never comparable, and cross-scoring them evaluated a model on 717 studies it had trained on.
    """
    monkeypatch.setattr("sys.argv", ["train", "--labels", "a.csv", "--seed", "7"])
    options = TrainConfig.parse()
    assert options.run.seed == 7
    assert options.run.split_seed == 2026, "the split is a property of the corpus, not of the run"


def test_parse_takes_every_default_from_the_tree(monkeypatch: pytest.MonkeyPatch):
    """A run given no flags must be the tree every other test constructs.

    THE DRIFT THIS PINS: a leaf's default and its flag's default were once written twice and never
    compared. A parser default that moves is silent in both directions — the CLI fits a recipe no test
    checks, and the tests prove a recipe no arm runs.
    """
    monkeypatch.setattr("sys.argv", ["train", "--labels", "labels.csv"])

    assert TrainConfig.parse() == TrainConfig().with_leaves(labels=Path("labels.csv"))


def test_parse_compiles_the_encoder_unless_the_run_asks_for_eager(monkeypatch: pytest.MonkeyPatch):
    """Compilation is what a run gets for saying nothing, and `--eager-encoder` is the only way out.

    THE OMISSION THIS CLOSES: `--compile-encoder` was opt-in, and twenty-two of thirty-six accepted arms
    never named it — 21% of every step on the bulk of the arithmetic, lost to a flag nobody typed rather
    than to a choice anyone made. A default that must be asked for is a default in name only.
    """
    monkeypatch.setattr("sys.argv", ["train", "--labels", "labels.csv"])
    assert TrainConfig.parse().run.compile_encoder, "a run that says nothing about speed still gets it"

    monkeypatch.setattr("sys.argv", ["train", "--labels", "labels.csv", "--eager-encoder"])
    assert not TrainConfig.parse().run.compile_encoder, "the throughput control can still measure eager"


def test_of(tmp_path: Path):
    """Flags come from a LIST and land in their node, and a flag on the line beats the recipe file."""
    recipe = tmp_path / "r.toml"
    recipe.write_text("epochs = 36\nema = true\nmixup = 0.2\n", encoding="utf-8")

    tree = TrainConfig.of(["--labels", "l.csv", "--config", str(recipe), "--epochs", "4", "--eager-encoder"])

    assert tree.optim.epochs == 4, "an explicit flag overrides the recipe file"
    assert tree.optim.ema is True
    assert tree.regularisation.mixup == pytest.approx(0.2)
    assert tree.run.compile_encoder is False, "a renamed flag still lands on its field"
    assert tree.run.config == recipe, "the file travels to the stamp beside the values it set"
    assert str(tree.run.labels) == "l.csv", "the list is read, not the process's own argv"


def test_of_flat():
    """A mapping names leaves; an unnamed leaf keeps its default."""
    tree = TrainConfig.of_flat({"mixup": 0.3, "epochs": 7})

    assert tree.regularisation == dataclasses.replace(RegularisationConfig(), mixup=0.3)
    assert tree.optim.epochs == 7


def test_with_leaves():
    """Named leaves change wherever they hang; every other leaf is kept, not reset to its default."""
    tree = TrainConfig().with_leaves(slices=20, mixup=0.1).with_leaves(epochs=5)

    assert tree.data.slices == 20
    assert tree.regularisation.mixup == pytest.approx(0.1)
    assert tree.optim.epochs == 5


def test_flat():
    """The flat view round-trips through `of_flat` and holds every leaf."""
    tree = TrainConfig.of_flat({"slices": 20, "variant": "base"})

    assert TrainConfig.of_flat(tree.flat) == tree
    assert tree.flat["slices"] == 20
    assert len(tree.flat) == len(list(TrainConfig.leaves()))


def test_recipe():
    """Every leaf, as strings — the account `train` logs at second zero and the checkpoint stamps later.

    Iterated from the tree rather than listed: a leaf added to a run reaches both the log line and the
    stamp without anyone remembering to add it here. FLAT, so checkpoints on disk keep reading.
    """
    options = TrainConfig().with_leaves(auc_margin=0.5, whole_stack=True, labels=Path("a.csv"))
    recipe = options.recipe

    assert recipe.keys() == options.flat.keys(), "every leaf, not a chosen few"
    assert recipe["auc_margin"] == "0.5", "the value the process parsed, so a typo is visible in the log"
    assert recipe["whole_stack"] == "True"
    assert recipe["labels"] == "a.csv", "a Path as a string: a checkpoint read with weights_only cannot hold one"


def test_stamp():
    """The stamp carries the tree's own values, so a checkpoint records the run and not a module default."""
    tree = TrainConfig().with_leaves(slices=20, crop_mm=80.0, band_low=0.1, band_high=0.9, ema=True, bf16=True)
    stamp = tree.stamp

    assert stamp.recipe == tree.recipe
    assert stamp.slices == 20
    assert stamp.crop == tree.data.crop
    assert stamp.band == tree.data.band
    assert stamp.ema
    assert stamp.bf16


def test_of_recipe():
    """A recorded recipe reads back as the run that wrote it — every leaf, at its declared type.

    The switches are what make it a real inverse — `bool("False")` is True, so a reader using the type's
    own constructor would arm every option a run deliberately left off.
    """
    options = TrainConfig().with_leaves(
        slices=20,
        batch=6,
        variant="v3base",
        one_cycle=True,
        ema=False,
        rank_weight=0.5,
        labels=Path("a.csv"),
        teacher_targets=None,
        out=Path("member.pt"),
    )

    assert TrainConfig.of_recipe(options.recipe) == options, "every leaf, both ways"
    assert TrainConfig.of_recipe({"band": "0.1 0.9"}) == TrainConfig(), (
        "a flag this tree renamed is dropped, not raised on: the ledger spans the rename"
    )
    assert TrainConfig.of_recipe({"slices": "20"}).optim.batch == TrainConfig().optim.batch, (
        "and a leaf the recipe never mentions keeps the default the run that wrote it got"
    )


def test_spelled(tmp_path: Path):
    """A recipe file's keys read as stated flags ahead of the line; a line with no file is unchanged."""
    recipe = tmp_path / "r.toml"
    recipe.write_text("band_low = 0.1\nslices = 20\n", encoding="utf-8")

    assert TrainConfig.spelled(["--config", str(recipe), "--ema"]) == [
        "--band-low",
        "--slices",
        "--config",
        str(recipe),
        "--ema",
    ]
    assert TrainConfig.spelled(["--ema"]) == ["--ema"]


def test_of_file(tmp_path: Path):
    """Values arrive typed as their fields; no file is no values; a misspelt key refuses."""
    recipe = tmp_path / "r.toml"
    recipe.write_text('warmup = 0.25\nslices = 20\npadding_mode = "reflection"\nunread_targets = "u.npz"\n', "utf-8")

    assert TrainConfig.of_file(recipe) == {
        "warmup": 0.25,
        "slices": 20,
        "padding_mode": "reflection",
        "unread_targets": Path("u.npz"),
    }
    assert TrainConfig.of_file(None) == {}
    recipe.write_text("stochastic_dept = 0.05\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stochastic_dept"):
        TrainConfig.of_file(recipe)


def test_typed():
    """Text reads as the declared type, an optional reads its stated half, and the unstated marker is None."""
    assert TrainConfig.typed(int, "3") == 3
    assert TrainConfig.typed(Path | None, "a.npz") == Path("a.npz")
    assert TrainConfig.typed(bool, "True") is True
    assert TrainConfig.typed(float | None, NOT_STATED) is None


def test_supervision(monkeypatch: pytest.MonkeyPatch):
    """The two label switches must reach the targets, from argv, without `main` in the way.

    Both change what the model is FITTED ON while every recorded config keeps agreeing with itself —
    the shape of nine defects here already — and they map to a data argument rather than to a policy
    object, so the `options_to_policies` walk cannot see them.
    """
    monkeypatch.setattr("sys.argv", ["train", "--labels", "labels.csv", "--weight-floor", "--calibrated-targets"])
    asked = TrainConfig.parse().targets.supervision

    assert asked.floor == CONFIDENCE_FLOOR, "a finding the report never mentions still pulls with this"
    assert asked.calibrated
    assert TrainConfig().targets.supervision == Supervision(), "unasked-for, the run trains on the raw levels"


def test_crop():
    """The two flags argparse must land as scalars, read back as the one value everything else wants."""
    assert TrainConfig().data.crop == Crop()
    assert TrainConfig().with_leaves(crop_mm=80.0, centre="tissue").data.crop == Crop(mm=80.0, centre="tissue")


def test_band():
    """The band flags, read back as the rule that names the cache and is stamped on the member."""
    assert TrainConfig().data.band == Band(), "saying nothing is the reading every member on disk has"
    assert TrainConfig().with_leaves(band_low=0.1, band_high=0.9).data.band == Band(0.1, 0.9)


def test_reading():
    """The geometry a run decodes under is BOTH axes, because the cache it selects is named by both."""
    stated = TrainConfig().with_leaves(crop_mm=80.0, centre="tissue", band_low=0.1, band_high=0.9)

    assert TrainConfig().data.reading == Reading(), "the default reading is the historic cache"
    assert stated.data.reading == Reading(crop=Crop(mm=80.0, centre="tissue"), band=Band(0.1, 0.9))


def test_routing():
    """The scheme and tie flags argparse lands as scalars, read back as the pair everything else wants."""
    assert TrainConfig().data.routing == Routing()
    assert TrainConfig().with_leaves(tie="last").data.routing.tie == "last"


def test_cache():
    """The arrays a run reads, resolved from the run — not respelled by each entrypoint that wants them.

    `main` and `arm_check` each held their own six-axis `StudyCache(...)`, agreeing with `CacheFlags`
    by inspection. A seventh axis added to the flags and forgotten in one of them is not a crash: it is
    an arm that trains on pixels its recorded config does not describe, and a holdout number filed
    under a reading nobody ran. The key is asserted whole, because the key IS the file name.
    """
    stated = TrainConfig().with_leaves(
        slices=6, image_size=448, crop_mm=80.0, centre="tissue", band_low=0.1, band_high=0.9, tie="last"
    )
    named = stated.data.cache(Path("."))

    assert named.key == "448px_6sl_80mm_tissue_0.1-0.9band_last"
    assert named.directory == Path(".")
    assert named.reading == stated.data.reading, "the run's geometry, not the module default"
    assert named.routing == stated.data.routing
    assert TrainConfig().with_leaves(cache_studies=60).data.cache(Path(".")).key.endswith("_first60"), (
        "the smoke truncation is an axis of the NAME, so a partial decode cannot overwrite the full one"
    )
    argv = TrainConfig.parser().parse_args(["--labels", "labels.csv"])

    assert TrainConfig().data.cache(Path(".")).key == CacheFlags.cache(argv, Path(".")).key, (
        "the tree half and the namespace half of one rule, pinned equal on the reading every "
        "member on disk was fitted under"
    )


def test_positions():
    """The three flags that cut the fold's split, read back as the one value that knows the rules."""
    assert TrainConfig().run.positions == FitPositions(), "saying nothing fits on the whole split"
    assert TrainConfig().with_leaves(studies=50, train_fraction=0.5, split_seed=7).run.positions == FitPositions(
        studies=50, train_fraction=0.5, split_seed=7
    )
    assert TrainConfig().with_leaves(seed=7).run.positions == FitPositions(), (
        "the MODEL seed does not move which studies exist, as with the fold — only --split-seed does"
    )
