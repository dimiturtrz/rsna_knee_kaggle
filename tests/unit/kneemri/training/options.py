from pathlib import Path

import pytest

from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.crop_centre import Crop
from core.data.corpus.folds import FitPositions
from core.data.corpus.slot_reader import Band, Reading
from core.data.corpus.slots import Routing
from core.data.labels.targets import CONFIDENCE_FLOOR, Supervision
from kneemri.training.flag_vocabulary import FlagVocabulary
from kneemri.training.options import TrainingOptions


def test_parse(monkeypatch: pytest.MonkeyPatch):
    """Flags become a typed options object, so a run's configuration is inspectable without argv."""
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--labels", "labels.csv", "--fold", "3", "--epochs", "30", "--one-cycle", "--intensity-aug"],
    )
    options = TrainingOptions.parse()
    assert options.fold == 3
    assert options.epochs == 30
    assert options.one_cycle
    assert options.intensity_aug
    assert options.labels == Path("labels.csv")
    assert options.eval_labels is None, "no ruler given means score on what you trained on"


def test_recipe():
    """Every flag, as strings — the account `train` logs at second zero and the checkpoint stamps later.

    Iterated from the dataclass rather than listed, which is the whole point: a flag added to a run
    reaches both the log line and the stamp without anyone remembering to add it here.
    """
    options = TrainingOptions(auc_margin=0.5, whole_stack=True, labels=Path("a.csv"))
    recipe = options.recipe

    assert recipe.keys() == vars(options).keys(), "every flag, not a chosen few"
    assert recipe["auc_margin"] == "0.5", "the value the process parsed, so a typo is visible in the log"
    assert recipe["whole_stack"] == "True"
    assert recipe["labels"] == "a.csv", "a Path as a string: a checkpoint read with weights_only cannot hold one"


def test_of_recipe():
    """A recorded recipe reads back as the run that wrote it — every field, at its declared type.

    Round-tripped against `recipe` rather than checked field by field, for the reason `test_recipe`
    iterates the dataclass: a flag added to a run must survive both directions without anyone
    remembering this test. The switches are what make it a real inverse — `bool("False")` is True, so a
    reader using the type's own constructor would arm every option a run deliberately left off.
    """
    options = TrainingOptions(
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

    assert TrainingOptions.of_recipe(options.recipe) == options, "every field, both ways"
    assert TrainingOptions.of_recipe({"band": "0.1 0.9"}) == TrainingOptions(), (
        "a flag this tree renamed is dropped, not raised on: the ledger spans the rename"
    )
    assert TrainingOptions.of_recipe({"slices": "20"}).batch == TrainingOptions().batch, (
        "and a field the recipe never mentions keeps the default the run that wrote it got"
    )


def test_parse_reads_a_separate_evaluation_ruler(monkeypatch: pytest.MonkeyPatch):
    """The ruler is separable, which is what makes two label arms comparable."""
    monkeypatch.setattr("sys.argv", ["train", "--labels", "a.csv", "--eval-labels", "b.csv"])
    options = TrainingOptions.parse()
    assert options.labels == Path("a.csv")
    assert options.eval_labels == Path("b.csv")


def test_supervision(monkeypatch: pytest.MonkeyPatch):
    """The two label switches must reach the targets, from argv, without `main` in the way.

    Both change what the model is FITTED ON while every recorded config keeps agreeing with itself —
    the shape of nine defects here already — and they map to a data argument rather than to a policy
    object, so the `options_to_policies` walk cannot see them.
    """
    monkeypatch.setattr("sys.argv", ["train", "--labels", "labels.csv", "--weight-floor", "--calibrated-targets"])
    asked = TrainingOptions.parse().supervision

    assert asked.floor == CONFIDENCE_FLOOR, "a finding the report never mentions still pulls with this"
    assert asked.calibrated
    assert TrainingOptions().supervision == Supervision(), "unasked-for, the run trains on the raw levels"


def test_parse_keeps_the_split_seed_independent_of_the_model_seed(monkeypatch: pytest.MonkeyPatch):
    """Changing the model seed must NOT move the fold split.

    This exists because it once did. `ReportFolds` was constructed with the training seed, so seed 7's
    fold-0 holdout shared 192 of 909 studies with seed 2026's — the two runs' holdout numbers were
    never comparable, and cross-scoring them evaluated a model on 717 studies it had trained on.
    """
    monkeypatch.setattr("sys.argv", ["train", "--labels", "a.csv", "--seed", "7"])
    options = TrainingOptions.parse()
    assert options.seed == 7
    assert options.split_seed == 2026, "the split is a property of the corpus, not of the run"


def test_parse_takes_every_default_from_the_dataclass(monkeypatch: pytest.MonkeyPatch):
    """A run given no flags must be the object every other test constructs — all 45 fields of it.

    THE DRIFT THIS PINS: each field's default is written twice, once here and once as the flag's, and
    nothing has ever compared them. A parser default that moves is silent in both directions — the CLI
    fits a recipe no test checks, and the tests prove a recipe no arm runs.
    """
    monkeypatch.setattr("sys.argv", ["train", "--labels", "labels.csv"])

    parsed = TrainingOptions.parse()

    assert parsed == TrainingOptions(labels=Path("labels.csv"))


def test_parse_compiles_the_encoder_unless_the_run_asks_for_eager(monkeypatch: pytest.MonkeyPatch):
    """Compilation is what a run gets for saying nothing, and `--eager-encoder` is the only way out.

    THE OMISSION THIS CLOSES: `--compile-encoder` was opt-in, and twenty-two of thirty-six accepted arms
    never named it — 21% of every step on the bulk of the arithmetic, lost to a flag nobody typed rather
    than to a choice anyone made. A default that must be asked for is a default in name only.
    """
    monkeypatch.setattr("sys.argv", ["train", "--labels", "labels.csv"])
    assert TrainingOptions.parse().compile_encoder, "a run that says nothing about speed still gets it"

    monkeypatch.setattr("sys.argv", ["train", "--labels", "labels.csv", "--eager-encoder"])
    assert not TrainingOptions.parse().compile_encoder, "the throughput control can still measure eager"


def test_crop():
    """The two flags argparse must land as scalars, read back as the one value everything else wants."""
    assert TrainingOptions().crop == Crop()
    assert TrainingOptions(crop_mm=80.0, centre="tissue").crop == Crop(mm=80.0, centre="tissue")


def test_band():
    """The band flags, read back as the rule that names the cache and is stamped on the member."""
    assert TrainingOptions().band == Band(), "saying nothing is the reading every member on disk has"
    assert TrainingOptions(band_low=0.1, band_high=0.9).band == Band(0.1, 0.9)


def test_reading():
    """The geometry a run decodes under is BOTH axes, because the cache it selects is named by both."""
    stated = TrainingOptions(crop_mm=80.0, centre="tissue", band_low=0.1, band_high=0.9)

    assert TrainingOptions().reading == Reading(), "the default reading is the historic cache"
    assert stated.reading == Reading(crop=Crop(mm=80.0, centre="tissue"), band=Band(0.1, 0.9))


def test_routing():
    """The scheme and tie flags argparse lands as scalars, read back as the pair everything else wants."""
    assert TrainingOptions().routing == Routing()
    assert TrainingOptions(tie="last").routing.tie == "last"


def test_cache():
    """The arrays a run reads, resolved from the run — not respelled by each entrypoint that wants them.

    `main` and `arm_check` each held their own six-axis `StudyCache(...)`, agreeing with `CacheFlags`
    by inspection. A seventh axis added to the flags and forgotten in one of them is not a crash: it is
    an arm that trains on pixels its recorded config does not describe, and a holdout number filed
    under a reading nobody ran. The key is asserted whole, because the key IS the file name.
    """
    stated = TrainingOptions(
        slices=6, image_size=448, crop_mm=80.0, centre="tissue", band_low=0.1, band_high=0.9, tie="last"
    )
    named = stated.cache(Path("."))

    assert named.key == "448px_6sl_80mm_tissue_0.1-0.9band_last"
    assert named.directory == Path(".")
    assert named.reading == stated.reading, "the run's geometry, not the module default"
    assert named.routing == stated.routing
    assert TrainingOptions(cache_studies=60).cache(Path(".")).key.endswith("_first60"), (
        "the smoke truncation is an axis of the NAME, so a partial decode cannot overwrite the full one"
    )
    argv = FlagVocabulary.parser().parse_args(["--labels", "labels.csv"])

    assert TrainingOptions().cache(Path(".")).key == CacheFlags.cache(argv, Path(".")).key, (
        "the dataclass half and the namespace half of one rule, pinned equal on the reading every "
        "member on disk was fitted under"
    )


def test_positions():
    """The three flags that cut the fold's split, read back as the one value that knows the rules."""
    assert TrainingOptions().positions == FitPositions(), "saying nothing fits on the whole split"
    assert TrainingOptions(studies=50, train_fraction=0.5, split_seed=7).positions == FitPositions(
        studies=50, train_fraction=0.5, split_seed=7
    )
    assert TrainingOptions(seed=7).positions == FitPositions(), (
        "the MODEL seed does not move which studies exist, as with the fold — only --split-seed does"
    )


def test_of():
    """Flags come from a LIST, so a caller can price a recipe it is not itself running."""
    options = TrainingOptions.of(["--labels", "l.csv", "--slices", "12", "--image-size", "224"])

    assert options.slices == 12, "the list is read, not the process's own argv"
    assert options.image_size == 224
    assert str(options.labels) == "l.csv"
