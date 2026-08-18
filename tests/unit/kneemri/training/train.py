import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from kneemri.corpus.crop_centre import Crop
from kneemri.corpus.slot_reader import Band, Reading
from kneemri.corpus.slots import Routing
from kneemri.labels.targets import CONFIDENCE_FLOOR, Supervision
from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.models.model import StudyDesign
from kneemri.training.landed import Request
from kneemri.training.objective import Objective, RankedObjective
from kneemri.training.remembered_ranking import RememberedRanking
from kneemri.training.train import TrainingOptions
from kneemri.training.trainer import Adaptation


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


def test_parser():
    """Every field is reachable from argv, and every flag lands on a field.

    A field with no flag is a knob only a test can turn; a flag with no field would raise on `parse`,
    which is loud — the silent half is the first, and it is the half that queues an arm whose recipe
    cannot be asked for.
    """
    dests = set(vars(TrainingOptions.parser().parse_args(["--labels", "labels.csv"])))

    assert dests == {field.name for field in dataclasses.fields(TrainingOptions)}


def test_objective():
    """The flags choose the loss, and the default must stay plain weighted cross-entropy.

    Checked without a GPU, a corpus or a downloaded encoder on purpose: the ranking arm sits behind a
    queue, and a miswired objective would otherwise surface only after the card had been waited on.
    """
    assert type(TrainingOptions().objective()) is Objective
    assert not TrainingOptions().objective().normalise_by_count

    ranked = TrainingOptions(rank_weight=0.5).objective()
    assert isinstance(ranked, RankedObjective)
    assert ranked.weight == 0.5

    assert TrainingOptions(rank_weight=0.5, normalise_by_count=True).objective().normalise_by_count
    assert TrainingOptions(per_label_loss=True).objective().per_label, "a per-label loss is a different objective"
    assert TrainingOptions(rank_weight=0.5, per_label_loss=True).objective().per_label, "and it survives the ranking"


def test_objective_pair_memory():
    """A memory only exists to widen the ranking term, so asking for one without the term must REFUSE.

    Same reason the rest of `objective` is checked here: the failure this guards is an arm that trains the
    plain recipe under the memory arm's name and gets read as a refutation of the mechanism. A refusal at
    flag-parse time costs nothing; the silent no-op costs a queue slot and a wrong conclusion.
    """
    remembered = TrainingOptions(rank_weight=0.5, pair_memory=64).objective()
    assert isinstance(remembered, RememberedRanking)
    assert remembered.memory.remembered.maxlen == 64, "the flag must reach the pool it sizes"
    assert remembered.weight == 0.5

    with pytest.raises(ValueError, match="no ranking term"):
        TrainingOptions(pair_memory=64).objective()


def test_adaptation():
    """Which blocks move and how fast are reachable, and default to the inherited pair.

    Both were module constants with no flags - which is why the depth bracket could not distinguish
    "more blocks do not help" from "no block moved at this rate". They travel together because a depth
    result is uninterpretable without the rate it was measured at.
    """
    assert TrainingOptions().adaptation() == Adaptation()
    default = TrainingOptions().adaptation()
    assert round(default.head_lr / default.backbone_lr) == 125, "the inherited head:encoder ratio"

    faster = TrainingOptions(backbone_lr=1e-4, unfreeze_last=12).adaptation()
    assert faster.backbone_lr == 1e-4
    assert faster.unfreeze_last == 12
    assert faster.head_lr == Adaptation().head_lr, "raising the encoder rate must not move the head's"


def test_fitted_trainer(tmp_path: Path):
    """Every optional policy a run asked for is attached, and nothing it did not ask for is.

    This is the function that replaced a run of `if arguments.x: trainer.y()` in `main`, and the
    failure it guards against is silent: a flag that stops being wired does not raise, it trains a
    different recipe than the one on the command line and reports a plausible number for it.
    """
    import torch  # noqa: PLC0415 - the rest of this module needs no torch

    from kneemri.corpus.slot_reader import GROUP  # noqa: PLC0415
    from kneemri.models.ema import LiveWeights, WeightAverage  # noqa: PLC0415
    from kneemri.models.model import StudyDesign, StudyModel  # noqa: PLC0415
    from kneemri.training.report_supervision import NoReportSupervision, ReportSupervision  # noqa: PLC0415

    from ..models.model import DIM, SLOTS, _FakeBackbone  # noqa: PLC0415

    device = torch.device("cpu")
    plain = TrainingOptions().fitted_trainer(StudyModel(_FakeBackbone(), DIM, SLOTS), device, [], 120)
    assert plain.window == GROUP, "a default run reads one group, not the stack"
    assert plain.read_size is None
    assert plain.schedule is None
    assert plain.autocast_dtype is torch.bfloat16, "bf16 is the DEFAULT, and a precision is a numerics change"
    assert type(plain.average) is LiveWeights, "no EMA unless asked: it changes which weights are SAVED"
    assert type(plain.reports) is NoReportSupervision

    model = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max"))
    asked = TrainingOptions(whole_stack=True, slices=16, train_size=392, one_cycle=True, epochs=2, ema=True, bf16=False)
    trainer = asked.fitted_trainer(model, device, [], 120)
    assert trainer.window == 16
    assert trainer.read_size == 392
    assert trainer.schedule is not None, "one-cycle must reach the trainer, or the rate never anneals"
    assert type(trainer.average) is WeightAverage, "the EMA is what `evaluated()` scores and the checkpoint keeps"
    assert trainer.autocast_dtype is torch.float16, "--fp16 is the only way back, so it must land"
    assert trainer.scaler is not None, "fp16 has a scale that can collapse, so the scaler comes with it"

    import numpy as np  # noqa: PLC0415 - only this branch needs a vectors file on disk

    vectors = tmp_path / "reports.npz"
    np.savez(vectors, identifiers=np.array(["s0"]), vectors=np.zeros((1, 4), dtype=np.float32))
    supervised = TrainingOptions(report_weight=0.3, report_vectors=vectors).fitted_trainer(
        StudyModel(_FakeBackbone(), DIM, SLOTS), device, ["s0"], 120
    )
    attached = supervised.reports
    assert isinstance(attached, ReportSupervision), "the term the loop ADDS, so an unattached one is a plain run"
    assert attached.weight == 0.3


def test_reports(tmp_path: Path):
    """Report supervision is all-or-nothing: a weight with no vectors is refused, not skipped.

    Skipping it trained the PLAIN recipe under the report arm's name, so the arm would have read as a
    refutation of the mechanism when it refuted only the invocation.
    """
    import numpy as np  # noqa: PLC0415 - the rest of this module needs no numpy

    assert TrainingOptions().reports([]) is None, "no weight, no supervision, no file read"

    with pytest.raises(ValueError, match="--report-vectors"):
        TrainingOptions(report_weight=0.3).reports([])

    vectors = tmp_path / "reports.npz"
    np.savez(vectors, identifiers=np.array(["s0"]), vectors=np.zeros((1, 4), dtype=np.float32))
    supervision = TrainingOptions(report_weight=0.3, report_vectors=vectors).reports(["s0"])
    assert supervision is not None
    assert supervision.weight == 0.3


def test_fitted_model(monkeypatch: pytest.MonkeyPatch):
    """Adaptation happens BEFORE the wrappers, because both wrap whatever they are handed.

    Adapters injected after compilation would sit outside the compiled graph, and the run would report
    a speed and a score belonging to two different models. The order is the reason this method exists,
    so the order is what the test pins.
    """
    monkeypatch.setattr(
        "sys.argv",
        ["train", "--labels", "a.csv", "--lora-rank", "4", "--recompute-activations"],
    )
    options = TrainingOptions.parse()
    order: list[str] = []

    class _Model:
        def low_rank(self, rank: int) -> int:
            order.append(f"low_rank:{rank}")
            return 36

        def recompute_activations(self) -> None:
            order.append("recompute")

        def compile_encoder(self) -> None:
            order.append("compile")

    monkeypatch.setattr("kneemri.training.train.StudyModel", lambda *args, **kwargs: _Model())
    monkeypatch.setattr(
        "kneemri.training.train.AutoModel", type("_Auto", (), {"from_pretrained": staticmethod(lambda name: None)})
    )
    card = torch.device("cuda")
    options.fitted_model(ENCODERS[SMALL], 6, StudyDesign(), card)

    assert order == ["low_rank:4", "recompute", "compile"], "adapters go in BEFORE the graph is frozen around them"

    order.clear()
    TrainingOptions().fitted_model(ENCODERS[SMALL], 6, StudyDesign(), card)
    assert order == ["compile"], "a plain run is the pretrained encoder unwrapped, and compiled since it is free"

    order.clear()
    TrainingOptions(compile_encoder=False).fitted_model(ENCODERS[SMALL], 6, StudyDesign(), card)
    assert order == [], "and eager is reachable, for the control that has to measure what compiling buys"

    order.clear()
    TrainingOptions().fitted_model(ENCODERS[SMALL], 6, StudyDesign(), torch.device("cpu"))
    assert order == [], "and a CPU run is never compiled: inductor's gain is CUDA kernels, and the trace dies"


def _probe(default: object) -> object:
    """A value of the field's own type that is NOT its default, so a field that did not move is visible."""
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 7
    return float(default) + 0.5  # type: ignore[arg-type]


def test_requested():
    """Every flag reaches the request field it NAMES, and moves no other one.

    A one-hot walk rather than seventeen hand-written asserts, because the failure being guarded is a
    CROSSED WIRE: `weight_floor=self.calibrated_targets` reads a real flag into a real field and every
    type agrees, so the pre-flight check would then clear a run for a switch nobody asked for and refuse
    it for one it did — the check itself failing in the exact shape it exists to catch. Only moving one
    flag at a time and demanding that exactly one field follow can see that.
    """
    cpu = torch.device("cpu")
    shared = {field.name for field in dataclasses.fields(Request)} & {
        field.name for field in dataclasses.fields(TrainingOptions)
    }

    assert len(shared) == 14, "pinned, because a renamed flag would otherwise drop silently out of the walk"

    plain = TrainingOptions().requested(cpu)
    for name in sorted(shared):
        probe = _probe(getattr(TrainingOptions(), name))
        moved = dataclasses.replace(TrainingOptions(), **{name: probe}).requested(cpu)

        assert getattr(moved, name) == probe, f"--{name} does not reach the request field of its own name"
        followed = {other for other in shared if getattr(moved, other) != getattr(plain, other)}
        assert followed == {name}, f"--{name} also moved {followed - {name}}"


def test_requested_resolves_the_three_fields_that_are_not_a_flag():
    """Two are an argument's PRESENCE, and compiling is decided by the DEVICE rather than by the flag.

    The device half is the one that bites on this machine: `fitted_model` compiles on CUDA only, so a
    request built from the flag alone would report an un-landed compile on every CPU smoke run.
    """
    asked = TrainingOptions(recompute_activations=True, report_vectors=Path("vectors.npz")).requested(
        torch.device("cpu")
    )

    assert asked.recompute is True
    assert asked.reports is True, "the VECTORS decide it — a weight with no vectors is refused earlier"
    assert TrainingOptions().requested(torch.device("cpu")).reports is False
    assert TrainingOptions(compile_encoder=True).requested(torch.device("cpu")).compiling is False
    assert TrainingOptions(compile_encoder=True).requested(torch.device("cuda")).compiling is True
    assert TrainingOptions(compile_encoder=False).requested(torch.device("cuda")).compiling is False


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


def test_sampled():
    """A cap that keeps the fold's ratio, and NOTHING at all when no cap is asked for."""
    training, holdout = np.arange(800), np.arange(800, 1000)
    uncapped = TrainingOptions().sampled(training, holdout)

    assert (uncapped[0] == training).all() and (uncapped[1] == holdout).all(), "no cap touches nothing"

    few_training, few_holdout = TrainingOptions(studies=50).sampled(training, holdout)

    assert (len(few_training), len(few_holdout)) == (40, 10), "the 4:1 the fold already has"
    assert (few_training == training[:40]).all(), "the FIRST positions, so two smokes read the same studies"

    one_training, one_holdout = TrainingOptions(studies=1).sampled(training, holdout)

    assert (list(one_training), list(one_holdout)) == ([0], [800]), "both halves keep a study"

    whole = TrainingOptions(studies=5000).sampled(training, holdout)

    assert (whole[0] == training).all() and (whole[1] == holdout).all(), "a cap cannot invent"
