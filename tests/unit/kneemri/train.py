from pathlib import Path

import pytest

from kneemri.backbone import ENCODERS, SMALL
from kneemri.crop_centre import Crop
from kneemri.model import StudyDesign
from kneemri.objective import Objective, RankedObjective
from kneemri.slot_reader import Band, Reading
from kneemri.slots import Routing
from kneemri.train import TrainingOptions
from kneemri.trainer import Adaptation


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

    from kneemri.ema import LiveWeights, WeightAverage  # noqa: PLC0415
    from kneemri.model import StudyDesign, StudyModel  # noqa: PLC0415
    from kneemri.report_supervision import NoReportSupervision, ReportSupervision  # noqa: PLC0415
    from kneemri.trainer import GROUP  # noqa: PLC0415

    from .model import DIM, SLOTS, _FakeBackbone  # noqa: PLC0415

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
        ["train", "--labels", "a.csv", "--lora-rank", "4", "--recompute-activations", "--compile-encoder"],
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

    monkeypatch.setattr("kneemri.train.StudyModel", lambda *args, **kwargs: _Model())
    monkeypatch.setattr(
        "kneemri.train.AutoModel", type("_Auto", (), {"from_pretrained": staticmethod(lambda name: None)})
    )
    options.fitted_model(ENCODERS[SMALL], 6, StudyDesign())

    assert order == ["low_rank:4", "recompute", "compile"], "adapters go in BEFORE the graph is frozen around them"

    order.clear()
    TrainingOptions().fitted_model(ENCODERS[SMALL], 6, StudyDesign())
    assert order == [], "a plain run is the pretrained encoder, unwrapped and uncompiled"


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
