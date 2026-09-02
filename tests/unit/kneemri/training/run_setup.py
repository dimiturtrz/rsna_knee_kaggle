import dataclasses
from pathlib import Path

import numpy as np
import pytest
import torch

from core.findings import TARGETS
from kneemri.models.backbone import ENCODERS, SMALL
from kneemri.models.model import NO_LAYER_DECAY, StudyDesign
from kneemri.training.feature_distillation import FeatureDistilledObjective, Widths
from kneemri.training.landed import Request
from kneemri.training.objective import (
    DISSENT_TERM,
    FEATURE_TERM,
    MARGIN_TERM,
    ContrastiveObjective,
    MarginObjective,
    Objective,
    RankedObjective,
)
from kneemri.training.options import TrainingOptions
from kneemri.training.remembered_contrast import RememberedContrast
from kneemri.training.remembered_margin import RememberedMargin
from kneemri.training.remembered_ranking import RememberedRanking
from kneemri.training.run_setup import RunSetup
from kneemri.training.sampling import Disagreed, MixedCorpus, Ramped, Uniform
from kneemri.training.trainer import Adaptation


def _setup(**asked: object) -> RunSetup:
    """A run's flags and the builder over them, which is the pair every test here is about."""
    return RunSetup(TrainingOptions(**asked))  # type: ignore[arg-type]


def test_objective():
    """The flags choose the loss, and the default must stay plain weighted cross-entropy.

    Checked without a GPU, a corpus or a downloaded encoder on purpose: the ranking arm sits behind a
    queue, and a miswired objective would otherwise surface only after the card had been waited on.
    """
    assert type(_setup().objective()) is Objective
    assert not _setup().objective().normalise_by_count

    ranked = _setup(rank_weight=0.5).objective()
    assert isinstance(ranked, RankedObjective)
    assert ranked.weight == 0.5

    assert _setup(rank_weight=0.5, normalise_by_count=True).objective().normalise_by_count
    assert _setup(per_label_loss=True).objective().per_label, "a per-label loss is a different objective"
    assert _setup(rank_weight=0.5, per_label_loss=True).objective().per_label, "and it survives the ranking"


def test_objective_pair_memory():
    """A memory only exists to widen a term, so asking for one without any term must REFUSE.

    Same reason the rest of `objective` is checked here: the failure this guards is an arm that trains the
    plain recipe under the memory arm's name and gets read as a refutation of the mechanism. A refusal at
    flag-parse time costs nothing; the silent no-op costs a queue slot and a wrong conclusion.
    """
    remembered = _setup(rank_weight=0.5, pair_memory=64).objective()
    assert isinstance(remembered, RememberedRanking)
    assert remembered.memory.remembered.maxlen == 64, "the flag must reach the pool it sizes"
    assert remembered.weight == 0.5

    with pytest.raises(ValueError, match="nothing for it to widen"):
        _setup(pair_memory=64).objective()


def test_objective_composes_the_pool_with_the_contrastive_term():
    """The pool composes with the contrastive term, and the flag has to REACH the memory it sizes.

    `--pair-memory` was refused beside `--contrast-weight` until today, and not because the terms
    disagree: the chain returned one class and lost whichever branch came second. The composition is the
    fix `contrast_dense` pre-registered against its own reading, so what this pins is that asking for it
    builds the widened class AND that the capacity is the one the flag named — a memory silently left at
    the default would make the arm a reading about `PAIR_MEMORY` instead.
    """
    widened = _setup(contrast_weight=1.0, rank_weight=1.0, pair_memory=32).objective()
    assert isinstance(widened, RememberedContrast)
    assert widened.memory.capacity() == 32, "the flag must reach the pool it sizes"
    assert widened.contrast_weight == 1.0
    assert widened.weight == 1.0, "the house recipe's ranking term survives, or the arm is two changes"

    plain = _setup(contrast_weight=1.0).objective()
    assert isinstance(plain, ContrastiveObjective)
    assert not isinstance(plain, RememberedContrast), "no memory asked for, no memory built"


def test_objective_composes_the_pool_with_the_margin_term():
    """The pool composes with the AUC-margin term, widening the class-mean comparison across steps."""
    widened = _setup(auc_margin=0.8, pair_memory=32).objective()
    assert isinstance(widened, RememberedMargin)
    assert widened.memory.capacity() == 32, "the flag must reach the pool it sizes"
    assert widened.margin == 0.8

    plain = _setup(auc_margin=0.8).objective()
    assert isinstance(plain, MarginObjective)
    assert not isinstance(plain, RememberedMargin), "no memory asked for, no memory built"


def test_adaptation():
    """Which blocks move and how fast are reachable, and default to the inherited pair.

    Both were module constants with no flags - which is why the depth bracket could not distinguish
    "more blocks do not help" from "no block moved at this rate". They travel together because a depth
    result is uninterpretable without the rate it was measured at.
    """
    assert _setup().adaptation() == Adaptation()
    default = _setup().adaptation()
    assert round(default.head_lr / default.backbone_lr) == 125, "the inherited head:encoder ratio"

    faster = _setup(backbone_lr=1e-4, unfreeze_last=12).adaptation()
    assert faster.backbone_lr == 1e-4
    assert faster.unfreeze_last == 12
    assert faster.head_lr == Adaptation().head_lr, "raising the encoder rate must not move the head's"

    assert _setup().adaptation().layer_decay == NO_LAYER_DECAY, "flat is the shape the ledger was measured in"
    assert _setup(layer_decay=0.7).adaptation().layer_decay == 0.7, "the third member of the triple must land"

    assert _setup().adaptation().weight_decay == Adaptation().weight_decay, "the inherited 0.02"
    assert _setup(weight_decay=0.1).adaptation().weight_decay == 0.1, (
        "the decay rides on the adaptation because it is the OTHER half of how far the encoder moves: "
        "rate and decay set the norm a weight settles at jointly, and the rate sweep moved one of them"
    )


def test_trainer(tmp_path: Path):
    """Every optional policy a run asked for is attached, and nothing it did not ask for is.

    This is the function that replaced a run of `if arguments.x: trainer.y()` in `main`, and the
    failure it guards against is silent: a flag that stops being wired does not raise, it trains a
    different recipe than the one on the command line and reports a plausible number for it.
    """
    import torch  # noqa: PLC0415 - the rest of this module needs no torch

    from core.data.corpus.slot_reader import GROUP  # noqa: PLC0415
    from kneemri.models.ema import LiveWeights, WeightAverage  # noqa: PLC0415
    from kneemri.models.model import StudyDesign, StudyModel  # noqa: PLC0415
    from kneemri.training.report_supervision import NoReportSupervision, ReportSupervision  # noqa: PLC0415

    from ..models.model import DIM, SLOTS, _FakeBackbone  # noqa: PLC0415

    device = torch.device("cpu")
    plain = _setup().trainer(StudyModel(_FakeBackbone(), DIM, SLOTS), device, [], np.arange(120))
    assert plain.window == GROUP, "a default run reads one group, not the stack"
    assert plain.read_size is None
    assert plain.schedule is None
    assert plain.autocast_dtype is torch.bfloat16, "bf16 is the DEFAULT, and a precision is a numerics change"
    assert type(plain.average) is LiveWeights, "no EMA unless asked: it changes which weights are SAVED"
    assert type(plain.reports) is NoReportSupervision

    model = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max"))
    asked = _setup(whole_stack=True, slices=16, train_size=392, one_cycle=True, epochs=2, ema=True, bf16=False)
    trainer = asked.trainer(model, device, [], np.arange(120))
    assert trainer.window == 16
    assert trainer.read_size == 392
    assert trainer.schedule is not None, "one-cycle must reach the trainer, or the rate never anneals"
    assert type(trainer.average) is WeightAverage, "the EMA is what `evaluated()` scores and the checkpoint keeps"
    assert trainer.autocast_dtype is torch.float16, "--fp16 is the only way back, so it must land"
    assert trainer.scaler is not None, "fp16 has a scale that can collapse, so the scaler comes with it"

    # The warmup has to be attached BEFORE the schedule is built, and this is where that ordering is
    # observable: `one_cycle` reads the fraction once, into `pct_start`, so an attachment after it would
    # leave `policies()` reporting the value asked for over a cycle annealing on the inherited 0.15.
    tuned = _setup(one_cycle=True, epochs=2, warmup=0.4, grad_clip=1.5).trainer(
        StudyModel(_FakeBackbone(), DIM, SLOTS), device, [], np.arange(120)
    )
    assert tuned.warmup == 0.4
    assert tuned.clipping.max_norm == 1.5
    assert tuned.schedule is not None, "the fraction must not cost the run its schedule"

    vectors = tmp_path / "reports.npz"
    np.savez(vectors, identifiers=np.array(["s0"]), vectors=np.zeros((1, 4), dtype=np.float32))
    supervised = _setup(report_weight=0.3, report_vectors=vectors).trainer(
        StudyModel(_FakeBackbone(), DIM, SLOTS), device, ["s0"], np.arange(120)
    )
    attached = supervised.reports
    assert isinstance(attached, ReportSupervision), "the term the loop ADDS, so an unattached one is a plain run"
    assert attached.weight == 0.3


def test_reports(tmp_path: Path):
    """Report supervision is all-or-nothing: a weight with no vectors is refused, not skipped.

    Skipping it trained the PLAIN recipe under the report arm's name, so the arm would have read as a
    refutation of the mechanism when it refuted only the invocation.
    """
    assert _setup().reports([]) is None, "no weight, no supervision, no file read"

    with pytest.raises(ValueError, match="--report-vectors"):
        _setup(report_weight=0.3).reports([])

    vectors = tmp_path / "reports.npz"
    np.savez(vectors, identifiers=np.array(["s0"]), vectors=np.zeros((1, 4), dtype=np.float32))
    supervision = _setup(report_weight=0.3, report_vectors=vectors).reports(["s0"])
    assert supervision is not None
    assert supervision.weight == 0.3


def test_teacher(tmp_path: Path):
    """The teacher is all-or-nothing for `test_reports`' reason, and it is refused BEFORE the card.

    Half of this mechanism is worse than none of it: a `--distill-weight` with no readings would descend
    the label term alone under a distilling arm's name, and a `--teacher-targets` with no weight would
    load an array nothing reads. Either way the ledger gains a row that names a mechanism the run never
    ran, which is the failure every check in this file exists against.
    """
    assert _setup().teacher([], np.array([], dtype=np.int64)) is None, "no weight, no file, no readings"

    readings = tmp_path / "teacher.npz"
    np.savez(readings, identifiers=np.array(["s0"]), scores=np.zeros((1, len(TARGETS)), dtype=np.float32))

    with pytest.raises(ValueError, match="two halves of one mechanism"):
        _setup(distill_weight=0.3).teacher(["s0"], np.array([0]))
    with pytest.raises(ValueError, match="two halves of one mechanism"):
        _setup(teacher_targets=readings).teacher(["s0"], np.array([0]))

    teacher = _setup(distill_weight=0.3, teacher_targets=readings).teacher(["s0"], np.array([0]))
    assert teacher is not None
    assert teacher.rows(np.array([0])).shape == (1, len(TARGETS))

    # The OTHER consumer of the same file. The halves that must arrive together are "a term that reads a
    # teacher" and "a teacher", so a dissenting arm has to trip both branches exactly as a distilling one
    # does — the version of this gate that named only distillation let one through silently.
    with pytest.raises(ValueError, match="two halves of one mechanism"):
        _setup(dissent_weight=0.3).teacher(["s0"], np.array([0]))
    assert _setup(dissent_weight=0.3, teacher_targets=readings).teacher(["s0"], np.array([0])) is not None


def test_teacher_embeddings(tmp_path: Path):
    """All-or-nothing, refused BEFORE the card, and a SEPARATE mechanism from the targets teacher.

    A `--feature-weight` with no npz would descend the plain recipe under a representation arm's name,
    and a `--teacher-features` with no weight would load an array nothing reads. Both halves are checked
    here rather than on the first batch, because the first batch is forty minutes into a card.
    """
    width = 4
    features = tmp_path / "features.npz"
    np.savez(features, identifiers=np.array(["s0"]), embeddings=np.zeros((1, width), dtype=np.float32))

    assert _setup().teacher_embeddings([], np.array([], dtype=np.int64)) is None

    with pytest.raises(ValueError, match="two halves of one mechanism"):
        _setup(feature_weight=20.0).teacher_embeddings(["s0"], np.array([0]))
    with pytest.raises(ValueError, match="two halves of one mechanism"):
        _setup(teacher_features=features).teacher_embeddings(["s0"], np.array([0]))

    teacher = _setup(feature_weight=20.0, teacher_features=features).teacher_embeddings(["s0"], np.array([0]))
    assert teacher is not None
    assert teacher.width() == width, "the teacher's own width, which is what sizes the projection"


def test_objective_builds_the_feature_term_only_against_measured_widths():
    """The representation objective is reached by its flag, and it REFUSES to guess either width.

    Neither number is on the command line — one is a fact about the built model and the other about the
    loaded npz — so a call that reached here without them is a call that would fit a projection between
    two numbers nobody measured. The pre-fit callers (`arm_check`, `vram_need`) never reach it, since
    the branch they take is decided by a flag that is off.
    """
    widths = Widths(8, 5)
    chosen = _setup(feature_weight=20.0, rank_weight=1.0).objective(widths)

    assert isinstance(chosen, FeatureDistilledObjective)
    assert chosen.settings()[FEATURE_TERM] == 20.0
    assert chosen.weight == 1.0, "the house recipe's ranking term survives, or the arm is two changes"
    assert (chosen.projection.in_features, chosen.projection.out_features) == (8, 5)

    with pytest.raises(ValueError, match="must not guess"):
        _setup(feature_weight=20.0).objective()


def test_objective_refuses_to_compose_the_feature_term_with_the_terms_the_chain_would_drop():
    """The dispatch returns ONE class, so a second term flag is silently built or silently lost.

    `--rank-weight` is deliberately absent from the refusal: `FeatureDistilledObjective` extends the
    ranked one and carries it, which is what keeps the arm a one-variable change against its control.
    """
    for conflict in ({"distill_weight": 0.3}, {"dissent_weight": 0.3}, {"contrast_weight": 0.3}, {"pair_memory": 8}):
        with pytest.raises(ValueError, match="not composed"):
            _setup(feature_weight=20.0, **conflict).objective(Widths(8, 5))

    assert _setup(feature_weight=20.0, rank_weight=1.0).objective(Widths(8, 5)) is not None


def test_objective_refuses_to_compose_dissent_with_the_terms_it_replaces():
    """A dissenting arm that also passes one of these would have that flag SILENTLY dropped.

    `DecorrelatedObjective` holds the flat-mass ranking term at zero on purpose, and neither the teacher
    pull nor the remembered pool is composed with it. Each of the three would otherwise be recorded in
    the config and ignored by the consumer — the failure mode this whole file is written against, and one
    this repo has now paid a full card slot for once.
    """
    for conflict in ({"distill_weight": 0.3}, {"rank_weight": 0.3}, {"pair_memory": 8}):
        with pytest.raises(ValueError, match="not composed"):
            _setup(dissent_weight=0.3, **conflict).objective()

    assert _setup(dissent_weight=0.3).objective().settings()[DISSENT_TERM] == 0.3


def test_objective_refuses_to_compose_the_margin_with_the_term_it_replaces():
    """`--auc-margin` REMOVES cross-entropy, and those flags add themselves to it.

    Composed, the run would descend a sum that is neither objective, while its config reported both
    switches landed — the shape that has cost this repo a card slot before. The refusal names the margin
    flag rather than the stacked one, because that is the flag whose meaning the caller has misread.
    `--pair-memory` is NOT a conflict: it widens the margin term itself rather than adding a second term.
    """
    for conflict in ({"rank_weight": 0.3}, {"distill_weight": 0.3}, {"dissent_weight": 0.3}):
        with pytest.raises(ValueError, match="not composed"):
            _setup(auc_margin=1.0, **conflict).objective()

    chosen = _setup(auc_margin=0.8).objective()
    assert isinstance(chosen, MarginObjective)
    assert chosen.settings()[MARGIN_TERM] == 0.8


def test_model(monkeypatch: pytest.MonkeyPatch):
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

    monkeypatch.setattr("kneemri.training.run_setup.StudyModel", lambda *args, **kwargs: _Model())
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda name: None)
    card = torch.device("cuda")
    RunSetup(options).model(ENCODERS[SMALL], 6, StudyDesign(), card)

    assert order == ["low_rank:4", "recompute", "compile"], "adapters go in BEFORE the graph is frozen around them"

    order.clear()
    _setup().model(ENCODERS[SMALL], 6, StudyDesign(), card)
    assert order == ["compile"], "a plain run is the pretrained encoder unwrapped, and compiled since it is free"

    order.clear()
    _setup(compile_encoder=False).model(ENCODERS[SMALL], 6, StudyDesign(), card)
    assert order == [], "and eager is reachable, for the control that has to measure what compiling buys"

    order.clear()
    _setup().model(ENCODERS[SMALL], 6, StudyDesign(), torch.device("cpu"))
    assert order == [], "and a CPU run is never compiled: inductor's gain is CUDA kernels, and the trace dies"


def _probe(default: object) -> object:
    """A value of the field's own type that is NOT its default, so a field that did not move is visible."""
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 7
    if default is None or isinstance(default, str):
        # A NAMED FINDING and not an arbitrary string: `--focus-label` is validated against the twelve,
        # so a probe of "x" would raise inside the policy rather than fail the wiring assert it is here
        # to make. Its default is None, which is why the int/float branches above cannot reach it.
        return TARGETS[3]
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

    assert len(shared) == 41, "pinned, because a renamed flag would otherwise drop silently out of the walk"

    plain = _setup().requested(cpu)
    for name in sorted(shared):
        probe = _probe(getattr(TrainingOptions(), name))
        moved = _setup(**{name: probe}).requested(cpu)

        assert getattr(moved, name) == probe, f"--{name} does not reach the request field of its own name"
        followed = {other for other in shared if getattr(moved, other) != getattr(plain, other)}
        assert followed == {name}, f"--{name} also moved {followed - {name}}"


def test_requested_resolves_the_three_fields_that_are_not_a_flag():
    """Two are an argument's PRESENCE, and compiling is decided by the DEVICE rather than by the flag.

    The device half is the one that bites on this machine: `RunSetup.model` compiles on CUDA only, so a
    request built from the flag alone would report an un-landed compile on every CPU smoke run.
    """
    asked = _setup(recompute_activations=True, report_vectors=Path("vectors.npz")).requested(torch.device("cpu"))

    assert asked.recompute is True
    assert asked.reports is True, "the VECTORS decide it — a weight with no vectors is refused earlier"
    assert _setup().requested(torch.device("cpu")).reports is False
    assert _setup(compile_encoder=True).requested(torch.device("cpu")).compiling is False
    assert _setup(compile_encoder=True).requested(torch.device("cuda")).compiling is True
    assert _setup(compile_encoder=False).requested(torch.device("cuda")).compiling is False


def test_budget():
    """The cap is a function of the RECIPE, and two callers now need that answer rather than one.

    The trainer caps its own allocator with it; the queue's free-VRAM gate asks the same question before
    it will start an arm, and asked a hard-coded 18432 MiB instead — a 12-slice arm's figure handed to
    every arm. `encoder_adapt_thawed` reserves 9.6 GB at batch 2 and parked behind it with 15.8 GB free.
    """
    encoder = ENCODERS[SMALL]

    budget = _setup(slices=20, batch=2, unfreeze_last=12).budget()
    assert budget.slices == 20
    assert budget.batch == 2
    assert budget.retained == encoder.retained_units(336, 12), "the units of the blocks the run opened"
    assert budget.params_m == encoder.params_m

    assert _setup(slices=20, train_size=448).budget().units == encoder.activation_units(448), (
        "--train-size is the grid the step encodes, so the cap must read it over --image-size"
    )
    assert _setup(slices=20, unfreeze_last=12).budget().needed() > _setup(slices=20).budget().needed(), (
        "opening twice the blocks retains twice the stack, which is the term that refused the arm"
    )
    assert _setup(slices=20, unfreeze_last=0, lora_rank=8).budget().retained == encoder.activation_units(336), (
        "under LoRA the base weights are frozen and every block still carries a retaining adapter, and "
        "the clamp is the ENCODER's — a caller clamping against `depth` opens 12 of a 36-block trunk"
    )


def test_drawing(tmp_path: Path):
    """The two flags resolve to three policies, and both degenerate cases stay the recipes they were.

    A ramp is a modifier on the tilt, so a ramp asked for without a spread file is still the uniform
    policy — the alternative is an arm that quietly trains the ledger's default under a curriculum's
    name.
    """
    weights = tmp_path / "spread.npz"
    np.savez(weights, studies=np.array(["a", "b"]), spread=np.array([0.10, 0.30]))
    identifiers = ["a", "b"]

    assert isinstance(_setup().drawing(identifiers), Uniform)
    assert isinstance(_setup(sample_ramp=6).drawing(identifiers), Uniform), "a ramp alone draws nothing"

    static = _setup(sample_spread=weights, sample_alpha=0.5).drawing(identifiers)
    assert isinstance(static, Disagreed) and not isinstance(static, Ramped)
    assert static.tilt() == pytest.approx(0.5)

    ramped = _setup(sample_spread=weights, sample_alpha=0.5, sample_ramp=6).drawing(identifiers)
    assert isinstance(ramped, Ramped)
    assert ramped.epochs == 6
    assert ramped.tilt() == pytest.approx(0.5), "the destination is what the switch check reads"


def test_drawing_mixed_corpus() -> None:
    identifiers = ["a", "b", "c"]
    corpus = np.array(["canonical", "external", "external"])
    policy = _setup(corpus_oversample=3.0).drawing(identifiers, corpus)
    assert isinstance(policy, MixedCorpus)
    assert policy.factor == pytest.approx(3.0)
    assert policy.target == "canonical"


def test_drawing_mixed_corpus_requires_corpus_column() -> None:
    with pytest.raises(ValueError, match="corpus"):
        _setup(corpus_oversample=3.0).drawing(["a", "b"])


def test_drawing_mixed_corpus_conflicts_with_disagreed() -> None:
    with pytest.raises(ValueError, match="sample-spread"):
        _setup(corpus_oversample=3.0, sample_spread=Path("x.npz")).drawing(["a", "b"])
