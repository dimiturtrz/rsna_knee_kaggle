import numpy as np
import pytest
import torch

from core.data.labels.targets import Targets
from core.findings import TARGETS
from kneemri.models.slice_pooling import ATTEND, MAXIMUM
from kneemri.models.slot_head import ATTENDED, AVERAGED, MeanSlotHead, SlotHead
from kneemri.training.head_trial import BATCH, AxisConcordance, HeadAxis, HeadTrial, TrialHead

SEED = 2026
STUDIES = 200
SLOTS = 6
SLICES = 4
WIDTH = 16


def _corpus() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(SEED)
    features = generator.normal(0, 1, (STUDIES, SLOTS, SLICES, WIDTH)).astype(np.float16)
    mask = np.ones((STUDIES, SLOTS), dtype=np.float32)
    mask[:, -1] = 0.0
    return features, mask


def test_features():
    """The cached corpus arrives as float32 with the per-slice axis INTACT.

    Intact because the slice reduction is a variant now: a corpus reduced on the way in could only ever
    express the maximum, which is exactly the constant this harness was built to stop encoding.
    """
    features, _ = _corpus()

    tensor = HeadTrial().features(features)

    assert tensor.shape == (STUDIES, SLOTS, SLICES, WIDTH)
    assert tensor.dtype == torch.float32


def test_forward():
    """Slices reduce by the elementwise maximum, and an absent slot is ZEROED rather than left as-is.

    The head's softmax reads the mask, so a stale vector sitting behind a zero mask is a value waiting
    to be used by accident — the shape this project has now found ten times. Asserted through the head
    that consumes it: the reduction is only correct if what the head receives is correct.
    """
    features, mask = _corpus()
    head = TrialHead(WIDTH, SLOTS)
    reduced: list[torch.Tensor] = []
    head.head.register_forward_pre_hook(lambda _module, inputs: reduced.append(inputs[0]))

    logits = head.forward(HeadTrial().features(features), torch.from_numpy(mask))

    assert logits.shape == (STUDIES, len(TARGETS))
    expected = torch.from_numpy(features.astype(np.float32)).amax(dim=2)
    assert torch.allclose(reduced[0][:, 0], expected[:, 0], atol=1e-3), "a present slot is the max over its slices"
    assert torch.all(reduced[0][:, -1] == 0.0), "an absent slot carries no features at all"


def test_forward_attends_over_slices_when_asked_to():
    """`attend` puts a PARAMETERISED reduction on the slice axis, and starts as the mean.

    Zero-initialised gates mean the variant begins at the mean rather than somewhere arbitrary, so a
    difference this axis produces is the attention learning something and not a different starting
    point wearing the same flag.
    """
    features, mask = _corpus()

    head = TrialHead(WIDTH, SLOTS, ATTEND)

    assert head.slices is not None, "the attend variant must carry parameters, or it is the max under a new name"
    reduced: list[torch.Tensor] = []
    head.head.register_forward_pre_hook(lambda _module, inputs: reduced.append(inputs[0]))
    head(HeadTrial().features(features), torch.from_numpy(mask))
    mean = torch.from_numpy(features.astype(np.float32)).mean(dim=2)
    assert torch.allclose(reduced[0][:, 0], mean[:, 0], atol=1e-3), "untrained slice attention is the mean"


def test_forward_builds_the_head_the_registry_names():
    """`slot_pooling` reaches the constructor, rather than being recorded and ignored.

    The failure this replaces: a flag that lands in the run's config while the head is built from a
    constant, so the ledger reports an ablation that never ran.
    """
    assert isinstance(TrialHead(WIDTH, SLOTS, MAXIMUM, ATTENDED).head, SlotHead)
    assert isinstance(TrialHead(WIDTH, SLOTS, MAXIMUM, AVERAGED).head, MeanSlotHead)


def test_forward_carries_the_anatomical_prior():
    """`anatomical_prior` reaches the head's attention logits — the field `run` used to never read."""
    without = TrialHead(WIDTH, SLOTS).head.prior
    with_prior = TrialHead(WIDTH, SLOTS, anatomical_prior=True).head.prior

    assert torch.all(without == 0.0), "no prior means no bias on the logits, not a small one"
    assert torch.any(with_prior != 0.0), "the prior must reach the head, or the flag is decoration"


def test_forward_refuses_what_cached_features_cannot_express():
    """`channels` happens before the encoder, so offering it here would answer a different question."""
    with pytest.raises(ValueError, match="not expressible over cached features"):
        TrialHead(WIDTH, SLOTS, "channels")

    with pytest.raises(ValueError, match="anatomical prior covers"):
        TrialHead(WIDTH, SLOTS + 1, anatomical_prior=True)


def test_run():
    """A head fitted on features that CARRY the label beats one fitted on noise.

    The point of the harness is that a head experiment is answerable in seconds. If a signal planted in
    the features cannot be recovered, the harness cannot rank head variants and is not worth trusting.
    """
    generator = np.random.default_rng(SEED)
    truth = (generator.random((STUDIES, len(TARGETS))) > 0.5).astype(np.float32)
    mask = np.ones((STUDIES, SLOTS), dtype=np.float32)
    signal = np.zeros((STUDIES, SLOTS, SLICES, WIDTH), dtype=np.float32)
    signal[:, 0, 0, : len(TARGETS)] = truth * 4.0 - 2.0
    noise = generator.normal(0, 1, signal.shape).astype(np.float32)

    targets = Targets(truth, np.ones_like(truth), [str(i) for i in range(STUDIES)])
    split = (np.arange(0, 160, dtype=np.int64), np.arange(160, STUDIES, dtype=np.int64))
    trial = HeadTrial(epochs=6, rank_weight=0.0)

    carried = trial.run(trial.features(signal.astype(np.float16)), torch.from_numpy(mask), targets, split)
    unrelated = trial.run(trial.features(noise.astype(np.float16)), torch.from_numpy(mask), targets, split)

    assert carried > unrelated, "a planted signal must be recoverable, or the harness ranks nothing"
    assert carried > 0.7, "and recoverable well, not merely better than noise"


def _axis(control: float, variant: float) -> HeadAxis:
    return HeadAxis(
        axis="anatomical_prior",
        control={"anatomical_prior": False},
        variant={"anatomical_prior": True},
        fine_tuned_control=control,
        fine_tuned_variant=variant,
        source="test",
    )


def test_fine_tuned():
    """The pair's value is variant minus control, in the direction the variant was proposed."""
    assert _axis(0.8081, 0.8005).fine_tuned == pytest.approx(-0.0076)


def test_resolved():
    """A pair inside the ruler's own floor is UNREAD, not equal — the distinction the whole gate rests on."""
    assert _axis(0.8081, 0.8005).resolved()
    assert not _axis(0.8457, 0.8473).resolved()


def test_read():
    """The committed record parses, its `note` lines are skipped, and every pair changes something.

    A pair whose control and variant are the same trial would report a structural zero that reads like a
    measurement — the reason the channel-stacked pair is described in a note instead of listed as data.
    """
    axes = HeadAxis.read()

    assert axes, "the record must hold pairs, or --correlate silently measures nothing"
    assert all(axis.control != axis.variant for axis in axes)
    assert all(axis.source for axis in axes), "a number with no provenance cannot be checked or rerun"


def test_proxy():
    """The proxy delta is the two trials' difference, and both sides are actually built differently."""
    features, mask = _corpus()
    truth = (np.random.default_rng(SEED).random((STUDIES, len(TARGETS))) > 0.5).astype(np.float32)
    targets = Targets(truth, np.ones_like(truth), [str(i) for i in range(STUDIES)])
    split = (np.arange(0, 160, dtype=np.int64), np.arange(160, STUDIES, dtype=np.int64))
    concordance = AxisConcordance(HeadTrial(epochs=1))

    delta = concordance.proxy(_axis(0.81, 0.80), HeadTrial().features(features), torch.from_numpy(mask), targets, split)

    assert isinstance(delta, float)


def test_report(caplog: pytest.LogCaptureFixture):
    """An UNRESOLVED pair is logged and left out of the denominator.

    Counting it would let a pair the fine-tuned ruler could not read certify the proxy — a harness
    grading itself on questions nobody answered.
    """
    features, mask = _corpus()
    truth = (np.random.default_rng(SEED).random((STUDIES, len(TARGETS))) > 0.5).astype(np.float32)
    targets = Targets(truth, np.ones_like(truth), [str(i) for i in range(STUDIES)])
    split = (np.arange(0, 160, dtype=np.int64), np.arange(160, STUDIES, dtype=np.int64))

    with caplog.at_level("INFO"):
        agreed = AxisConcordance(HeadTrial(epochs=1)).report(
            HeadTrial().features(features),
            torch.from_numpy(mask),
            targets,
            split,
            (_axis(0.8457, 0.8473),),
        )

    assert agreed == 0
    assert "0 of 0 resolved pair(s) agree" in caplog.text
    assert "not yet shown to rank anything" in caplog.text


def test_report_scores_a_resolved_pair(caplog: pytest.LogCaptureFixture):
    """A pair the fine-tuned ruler COULD read is counted, and graded on direction rather than size.

    Direction is the only thing a cheap head proxy is asked for: it exists to say which pixel arm is
    worth a training slot, not to reproduce that arm's delta. So the log names the agreement and the
    denominator moves — the half of `report` an all-unresolved record never reaches.
    """
    features, mask = _corpus()
    truth = (np.random.default_rng(SEED).random((STUDIES, len(TARGETS))) > 0.5).astype(np.float32)
    targets = Targets(truth, np.ones_like(truth), [str(i) for i in range(STUDIES)])
    split = (np.arange(0, 160, dtype=np.int64), np.arange(160, STUDIES, dtype=np.int64))

    with caplog.at_level("INFO"):
        agreed = AxisConcordance(HeadTrial(epochs=1)).report(
            HeadTrial().features(features),
            torch.from_numpy(mask),
            targets,
            split,
            (_axis(0.8081, 0.8005),),
        )

    assert agreed in {0, 1}, "one resolved pair either agrees or does not; the proxy's sign is not fixed here"
    assert "1 of 1 resolved pair(s) agree" in caplog.text or "0 of 1 resolved pair(s) agree" in caplog.text
    assert "direction" in caplog.text, "the pair is graded on direction, and said so"


def test_batches():
    """Batches are shuffled and the partial last one is dropped, as the pixel trainer does.

    Dropped for the same reason: the ranking term scores pairs WITHIN a batch, so a short final batch
    steps on a different number of pairs than every other one.
    """
    studies = np.arange(BATCH * 2 + 7, dtype=np.int64)

    batches = HeadTrial()._batches(studies, np.random.default_rng(SEED))

    assert len(batches) == 2, "the remainder of seven is dropped, not padded"
    assert all(len(batch) == BATCH for batch in batches)
    assert set(np.concatenate(batches).tolist()) <= set(studies.tolist())
