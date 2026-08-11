from dataclasses import dataclass
from typing import cast, override

import numpy as np
import pytest
import torch
from torch import Tensor

from kneemri.model import POOL_PARTS, TARGETS, StudyDesign, StudyModel
from kneemri.slot_head import REPORT_DIMS

SEED = 2026
DIM = 8
PATCHES = 16
SLOTS = 6
GROUP = 3


@dataclass
class _Output:
    last_hidden_state: Tensor


class _FakeBackbone(torch.nn.Module):
    """A stand-in encoder whose tokens vary PER DIMENSION with the input.

    An earlier version returned `base + mean(pixels)` — the same scalar added to every feature. The
    head begins with a LayerNorm, which removes exactly that, so the stand-in made the model's output
    independent of its pixels and no test could detect a change in pooling. A useful double has to
    perturb the feature vector's SHAPE, not just its level.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(DIM, DIM) for _ in range(12)])
        self.checkpointing: dict[str, object] | None = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, object]) -> None:
        """Stand-in for the transformers hook, recording how it was asked."""
        self.checkpointing = gradient_checkpointing_kwargs

    @override
    def forward(self, *, pixel_values: Tensor) -> _Output:
        n = pixel_values.shape[0]
        rows = pixel_values.mean(dim=(1, 3))[:, :DIM] if pixel_values.shape[2] >= DIM else None
        if rows is None:
            rows = pixel_values.mean(dim=(1, 3)).repeat(1, (DIM // pixel_values.shape[2]) + 1)[:, :DIM]
        base = torch.arange(PATCHES + 1, dtype=torch.float32).reshape(1, -1, 1) * 0.01
        return _Output(last_hidden_state=base + rows.reshape(n, 1, DIM))


@pytest.fixture
def study_model() -> StudyModel:
    torch.manual_seed(SEED)
    return StudyModel(_FakeBackbone(), DIM, SLOTS)


def test_forward(study_model: StudyModel):
    """One logit per diagnosis per study, and a masked-out slot must change the answer."""
    torch.manual_seed(SEED)
    imgs = torch.randint(0, 256, (2, SLOTS, GROUP, 14, 14), dtype=torch.uint8)
    partial = torch.ones(2, SLOTS)
    partial[:, -1] = 0.0
    study_model.eval()
    with torch.no_grad():
        every_slot = study_model.forward(imgs, torch.ones(2, SLOTS), 14)
        one_absent = study_model.forward(imgs, partial, 14)
    assert every_slot.shape == (2, len(TARGETS))
    assert not torch.allclose(every_slot, one_absent), "the presence mask must reach the softmax"


def test_forward_resizes_to_the_requested_size(study_model: StudyModel):
    """A cache held at one resolution is resampled to whatever the member was fitted at."""
    torch.manual_seed(SEED)
    imgs = torch.randint(0, 256, (1, SLOTS, GROUP, 28, 28), dtype=torch.uint8)
    study_model.eval()
    with torch.no_grad():
        out = study_model.forward(imgs, torch.ones(1, SLOTS), 14)
    assert out.shape == (1, len(TARGETS))


def test_fingerprint(study_model: StudyModel):
    """The same model answers the seeded question identically twice; a different model does not."""
    device = torch.device("cpu")
    first = study_model.fingerprint(14, GROUP, device)
    assert first.shape == (2, len(TARGETS))
    assert np.array_equal(first, study_model.fingerprint(14, GROUP, device))
    torch.manual_seed(SEED + 1)
    other = StudyModel(_FakeBackbone(), DIM, SLOTS)
    assert not np.allclose(first, other.fingerprint(14, GROUP, device))


def test_fingerprint_leaves_training_mode_untouched(study_model: StudyModel):
    """A check run mid-training must not silently leave the model in eval."""
    study_model.train()
    study_model.fingerprint(14, GROUP, torch.device("cpu"))
    assert study_model.training


def test_pool_parts_widths_match_the_head_input():
    """`cls_mean` concatenates two `dim`-wide parts, `cls_mean_focal` three — the head's width follows."""
    assert POOL_PARTS["cls_mean"] == 2
    torch.manual_seed(SEED)
    focal = StudyModel(_FakeBackbone(), DIM, 2, StudyDesign(pool="cls_mean_focal"))
    assert focal.head.proj[1].in_features == DIM * POOL_PARTS["cls_mean_focal"]


def test_parameter_groups(study_model: StudyModel):
    """The head gets its own learning rate; the encoder gets a slower one and only its last blocks."""
    groups = study_model.parameter_groups(head_lr=1e-3, backbone_lr=8e-6, unfreeze_last=2)
    assert [group["lr"] for group in groups] == [1e-3, 8e-6]
    opened = cast("list[torch.nn.Parameter]", groups[1]["params"])
    assert opened, "the last blocks must remain trainable"
    assert all(parameter.requires_grad for parameter in opened)
    frozen = [p for p in study_model.backbone.parameters() if not p.requires_grad]
    assert frozen, "the early blocks must be frozen"


def test_forward_per_slice_pooling_differs_from_channel_stacking():
    """The two aggregations are different functions, and a test double must be able to tell.

    The first version of this could not: the stand-in encoder added the same scalar to every feature
    dimension, and the head's opening LayerNorm removed it, so the model's output did not depend on
    its pixels at all. A double has to perturb the feature vector's SHAPE to be worth anything.
    """
    imgs = torch.randint(0, 256, (2, SLOTS, GROUP, 14, 14), dtype=torch.uint8)
    mask = torch.ones(2, SLOTS)
    outputs = {}
    for mode in ("channels", "max"):
        torch.manual_seed(SEED)
        model = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling=mode)).eval()
        with torch.no_grad():
            outputs[mode] = model.forward(imgs, mask, 14)
    assert not torch.allclose(outputs["channels"], outputs["max"]), "per-slice pooling must change the function"


def test_per_slice():
    """Which poolings encode a slice ALONE — the one fact three call sites used to each decide alone.

    "attend" is the case that broke: the encoder branched on a set containing it, the attention module
    built itself on it, and the trainer's whole-stack guard tested `!= "max"` — so an arm that had
    been written, reviewed and queued died at startup with 'whole-stack reading needs per-slice
    pooling, not attend', against a model that supports precisely that.
    """
    for mode in ("max", "attend"):
        assert StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling=mode)).per_slice, mode
    assert not StudyModel(_FakeBackbone(), DIM, SLOTS).per_slice, "channel stacking reads three at once"


def test_forward_is_sensitive_to_its_pixels():
    """A guard for the double itself: two different STRUCTURED images must give different logits.

    Written because a degenerate stand-in silently made every pooling test vacuous — it added the same
    scalar to every feature, and the head's opening LayerNorm removed it.

    The images are structured rather than uniform on purpose. A uniform image drives this double to a
    constant feature vector, which LayerNorm legitimately erases; a real encoder would still separate
    black from white through its biases and position embeddings. Asserting on uniform inputs would be
    testing the stand-in's limits rather than the model's behaviour.
    """
    torch.manual_seed(SEED)
    model = StudyModel(_FakeBackbone(), DIM, SLOTS).eval()
    mask = torch.ones(1, SLOTS)
    generator = torch.Generator().manual_seed(SEED)
    first = torch.randint(0, 256, (1, SLOTS, GROUP, 14, 14), generator=generator, dtype=torch.uint8)
    second = torch.randint(0, 256, (1, SLOTS, GROUP, 14, 14), generator=generator, dtype=torch.uint8)
    with torch.no_grad():
        assert not torch.allclose(model.forward(first, mask, 14), model.forward(second, mask, 14)), (
            "the model must respond to its input"
        )


def test_load_trained():
    """Every member on the leaderboard was saved without `head.prior`, and must still load.

    A strict load rejects them all. The missing parameter is unambiguous — it is zeros unless the design
    asks for the prior — so it keeps its initialised value while everything the checkpoint does carry is
    still matched by name and shape.
    """
    torch.manual_seed(SEED)
    model = StudyModel(_FakeBackbone(), DIM, SLOTS)
    older = {name: value for name, value in model.state_dict().items() if name != "head.prior"}

    torch.manual_seed(SEED + 1)
    loaded = StudyModel(_FakeBackbone(), DIM, SLOTS)
    loaded.load_trained(older)

    assert torch.equal(loaded.head.out.weight, model.head.out.weight)
    assert float(loaded.head.prior.abs().sum()) == 0.0


def test_load_trained_still_rejects_a_checkpoint_that_does_not_fit():
    """Narrower than strict=False: a wrong shape is still an error, not a silent partial load."""
    model = StudyModel(_FakeBackbone(), DIM, SLOTS)
    state = dict(model.state_dict())
    state["head.out.weight"] = torch.zeros(3, 3)
    with pytest.raises(RuntimeError):
        model.load_trained(state)


def test_slot_features(study_model: StudyModel):
    """One embedding per slot, before any head reads them — the shape the two readouts share."""
    imgs = torch.randint(0, 256, (2, SLOTS, GROUP, 14, 14), dtype=torch.uint8)
    features = study_model.slot_features(imgs, 14)
    assert features.shape[:2] == (2, SLOTS)
    assert torch.isfinite(features).all()


def test_supervised(study_model: StudyModel):
    """Both readouts come from ONE encoder pass, and the logits match plain `forward` exactly.

    The equality is the contract: the fingerprint, the kernel and every published member read
    `forward`, so a member fitted through `supervised` has to be the same function at inference.
    """
    model = study_model.eval()
    imgs = torch.randint(0, 256, (2, SLOTS, GROUP, 14, 14), dtype=torch.uint8)
    mask = torch.ones(2, SLOTS)

    with torch.no_grad():
        logits, report = model.supervised(imgs, mask, 14)
        plain = model(imgs, mask, 14)

    assert torch.allclose(logits, plain, atol=1e-6), "training and inference must be the same function"
    assert report.shape == (2, REPORT_DIMS)


def test_recompute_activations(study_model: StudyModel):
    """Checkpointing is turned on through the encoder's own hook, non-reentrant.

    use_reentrant=False matters: the reentrant implementation silently produces no gradients when the
    checkpointed block's inputs do not require grad, which is exactly our case - the early encoder
    blocks are frozen, so the first trainable block receives an input with no grad history.
    """
    study_model.recompute_activations()

    backbone = cast("_FakeBackbone", study_model.backbone)
    assert backbone.checkpointing == {"use_reentrant": False}


def test_compile_encoder(study_model: StudyModel):
    """The encoder is replaced by a compiled callable, and the model still answers.

    Asserting on behaviour rather than on the wrapper's type: torch.compile returns different objects
    across versions, and what matters is that the swap does not break the forward path.
    """
    before = study_model.backbone
    study_model.compile_encoder()

    assert study_model.backbone is not before, "the encoder must actually be replaced"
    imgs = torch.randint(0, 256, (1, SLOTS, GROUP, 14, 14), dtype=torch.uint8)
    with torch.no_grad():
        assert study_model.forward(imgs, torch.ones(1, SLOTS), 14).shape == (1, len(TARGETS))
