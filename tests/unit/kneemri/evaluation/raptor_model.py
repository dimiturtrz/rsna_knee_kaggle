"""The raptor model reimplementation, checked on the parts that are ours to get wrong.

The equality oracle in `raptor_model.main` is the real proof and it needs their checkout and a 292 MB
checkpoint, so it is not a unit test. These cover the pieces that carry a decision — which pooling an
architecture name selects, that the gate is softmaxed over WINDOWS, and that a checkpoint round-trips
through `load_state_dict(strict=True)` with no key remapping.
"""

from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from kneemri.evaluation.raptor_model import (
    ATTENTION_WIDTH,
    FINDINGS,
    Architecture,
    RaptorClassifier,
    RaptorMember,
)

FEATURES = 16
"""Backbone width in the stubbed tests — small, because nothing here depends on its size."""

WINDOWS = 4
SIDE = 8
TEST_ARCHITECTURE = "resnet18"
"""A real timm name, built with `pretrained=False` so the test touches no network and no cache."""

TEST_RESOLUTION = 32

RESNET18_WIDTH = 512
"""What `resnet18` ends at — the width the head must take FROM the backbone, never from a default."""


class Stub(nn.Module):
    """A backbone that answers a fixed width, so the head can be tested without a real network."""

    num_features = FEATURES

    @override
    def forward(self, images: Tensor) -> Tensor:
        """One feature vector per image, deterministic in the image's own mean."""
        return images.mean(dim=(1, 2, 3)).unsqueeze(1).expand(-1, FEATURES).contiguous()


def classifier() -> RaptorClassifier:
    """A head over the stub, seeded so two constructions in one test are comparable."""
    torch.manual_seed(0)
    return RaptorClassifier(Stub(), features=FEATURES).eval()


def test_transformer() -> None:
    """`coatnet` and `maxvit` carry `vit` in the name and must NOT take the CLS-token path."""
    assert Architecture("vit_base_patch16_224", 224).transformer
    assert Architecture("dinov2_vitl14", 518).transformer
    assert not Architecture("coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k", 384).transformer, (
        "the shipped arm is a hybrid: routed as a transformer it would pool a token it does not have"
    )
    assert not Architecture("maxvit_base_tf_384", 384).transformer
    assert not Architecture("convnext_base", 384).transformer


def test_built() -> None:
    """The classifier is removed, so the backbone hands back features rather than class scores."""
    backbone = Architecture(TEST_ARCHITECTURE, TEST_RESOLUTION).built().eval()
    output = backbone(torch.zeros(1, 3, TEST_RESOLUTION, TEST_RESOLUTION))

    assert output.shape == (1, backbone.num_features)


def test_width() -> None:
    """The head is sized from the backbone's own width — a wrong one refuses the checkpoint outright."""
    backbone = Architecture(TEST_ARCHITECTURE, TEST_RESOLUTION).built()

    assert Architecture.width(backbone) == RESNET18_WIDTH
    assert Architecture.width(Stub()) == FEATURES


def test_encode() -> None:
    """The window axis survives the backbone, which sees a flat batch of images."""
    model = classifier()
    features = model.encode(torch.rand(2, WINDOWS, 3, SIDE, SIDE))

    assert features.shape == (2, WINDOWS, FEATURES)


def test_head() -> None:
    """The gate is softmaxed over WINDOWS, not over findings — the axis the whole member turns on."""
    model = classifier()
    features = torch.rand(1, WINDOWS, FEATURES)
    logits = model.head(features)

    assert logits.shape == (1, FINDINGS)

    normed = model.norm(features)
    attention = torch.softmax(model.att(normed), dim=1)
    assert torch.allclose(attention.sum(dim=1), torch.ones(1, FINDINGS), atol=1e-5), (
        "softmaxed over findings instead, each window would sum to one and every finding would read "
        "the same pooled vector — a well-formed member that has lost its per-finding attention"
    )
    pooled = torch.einsum("bkn,bkf->bnf", attention, normed)
    assert torch.allclose(logits, (pooled * model.clsW).sum(-1) + model.clsb)


def test_forward() -> None:
    """The whole model is the head over the encoding, and nothing else."""
    model = classifier()
    windows = torch.rand(1, WINDOWS, 3, SIDE, SIDE)

    assert torch.allclose(model.forward(windows), model.head(model.encode(windows)))
    assert torch.allclose(model.forward(windows), model(windows))


def test_loaded(tmp_path: Path) -> None:
    """A checkpoint names its own architecture and loads STRICTLY — no key remapping anywhere."""
    torch.manual_seed(1)
    backbone = Architecture(TEST_ARCHITECTURE, TEST_RESOLUTION).built()
    written = RaptorClassifier(backbone, features=Architecture.width(backbone))
    path = tmp_path / "arm.pt"
    torch.save({"arch": TEST_ARCHITECTURE, "res": TEST_RESOLUTION, "model": written.state_dict()}, path)

    member = RaptorMember.loaded(path, "cpu")

    assert member.architecture == Architecture(TEST_ARCHITECTURE, TEST_RESOLUTION), (
        "the checkpoint's own `arch`/`res` must win over the defaults, or a second arm is served at "
        "the first arm's resolution and reads as merely weak"
    )
    assert torch.equal(member.model.clsW, written.clsW)
    assert member.model.att[0].out_features == ATTENTION_WIDTH
    assert not member.model.training


def test_loaded_refuses_a_checkpoint_whose_shapes_are_not_this_head(tmp_path: Path) -> None:
    """Strict is the point: a mismatched checkpoint has to fail loudly, not load part of itself."""
    path = tmp_path / "wrong.pt"
    torch.save({"arch": TEST_ARCHITECTURE, "res": TEST_RESOLUTION, "model": {"clsb": torch.zeros(3)}}, path)

    with pytest.raises(RuntimeError):
        RaptorMember.loaded(path, "cpu")


def test_probs() -> None:
    """Twelve probabilities for one study, in [0, 1], with the batch axis added and removed here."""
    member = RaptorMember(classifier(), Architecture(TEST_ARCHITECTURE, SIDE))
    probabilities = member.probs(torch.rand(WINDOWS, 3, SIDE, SIDE), "cpu")

    assert probabilities.shape == (FINDINGS,)
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0


def test_scored() -> None:
    """A decoded study goes in whole: the member windows it at its OWN resolution, not the volume's."""
    depth = 64
    volume = np.zeros((depth, SIDE, SIDE), np.uint8)
    volume[2:20] = 200
    mask = (volume.reshape(depth, -1).sum(1) > 0).astype(np.uint8)
    member = RaptorMember(classifier(), Architecture(TEST_ARCHITECTURE, SIDE * 2))

    assert member.scored(volume, mask, "cpu").shape == (FINDINGS,)
