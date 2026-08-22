"""The encoder table: one variant, one repository, one width."""

from typing import override

import pytest
import torch
from torch import Tensor, nn

from kneemri.models.backbone import (
    BASE,
    CONVNEXT_BASE,
    ENCODERS,
    LARGE,
    SMALL,
    V3_BASE,
    V3_SMALL,
    V3_SMALL_PLUS,
    VJEPA2_LARGE,
    Encoder,
    MapAsTokens,
    Stage,
    TokenOutput,
)


class _FeatureMap(nn.Module):
    """A convolutional trunk's contract in three lines: `last_hidden_state` of shape (n, dim, h, w)."""

    def __init__(self, dim: int, side: int) -> None:
        super().__init__()
        self.dim, self.side = dim, side

    @override
    def forward(self, pixel_values: Tensor) -> TokenOutput:
        rows = torch.arange(self.side * self.side, dtype=torch.float32)
        emitted = rows.view(1, 1, self.side, self.side).expand(len(pixel_values), self.dim, self.side, self.side)
        return TokenOutput(emitted)


def test_mounted_source():
    """A kernel has no internet, so each variant names the Kaggle source it is attached from.

    The two kinds are both mounts and only one is a model: DINOv2 is Kaggle's, our DINOv3 snapshot is a
    private dataset with a directory per variant.
    """
    assert ENCODERS[SMALL].mounted_source == "metaresearch/dinov2/PyTorch/small/1"
    assert ENCODERS[V3_SMALL].mounted_source == "dimiturnt/dinov3-hf/vits16"


def test_every_variant_states_its_own_width():
    """The width follows from the variant — it is what three separate constants used to disagree about."""
    assert ENCODERS[SMALL].dim == 384
    assert ENCODERS[BASE].dim == 768
    assert ENCODERS[LARGE].dim == 1024
    assert all(name == encoder.variant for name, encoder in ENCODERS.items()), (
        "a variant whose key and name disagree picks the wrong mount at inference and loads silently"
    )

    # WIDTH NO LONGER IDENTIFIES AN ENCODER, and this assertion used to say it must. DINOv3-small is
    # 384-wide exactly like DINOv2-small, and v3base matches base at 768 — so a member fitted on one
    # and rebuilt as the other loads with NO shape error, which is precisely what that assertion was
    # protecting against. The protection moved rather than vanished: the repository is unique per
    # variant, the member records its variant, and `MemberPackage.check` refuses a member whose
    # fingerprint moves — which it does by order one when the weights are a different network.
    repositories = [encoder.repository for encoder in ENCODERS.values()]
    assert len(set(repositories)) == len(repositories), "two variants pointing at one repository is a typo"


def test_mount_is_refused_when_there_is_none():
    """An encoder with no Kaggle mirror must raise HERE, not inside a kernel.

    The alternative is a kernel that mounts nothing, fails deep inside a load, and burns a submission
    slot to tell us something the encoder table already knew. DINOv3 no longer exercises this — its
    weights are published as a private dataset — so the case is stated with an encoder that has none.
    """
    with pytest.raises(ValueError, match="no Kaggle mount"):
        _ = Encoder("nowhere", "facebook/nothing", 384).mounted_source


def test_dinov3_is_mountable_from_our_own_dataset():
    """Kaggle's only DINOv3 is Keras's, in a preset format `AutoModel.from_pretrained` cannot read."""
    assert ENCODERS[V3_SMALL].mounted_source == "dimiturnt/dinov3-hf/vits16"
    assert ENCODERS[V3_BASE].mounted_source == "dimiturnt/dinov3-hf/vitb16"
    assert ENCODERS[V3_SMALL].patch == 16, "patch 16 decides which read sizes tile cleanly"
    assert ENCODERS[V3_BASE].dim == 768


def test_locates():
    """Each encoder recognises its own mount and no sibling's — by SEGMENT, never by substring.

    `vits16` is a substring of `vits16plus`, and BOTH ARE 384 WIDE, so a substring rule would let each
    claim the other's mount and build a member on weights it was not fitted on with no shape error to
    show for it. That silent version is the one this matching exists to prevent; the loud version is
    merely a kernel that refuses to start.
    """
    assert ENCODERS[SMALL].locates("/kaggle/input/dinov2/pytorch/small/1")
    assert not ENCODERS[SMALL].locates("/kaggle/input/dinov3-hf/vits16")
    assert not ENCODERS[BASE].locates("/kaggle/input/dinov2/pytorch/small/1")
    assert ENCODERS[V3_SMALL].locates(r"C:\kaggle\input\dinov3-hf\vits16"), "a Windows path is the same mount"
    assert not ENCODERS[V3_SMALL].locates("/kaggle/input/dinov3-hf/vitb16")

    assert ENCODERS[V3_SMALL_PLUS].locates("/kaggle/input/dinov3-hf/vits16plus")
    assert not ENCODERS[V3_SMALL].locates("/kaggle/input/dinov3-hf/vits16plus"), "a substring is not a match"
    assert not ENCODERS[V3_SMALL_PLUS].locates("/kaggle/input/dinov3-hf/vits16")
    assert ENCODERS[V3_SMALL_PLUS].dim == ENCODERS[V3_SMALL].dim, "same width is what makes the mixup silent"

    assert not Encoder("nowhere", "facebook/nothing", 384).locates("/kaggle/input/anything"), (
        "an encoder that names no directory must match NOTHING rather than everything"
    )


def test_forward():
    """A feature map arrives as tokens with a class token in front, in row-major order.

    The head slices `tokens[:, 1:]` to get the patch grid and pools it. If the flatten dropped the
    prefix slot, that slice would silently eat the first spatial token and the model would train on a
    grid missing its top-left corner — a defect with no shape error to announce it.
    """
    adapter = MapAsTokens(_FeatureMap(dim=4, side=3), 4)
    emitted = adapter.forward(torch.zeros(2, 3, 96, 96)).last_hidden_state

    assert emitted.shape == (2, 1 + 9, 4), "one prefix token, then the 3x3 grid, at the trunk's own width"
    assert torch.equal(emitted[0, 1:, 0], torch.arange(9, dtype=torch.float32)), "row-major, like a ViT's patches"
    assert emitted[0, 0, 0] == pytest.approx(4.0), "the prefix holds the map's mean, which is what a CNN pools"


def test_load(monkeypatch: pytest.MonkeyPatch):
    """A convolutional trunk is wrapped and a transformer is not — decided ONCE, here.

    Eight call sites used to fetch their own weights and hand them straight to the head. That was right
    while every encoder emitted token sequences; a family that does not would otherwise put the same
    branch in all eight.
    """
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda source: _FeatureMap(dim=1024, side=11))

    adapted = ENCODERS[CONVNEXT_BASE].load()
    assert isinstance(adapted, MapAsTokens), "a feature map needs the adapter"
    assert adapted.config.hidden_size == 1024, "the head reads its width off the config"
    assert adapted.config.num_register_tokens == 0, "a CNN has no registers to exclude"
    assert not isinstance(ENCODERS[SMALL].load(), MapAsTokens), "a transformer already speaks the contract"


def test_units():
    """A stage costs blocks x tokens x width, and the tokens are the square of the side over the stride."""
    assert Stage(3, 4, 128).units(352) == 3 * 88 * 88 * 128, "88 = 352 / 4, and it is squared"
    assert Stage(3, 8, 128).units(352) == Stage(3, 4, 128).units(352) / 4, "one halving of the grid is a quarter"
    assert Stage(6, 4, 128).units(352) == 2 * Stage(3, 4, 128).units(352), "blocks are a plain multiplier"
    assert Stage(3, 4, 128, tubelet=2).units(352) == Stage(3, 4, 128).units(352) / 2, (
        "the quantity is PER SLICE, so a token spanning two slices halves it — the direction knee-2nxh had backwards"
    )


def test_activation_units():
    """The memory proxy the VRAM cap is a function of — and the shorthand a transformer gets for free.

    THIS IS WHAT FOUR DEAD ARMS HAD IN COMMON. The cap used to read the OUTPUT grid at the OUTPUT
    width, which is every block's grid in a transformer and one block's grid in four in a ConvNeXt: the
    smoke derived 2.84 GB and OOM'd against 28 GB of free card. The first assertion is the one that
    protects the existing ledger — a 12-deep ViT must still reduce to the old form exactly, or every
    recorded arm's cap moves.
    """
    small = ENCODERS[SMALL]
    assert small.activation_units(336) == 12 * (336 / 14) ** 2 * 384, "one stage, depth blocks, patch stride"
    assert small.activation_units(672) == 4 * small.activation_units(336), "tokens go as the square of the side"
    assert ENCODERS[BASE].activation_units(336) == 2 * small.activation_units(336), "width is a plain factor"

    convnext = ENCODERS[CONVNEXT_BASE]
    assert convnext.activation_units(352) == sum(stage.units(352) for stage in convnext.stages), (
        "a trunk that states its stages is priced by them and not by its patch and width"
    )
    assert convnext.activation_units(352) > 7 * (12 * (352 / 32) ** 2 * 1024), (
        "and that is 7.75x what the output grid alone accounts for, which is the size of the defect"
    )

    vjepa = ENCODERS[VJEPA2_LARGE]
    assert vjepa.activation_units(336) == 24 * (336 / 16) ** 2 * 1024 / 2, (
        "a tubelet trunk states no stages, so the depth-axis stride has to reach the shorthand"
    )


def test_require_tiling():
    """A read size the patch grid cannot tile is refused, with the sizes that would work.

    The two encoders disagree about which sizes are legal, which is the whole point: 336 and 448 tile
    under both, 392 only under patch 14, 384 only under patch 16. Our best DINOv2 recipe reads 392, so
    a DINOv3 run cannot simply copy it — and without this the failure is either a crash deep in the
    encoder or, worse, a silent interpolation of the position embeddings onto a grid the weights never
    saw.
    """
    for size in (336, 448):
        ENCODERS[SMALL].require_tiling(size)
        ENCODERS[V3_SMALL].require_tiling(size)

    ENCODERS[SMALL].require_tiling(392)
    with pytest.raises(ValueError, match="use 384 or 400"):
        ENCODERS[V3_SMALL].require_tiling(392)

    ENCODERS[V3_SMALL].require_tiling(384)
    with pytest.raises(ValueError, match="use 378 or 392"):
        ENCODERS[SMALL].require_tiling(384)
