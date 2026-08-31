"""The encoder table: one variant, one repository, one width."""

from collections.abc import Callable

import pytest
import torch
from torch import nn

from kneemri.models.backbone import (
    BASE,
    CONVNEXT_BASE,
    DATASET_MOUNT,
    ENCODERS,
    IMAGENET,
    LARGE,
    MODEL_MOUNT,
    RAD_DINO,
    SMALL,
    SWIN_BASE,
    V3_BASE,
    V3_SMALL,
    V3_SMALL_PLUS,
    VJEPA2_LARGE,
    Encoder,
    Normalization,
    Stage,
)
from kneemri.models.trunk_tokens import TrunkAsTokens

RAD_DINO_PREPROCESSOR = {
    "image_mean": [0.5307, 0.5307, 0.5307],
    "image_std": [0.2583, 0.2583, 0.2583],
    "crop_size": {"height": 518, "width": 518},
}
"""microsoft/rad-dino's own `preprocessor_config.json`, copied from the checkpoint rather than from the
forum post that pointed at it — the numbers are the whole finding, so they come from the primary source."""


def test_declared():
    """A checkpoint's own constants, read off the file the checkpoint ships them in."""
    assert Normalization.declared(RAD_DINO_PREPROCESSOR) == ENCODERS[RAD_DINO].normalization

    with pytest.raises(ValueError, match="no image_mean"):
        Normalization.declared({"crop_size": {"height": 518, "width": 518}})


def test_triple():
    """One constant per channel, or one constant for all three — configs are written both ways."""
    assert Normalization.triple([0.5307]) == (0.5307, 0.5307, 0.5307)
    assert Normalization.triple([0.485, 0.456, 0.406]) == IMAGENET.mean


def test_assert_declared():
    """The registry held to the checkpoint, wherever the checkpoint ships the file to hold it to."""
    ENCODERS[RAD_DINO].normalization.assert_declared(RAD_DINO_PREPROCESSOR, RAD_DINO)

    with pytest.raises(ValueError, match=r"0\.5307.*0\.485"):
        ENCODERS[RAD_DINO].normalization.assert_declared(
            {"image_mean": [0.485, 0.456, 0.406], "image_std": [0.229, 0.224, 0.225]}, RAD_DINO
        )


def test_every_encoder_declares_the_constants_it_was_pretrained_under():
    """Six of seven are ImageNet and one is not, which is exactly why a shared constant survived so long.

    Each of these was checked against that encoder's own `preprocessor_config.json` — DINOv2 and
    ConvNeXtV2 from HuggingFace, DINOv3 from the local snapshot, RAD-DINO from both. V-JEPA2 is the one
    the registry states without having read: no snapshot on disk and no readable file on the hub, so its
    ImageNet default is an ASSUMPTION and is marked as one here rather than passing quietly.
    """
    natural = [SMALL, BASE, LARGE, V3_SMALL, V3_SMALL_PLUS, V3_BASE, CONVNEXT_BASE]

    assert all(ENCODERS[variant].normalization == IMAGENET for variant in natural), "all verified ImageNet"
    assert ENCODERS[RAD_DINO].normalization.mean == (0.5307, 0.5307, 0.5307), "greyscale radiographs"
    assert ENCODERS[RAD_DINO].normalization.std == (0.2583, 0.2583, 0.2583)
    assert ENCODERS[VJEPA2_LARGE].normalization == IMAGENET, "UNVERIFIED — check before arming a vjepa arm"


def test_mounted_source():
    """A kernel has no internet, so each variant names the Kaggle source it is attached from.

    The two kinds are both mounts and only one is a model: DINOv2 is Kaggle's, our DINOv3 snapshot is a
    private dataset with a directory per variant.
    """
    assert ENCODERS[SMALL].mounted_source == "metaresearch/dinov2/PyTorch/small/1"
    assert ENCODERS[V3_SMALL].mounted_source == "dimiturnt/dinov3-hf/vits16"
    assert ENCODERS[SWIN_BASE].mounted_source == "dimiturnt/swin-base-384", (
        "flat, because the leaf shape exists for a dataset holding four variants and there is one Swin"
    )


def test_kind_of():
    """A mount STRING resolves back to the Kaggle API that publishes it, both kinds and the fallback.

    The subpath case is the one that matters: callers hold `dimiturnt/dinov3-hf/vits16`, not the bare
    dataset id, so a rule matching only whole mounts would read every dataset encoder as unknown.
    """
    assert Encoder.kind_of("metaresearch/dinov2/PyTorch/small/1") == MODEL_MOUNT
    assert Encoder.kind_of("dimiturnt/dinov3-hf/vits16") == DATASET_MOUNT
    assert Encoder.kind_of("someone/never-declared/leaf") == DATASET_MOUNT


def test_every_model_mount_is_a_kaggle_model_and_ours_are_datasets():
    """The kinds are declared per encoder, so the registry is held to the two owners it actually has.

    Everything under `metaresearch/` is Kaggle's own published model; everything under `dimiturnt/` is a
    dataset we pushed. Nothing enforces that in the dataclass, and a mount typed with the wrong kind
    sends the live check to an endpoint that answers 403 rather than an absence.
    """
    for encoder in ENCODERS.values():
        if encoder.mount is None:
            continue
        expected = MODEL_MOUNT if encoder.mount.startswith("metaresearch/") else DATASET_MOUNT
        assert encoder.mount_kind == expected, f"{encoder.variant} declares {encoder.mount_kind} for {encoder.mount}"


def test_every_mount_is_locatable():
    """Every mounted encoder must RECOGNISE the directory its own mount arrives in.

    `mount` and `leaf` are two independent strings and nothing tied them together: `mount` is what a
    kernel declares, `leaf` is what `locates` matches once Kaggle has mounted it. Swin was added with
    mount `dimiturnt/swin-base-384` and leaf `swin-base`, which agree to a reader and not to
    `str.split`. The cost of that disagreement is not an error: `find_encoder` raises, `kernel_run`
    catches it per member and drops that one, and the entry SCORES — without the member it was built
    for, on a slot that cannot be spent twice.

    Kaggle mounts a source under `/kaggle/input` without its owner, so the path is the mount minus its
    first segment. That covers both kinds: a model source keeps its whole framework path
    (`dinov2/PyTorch/small/1`), a dataset source is its slug and any directory inside it.
    """
    for encoder in ENCODERS.values():
        if encoder.mount is None:
            continue
        mounted = "/kaggle/input/" + encoder.mount.split("/", 1)[1]
        assert encoder.locates(mounted), f"{encoder.variant} would not find its own mount at {mounted}"


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


def test_raddino_is_mountable_from_our_own_dataset():
    """Mirrored mid-arm, because an unmirrored encoder makes a WINNING member offline-only.

    A dataset of its own rather than a leaf under a shared one: `--dir-mode zip` unpacks a subdirectory
    into the ROOT, so the layout `dinov3-hf` uses is not what that flag produces, and a mount whose leaf
    segment does not exist fails inside a kernel with the slot already spent.
    """
    assert ENCODERS[RAD_DINO].mounted_source == "dimiturnt/rad-dino"
    assert ENCODERS[RAD_DINO].locates("/kaggle/input/rad-dino")
    assert not ENCODERS[RAD_DINO].locates("/kaggle/input/raddino-hf"), "the flattened first attempt"
    assert not ENCODERS[RAD_DINO].locates("/kaggle/input/dinov3-hf/vitb16")
    assert ENCODERS[RAD_DINO].dim == ENCODERS[BASE].dim, "the control it is measured against is DINOv2-base"


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


def test_load(
    monkeypatch: pytest.MonkeyPatch,
    feature_map: Callable[[int, int], nn.Module],
    token_sequence: Callable[[int, int], nn.Module],
):
    """A convolutional trunk is wrapped and a transformer is not — decided ONCE, here.

    Eight call sites used to fetch their own weights and hand them straight to the head. That was right
    while every encoder emitted token sequences; a family that does not would otherwise put the same
    branch in all eight.
    """
    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda source: feature_map(1024, 11))

    adapted = ENCODERS[CONVNEXT_BASE].load()
    assert isinstance(adapted, TrunkAsTokens), "a feature map needs the adapter"
    assert adapted.config.hidden_size == 1024, "the head reads its width off the config"
    assert adapted.config.num_register_tokens == 0, "a CNN has no registers to exclude"
    assert adapted.forward(torch.zeros(1, 3, 352, 352)).last_hidden_state.shape == (1, 1 + 121, 1024), (
        "the map arm, so the 11x11 grid arrives flattened rather than as three surviving axes"
    )
    assert not isinstance(ENCODERS[SMALL].load(), TrunkAsTokens), "a transformer already speaks the contract"

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda source: token_sequence(1024, 144))
    wrapped = ENCODERS[SWIN_BASE].load()
    assert isinstance(wrapped, TrunkAsTokens), "a prefixless sequence still needs the prefix supplied"
    assert wrapped.forward(torch.zeros(1, 3, 384, 384)).last_hidden_state.shape == (1, 1 + 144, 1024), (
        "the SEQUENCE arm: the entry states which one it wants, and a flatten here would be a silent reshape"
    )


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


def test_per_block():
    """One block's share of a stage — the currency a SUFFIX can be counted in, which a total cannot."""
    assert Stage(3, 4, 128).per_block(352) == 88 * 88 * 128, "the block count is the only thing divided out"
    assert Stage(3, 4, 128).units(352) == 3 * Stage(3, 4, 128).per_block(352), "the total is still the sum of them"


def test_blocks():
    """The trunk as a run-ordered block list, which is the only place its REAL depth is stated.

    `cnxbase` declares `depth` 12 — the field's default, meaningless once `stages` is given — and runs
    36. A caller clamping `unfreeze_last` against `depth` therefore opened a third of the trunk and
    priced it as all of it (knee-rwjs).
    """
    convnext = ENCODERS[CONVNEXT_BASE]
    assert len(convnext.blocks) == 36, "three, three, twenty-seven, three"
    assert len(convnext.blocks) != convnext.depth, "and `depth` is not that number, which is the trap"
    assert convnext.blocks[-1].width == 1024, "run order, so the last block is the narrowest grid and widest channel"
    assert convnext.blocks[0].width == 128

    small = ENCODERS[SMALL]
    assert len(small.blocks) == small.depth, "a trunk that states no stages is one uniform stage deep"


def test_retained_units():
    """What the OPEN blocks hold — a suffix of the block list, not a fraction of the trunk's total.

    The cost model multiplied the whole-trunk sum by a scalar in the block COUNT until 2026-08-25,
    which prices the open suffix at the trunk's AVERAGE block. On a 12-deep ViT that is exact; on
    `cnxbase` the last six blocks are 9.7% of the bill where six of twelve uniform would be 50%.
    """
    small = ENCODERS[SMALL]
    assert small.retained_units(336, 12) == small.activation_units(336), "all twelve open is the whole trunk"
    assert small.retained_units(336, 6) == small.activation_units(336) / 2, "and a ViT's blocks are equal"
    assert small.retained_units(336, 0) == 0.0, "nothing open retains nothing"

    convnext = ENCODERS[CONVNEXT_BASE]
    assert convnext.retained_units(352, 6) < 0.1 * convnext.activation_units(352), (
        "the last six blocks of a downsampling trunk are a tenth of its bill, not half"
    )
    assert convnext.retained_units(352, 1000) == convnext.activation_units(352), (
        "the clamp is HERE, against 36 real blocks — a caller clamping against `depth` would open 12"
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
