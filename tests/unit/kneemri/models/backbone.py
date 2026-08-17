"""The encoder table: one variant, one repository, one width."""

import pytest

from kneemri.models.backbone import BASE, ENCODERS, LARGE, SMALL, V3_BASE, V3_SMALL, V3_SMALL_PLUS, Encoder


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
