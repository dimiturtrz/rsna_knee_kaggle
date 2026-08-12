"""The encoder table: one variant, one repository, one width."""

import pytest

from kneemri.backbone import BASE, ENCODERS, LARGE, SMALL, V3_BASE, V3_SMALL


def test_mounted_name():
    """A kernel has no internet, so each variant must name the Kaggle model it is mounted from."""
    assert ENCODERS[SMALL].mounted_name == "metaresearch/dinov2/PyTorch/small/1"


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
    """DINOv3 has no Kaggle mirror, so asking where a kernel would find it must raise HERE.

    The alternative is a kernel that mounts nothing, fails deep inside a load, and burns a submission
    slot to tell us something the encoder table already knew.
    """
    with pytest.raises(ValueError, match="no Kaggle mount"):
        _ = ENCODERS[V3_SMALL].mounted_name
    assert ENCODERS[V3_SMALL].patch == 16, "patch 16 decides which read sizes tile cleanly"
    assert ENCODERS[V3_BASE].dim == 768


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
