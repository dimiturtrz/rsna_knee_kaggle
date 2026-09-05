"""Which trunks are declared, and that each declaration is internally consistent.

The type these entries are written in — `Encoder`, `Stage`, `Normalization` — is tested beside it in
`encoder.py`; everything here is a fact about the TABLE.
"""

import pytest

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
    V3_BASE,
    V3_SMALL,
    V3_SMALL_PLUS,
    VJEPA2_LARGE,
    Encoder,
    MountKind,
)

RAD_DINO_PREPROCESSOR = {
    "image_mean": [0.5307, 0.5307, 0.5307],
    "image_std": [0.2583, 0.2583, 0.2583],
    "crop_size": {"height": 518, "width": 518},
}
"""microsoft/rad-dino's own `preprocessor_config.json`, copied from the checkpoint rather than from the
forum post that pointed at it — the numbers are the whole finding, so they come from the primary source."""


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


def test_of():
    """A mount STRING resolves back to the Kaggle API that publishes it, both kinds and the fallback.

    The subpath case is the one that matters: callers hold `dimiturnt/dinov3-hf/vits16`, not the bare
    dataset id, so a rule matching only whole mounts would read every dataset encoder as unknown.
    """
    assert MountKind.of("metaresearch/dinov2/PyTorch/small/1") == MODEL_MOUNT
    assert MountKind.of("dimiturnt/dinov3-hf/vits16") == DATASET_MOUNT
    assert MountKind.of("someone/never-declared/leaf") == DATASET_MOUNT


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
    # An EMPTY repository is not a repository — it is how a row says the weights come from a local
    # builder rather than from the hub, and there are now two of those (`ortholarge` from a converted
    # snapshot, `mricore` from a training checkpoint under the data root). Requiring "" to be unique
    # would cap local trunks at one, which is a rule about nothing. What must stay unique for those is
    # the builder, since that is what actually names the file the weights are read from.
    repositories = [encoder.repository for encoder in ENCODERS.values() if encoder.repository]
    assert len(set(repositories)) == len(repositories), "two variants pointing at one repository is a typo"

    builders = [encoder.builder for encoder in ENCODERS.values() if not encoder.repository]
    assert len(set(builders)) == len(builders), "two local variants reading one checkpoint is a typo"


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
