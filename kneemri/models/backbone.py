"""Which DINOv2 an encoder means, and how wide its features are.

`facebook/dinov2-small` with `DIM = 384` was written into the trainer, into the member-curve analysis
and into the Kaggle kernel as three independent constants, and the checkpoint recorded the variant as a
bare string that nothing read back. Changing encoder therefore meant changing a number in three files
and hoping they agreed — and a mismatched width does not fail loudly, it fails as a shape error deep in
a load, or not at all if the head happens to accept it.

The variant is the only free choice; the repository and the feature width follow from it.
"""

# Re-exported rather than moved out of reach: `Encoder` and its vocabulary were addressed as
# `kneemri.models.backbone.<name>` by some forty modules before the file passed the god-file ceiling,
# and a split that renames every one of those imports would bury the one change worth reviewing.
from kneemri.models.encoder import (
    DATASET_MOUNT as DATASET_MOUNT,
)
from kneemri.models.encoder import (
    HALF as HALF,
)
from kneemri.models.encoder import (
    IMAGENET as IMAGENET,
)
from kneemri.models.encoder import (
    MODEL_MOUNT as MODEL_MOUNT,
)
from kneemri.models.encoder import (
    PREPROCESSOR_FILES as PREPROCESSOR_FILES,
)
from kneemri.models.encoder import (
    UNIT as UNIT,
)
from kneemri.models.encoder import (
    Encoder as Encoder,
)
from kneemri.models.encoder import (
    Normalization as Normalization,
)
from kneemri.models.encoder import (
    Stage as Stage,
)
from kneemri.models.foreign_encoders import FOREIGN_ENCODERS
from kneemri.models.variants import (
    BASE as BASE,
)
from kneemri.models.variants import (
    COATNET as COATNET,
)
from kneemri.models.variants import (
    COATNET_2 as COATNET_2,
)
from kneemri.models.variants import (
    CONVNEXT as CONVNEXT,
)
from kneemri.models.variants import (
    CONVNEXT_BASE as CONVNEXT_BASE,
)
from kneemri.models.variants import (
    DINOV2 as DINOV2,
)
from kneemri.models.variants import (
    DINOV3 as DINOV3,
)
from kneemri.models.variants import (
    LARGE as LARGE,
)
from kneemri.models.variants import (
    MEDSAM as MEDSAM,
)
from kneemri.models.variants import (
    MEDSAM_BASE as MEDSAM_BASE,
)
from kneemri.models.variants import (
    MRICORE as MRICORE,
)
from kneemri.models.variants import (
    MRICORE_BASE as MRICORE_BASE,
)
from kneemri.models.variants import (
    ORTHO_LARGE as ORTHO_LARGE,
)
from kneemri.models.variants import (
    RAD_DINO as RAD_DINO,
)
from kneemri.models.variants import (
    RADDINO as RADDINO,
)
from kneemri.models.variants import (
    RADIMAGENET as RADIMAGENET,
)
from kneemri.models.variants import (
    RADIMAGENET_DENSENET as RADIMAGENET_DENSENET,
)
from kneemri.models.variants import (
    RADIODINO as RADIODINO,
)
from kneemri.models.variants import (
    RADIODINO_BASE as RADIODINO_BASE,
)
from kneemri.models.variants import (
    RADIODINO_SMALL as RADIODINO_SMALL,
)
from kneemri.models.variants import (
    RADIODINO_SMALL_P8 as RADIODINO_SMALL_P8,
)
from kneemri.models.variants import (
    SMALL as SMALL,
)
from kneemri.models.variants import (
    SWIN as SWIN,
)
from kneemri.models.variants import (
    SWIN_BASE as SWIN_BASE,
)
from kneemri.models.variants import (
    V3_BASE as V3_BASE,
)
from kneemri.models.variants import (
    V3_LARGE as V3_LARGE,
)
from kneemri.models.variants import (
    V3_SMALL as V3_SMALL,
)
from kneemri.models.variants import (
    V3_SMALL_PLUS as V3_SMALL_PLUS,
)
from kneemri.models.variants import (
    VIT as VIT,
)
from kneemri.models.variants import (
    VIT_IN21K as VIT_IN21K,
)
from kneemri.models.variants import (
    VJEPA2 as VJEPA2,
)
from kneemri.models.variants import (
    VJEPA2_LARGE as VJEPA2_LARGE,
)


class MountKind:
    """Which Kaggle API publishes a mount, answered by the registry that declared it.

    A registry QUERY rather than a fact about an entry, which is why it lives beside the table and not on
    `Encoder`: it was a staticmethod there until the split, and a staticmethod that reads a module global
    is a free function wearing a class name.
    """

    @staticmethod
    def of(mount: str) -> str:
        """The kind this mount is published as, or `DATASET_MOUNT` for one no encoder declares.

        A caller holding a mount STRING — every pre-flight check does, because `MemberPackage.mounts`
        returns strings — cannot otherwise tell a model version from a dataset subpath, and asking the
        wrong endpoint is not a soft failure: `kaggle datasets files metaresearch/dinov2` answers 403,
        which surfaces as a traceback out of a gate that was checking something else entirely.

        An unknown mount reads as a DATASET because that is what ours are; a mount no encoder declares
        is a question for the caller that invented it, not a kind this registry can invent an answer for.
        """
        for encoder in ENCODERS.values():
            if encoder.claims(mount):
                return encoder.mount_kind
        return DATASET_MOUNT


ENCODERS: dict[str, Encoder] = {
    SMALL: Encoder(
        SMALL,
        "facebook/dinov2-small",
        384,
        mount="metaresearch/dinov2/PyTorch/small/1",
        mount_kind=MODEL_MOUNT,
        family=DINOV2,
        leaf="small",
    ),
    # 86M parameters against 22M, and 768-wide features. The strongest public notebooks here run base.
    #
    # THE +0.0085 THIS COMMENT USED TO CLAIM WAS MEASURED ON A RULER WE NO LONGER USE, and on the
    # current one base does NOT clear the noise floor: rank24_s2026 (small) 0.8272 against
    # rankbase24_s2026 (base) 0.8283 is +0.0011, against a floor of 0.0053 that is the holdout's own
    # sampling error (knee-2rx). The old number predates the augmentation fix and the ledger rebuild
    # (knee-ghl), so it is not a second opinion about base — it is a measurement of a different setup.
    #
    # Left here rather than deleted because the stale claim nearly bought a 90-minute arm: it reads as
    # evidence FOR capacity, and our own current numbers say capacity is not the constraint (knee-7u5).
    BASE: Encoder(
        BASE,
        "facebook/dinov2-base",
        768,
        mount="metaresearch/dinov2/PyTorch/base/1",
        mount_kind=MODEL_MOUNT,
        family=DINOV2,
        leaf="base",
        params_m=86.6,
    ),
    # 300M parameters, 1024-wide. The rung ABOVE the public frontier: every strong notebook here stops
    # at base, and the reason is the scored kernel's T4 rather than anything about the task. That
    # constraint binds INFERENCE only — training happens on our own card, and the weights are mounted
    # from `metaresearch/dinov2/PyTorch/large/1`, which exists and is 1.2 GB.
    #
    # THAT ENDORSEMENT NO LONGER HOLDS EITHER, and it rested on the same stale ruler as base's. On the
    # current one: small 0.8272, base 0.8283, large_deep_rank24 0.8304 — and the large row also varies
    # depth, so its +0.0032 is not even a clean capacity measurement. All of it sits inside a 0.0053
    # floor. The lever that HAS cleared the floor since is whole-stack reading (+0.0147), which is a
    # data question, not a capacity one.
    LARGE: Encoder(
        LARGE,
        "facebook/dinov2-large",
        1024,
        mount="metaresearch/dinov2/PyTorch/large/1",
        mount_kind=MODEL_MOUNT,
        family=DINOV2,
        leaf="large",
        params_m=304.0,
    ),
    # DINOv3, authorised 2026-08-12. Patch 16 rather than 14, so the read sizes that tile cleanly are
    # multiples of 16: 336 (441 tokens), 384 (576), 448 (784). Note 392 - our best DINOv2 read size -
    # does NOT tile at patch 16, and 384 costs exactly what DINOv2 costs at 336.
    #
    # It also carries FOUR register tokens where v2 has none, which is why the head derives its prefix
    # count from the backbone's config instead of assuming one class token: folding registers into the
    # patch mean would average in the high-norm outliers they exist to absorb.
    #
    # THE MOUNT IS OURS. Kaggle has no official metaresearch/dinov3 — the only DINOv3 there is Keras's,
    # in KerasHub preset format, which `AutoModel.from_pretrained` cannot read. So the transformers
    # snapshot is published as a PRIVATE dataset, `dimiturnt/dinov3-hf`, holding one directory per
    # variant. Private is deliberate: it is enough for our own kernel to mount, and it is not a
    # redistribution to the public.
    #
    # THE LICENCE PERMITS THIS, checked before uploading rather than after: the DINOv3 licence allows
    # making the materials available to a third party "only under the terms of this Agreement", with a
    # copy of the Agreement provided alongside. The HF snapshot ships LICENSE.md and it was uploaded
    # with the weights, so that condition is met by construction. Publication of results additionally
    # requires acknowledging DINOv3, which the README must carry if a v3 member ever scores.
    V3_SMALL: Encoder(
        V3_SMALL,
        "facebook/dinov3-vits16-pretrain-lvd1689m",
        384,
        mount="dimiturnt/dinov3-hf/vits16",
        patch=16,
        family=DINOV3,
        leaf="vits16",
        params_m=21.6,
    ),
    # ViT-S+/16: the SAME 384-wide, 12-layer, 6-head, patch-16 shape as vits16 with a wider MLP —
    # 28.7M parameters against 21.6M. A drop-in for the head, and the apples-to-apples partner for the
    # small comparison, since capacity is the only thing that differs.
    V3_SMALL_PLUS: Encoder(
        V3_SMALL_PLUS,
        "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        384,
        mount="dimiturnt/dinov3-hf/vits16plus",
        patch=16,
        family=DINOV3,
        leaf="vits16plus",
        params_m=29.0,
    ),
    V3_BASE: Encoder(
        V3_BASE,
        "facebook/dinov3-vitb16-pretrain-lvd1689m",
        768,
        mount="dimiturnt/dinov3-hf/vitb16",
        patch=16,
        family=DINOV3,
        leaf="vitb16",
        params_m=86.6,
    ),
    # THE CONTROL FOR `ortholarge`, AND ONLY THAT, added 2026-09-02. Every other DINOv3 entry here was
    # added to be fitted; this one exists so the knee-continued release below is read against the exact
    # weights it was continued FROM. Same 24 blocks, 1024 width, 16 heads, patch 16 and 4 register
    # tokens — the pair differs in 1.25M knee images of continued self-distillation and in nothing else,
    # which is the only arrangement under which a delta is about the CORPUS rather than about capacity.
    #
    # NO MOUNT. It is a ruler, not a seat: a control that never ships needs no Kaggle mirror, and
    # arranging one before the delta is read would be paying for a decision that has not been made.
    V3_LARGE: Encoder(
        V3_LARGE,
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
        1024,
        patch=16,
        family=DINOV3,
        leaf="vitl16",
        params_m=303.1,
        depth=24,
    ),
    **FOREIGN_ENCODERS,
}
