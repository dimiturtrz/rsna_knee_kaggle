"""What a recorded config says about the network: the design switches, the encoder variant and its constants.

Its own module because `model.py` had crossed the god-module threshold (fan-in 17 x fan-out 10), and the
config-to-design reading is the cluster that could leave: four of the model's intra-package imports
(`backbone`, `variants`, and the pooling defaults) existed only to spell this value's defaults.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from kneemri.models.backbone import ENCODERS
from kneemri.models.encoder import IMAGENET, Normalization
from kneemri.models.slice_pooling import CHANNELS
from kneemri.models.slot_head import ATTENDED
from kneemri.models.token_pooling import CLS_MEAN
from kneemri.models.variants import SMALL

DEFAULT_VARIANT = SMALL
"""What a config that names no encoder was fitted on — every member published before `variant` was
recorded. Spelled here because this is where a config is turned back into a model."""


@dataclass(frozen=True)
class StudyDesign:
    """The model's behavioural switches, as one value rather than as a growing argument list.

    They arrived one at a time — a pooling mode, then a slice aggregation, then the anatomical prior —
    and each was a string or a flag threaded through the constructor, the trainer, the checkpoint and
    every analysis that rebuilds a member. Grouping them means a new switch is one field rather than an
    edit at each of sixteen call sites, and it means a checkpoint can record the design it was trained
    under as one object.
    """

    pool: str = CLS_MEAN
    slice_pooling: str = CHANNELS
    anatomical_prior: bool = False

    slice_regions: int = 1
    """How many depth regions `max` pooling takes a maximum within, before averaging across them.

    A COUNT AND NOT A NEW POOLING NAME, because the behaviour is entirely the count: one region is a
    single maximum over the band — the historic reading, the default, and what every member on disk was
    fitted under — and any other value is the same operation applied per region. A second entry in
    `SLICE_POOLINGS` would be a name that means "this field is not 1".

    WHY IT EXISTS. One maximum over the whole band makes every slice compete for the same channel, so
    the winner is whatever is loudest anywhere along the depth. Two arms measured what that costs.
    `band_wide` widened the band and bought the collateral ligaments at the central compartment's
    expense (MCL +0.0133, Medial OA -0.0088); `band_dense` restored the central density and bought the
    centre back at the collaterals' expense (against `band_wide`: PF OA +0.0194 and Medial OA +0.0122,
    both intervals clearing zero, against MCL -0.0111). Neither arm added or removed anatomy from the
    half it lost — only that half's SHARE of one pooled representation. Regions give each part of the
    depth its own maximum, so a quiet peripheral finding no longer competes with a loud central one.

    It is a field of the design rather than an argument of the pooling because the serve path rebuilds a
    member from the recorded config: a member fitted at four regions and served at one is a different
    function of the same pixels, and no fingerprint below the design would catch it."""

    through_plane: bool = False
    """Whether a slice reaches the encoder with its two depth NEIGHBOURS in the other two channels.

    THE ONE MECHANISM THE AGGREGATION CHAPTER LEFT OPEN. `attend_true` and `depth_true` between them
    saturated the slice-aggregation stage from three directions — more slices (-0.0001), more slots
    (+0.0002), and all four combining rules within 0.0019 of each other — and both closed with the same
    reading: the limit is UPSTREAM, in what one slice's feature vector contains. Every one of those arms
    mixed depth AFTER the encoder had already reduced each slice to a vector, so none of them could put
    through-plane information INTO that vector. This does, and it does it before any feature exists.

    IT IS NOT `channels`, and the difference is the whole reason this is a separate field rather than a
    revival. Channel stacking also feeds three slices to the three input channels, but it does so
    INSTEAD of per-slice pooling — one forward per slot, no slice axis at all — which couples the group's
    physical thickness to the slice count and is measurably why raising inference slices from 12 to 32
    cost -0.018 there. So the pixel-level mixing was only ever measured bundled with the loss of the
    slice axis, and the bundle is what lost. Here the slice axis, the pooling, the token count, the
    parameter count and the FLOP count are all exactly the control's: the only thing that changes is
    that the three channels hold (i-1, i, i+1) rather than three copies of i.

    Free by construction. The trunk's patch-embedding convolution already takes three channels and was
    pretrained on three that differ, so nothing is added, resized or initialised.
    """

    slot_pooling: str = ATTENDED
    """How the twelve diagnoses read the six slots: per-diagnosis attention, or a uniform `mean`.

    `mean` is the ABLATION and never a proposal — it exists so the attention has a denominator. Every
    number in the ledger was measured on a recipe that already contained the attention, so nothing says
    what it carries, and that is the bar every future pooling idea has to clear."""

    layer: int = -1
    """Which encoder block's tokens the head reads. -1 is the last, which is what every member so far
    was fitted on and therefore the only value that leaves the ledger's meaning untouched.

    It is a knob and not a defect: `last_hidden_state` is what the API returns and what linear probes
    conventionally use. But the final block of a self-supervised ViT is the most specialised toward its
    PRETRAINING objective, which is why both DINOv2 and DINOv3 ship `get_intermediate_layers` for dense
    work — and our findings are 1-3 mm while our head pools patch tokens, so this is the axis they live
    on. Untested for either encoder until now."""

    fuse_blocks: int = 0
    """How many blocks BEFORE the last add their normalised tokens to the last block's, each through a
    per-channel gate that starts at zero. 0 is every member so far.

    Reading one intermediate block alone REPLACES the pretrained readout (encoder_layer 9 cost -0.0227);
    a zero gate instead starts byte-identical to the control and lets the head pull in finer, less
    pretraining-specialised features only as far as the loss asks for them. Costs no extra forward pass."""

    slice_shift: int = 0
    """How many of the last blocks first mix in each slice's neighbours' tokens — see `SliceShift`. 0 is
    every member so far."""

    @staticmethod
    def lora_rank_of(config: Mapping[str, object]) -> int:
        """The rank of a config's adapters, or zero — read WITH the design, though not a field of it.

        Not a field because the rank changes which MODULES exist rather than how the forward pass reads
        them: it has to be applied to the built model before any weights arrive. It lives here so the
        two facts a config states about the network are read from one place.
        """
        return int(str(config.get("lora_rank", 0) or 0))

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "StudyDesign":
        """The design a checkpoint or manifest records, with every default spelled HERE and once.

        Two places rebuilt a trained model out of a recorded config — the serve path and the tool that
        ranks members — and each read its own subset with its own defaults. Their blind spots were
        complementary rather than shared, which is why neither showed up as a disagreement: the serve
        path forgot the prior, the ranking path forgot the pooling and the adapters, and a member
        fitted with either simply could not be loaded by the half that had never heard of it.

        A key the config does not carry means the value every member on disk was fitted with, so this
        reads an old manifest as the model it actually is rather than refusing it.
        """
        return cls(
            pool=str(config.get("pool", CLS_MEAN)),
            slice_pooling=str(config.get("slice_pooling", CHANNELS)),
            slot_pooling=str(config.get("slot_pooling", ATTENDED)),
            anatomical_prior=bool(config.get("anatomical_prior", False)),
            slice_regions=int(str(config.get("slice_regions", 1))),
            through_plane=bool(config.get("through_plane", False)),
            layer=int(str(config.get("encoder_layer", -1))),
            fuse_blocks=int(str(config.get("fuse_blocks", 0))),
            slice_shift=int(str(config.get("slice_shift", 0))),
        )

    @staticmethod
    def normalization_of(config: Mapping[str, object]) -> Normalization:
        """The constants the config's `variant` was pretrained under, ImageNet's for an unregistered one.

        Read from the REGISTRY rather than one shared constant: six of seven encoders were pretrained
        under the ImageNet constants and the seventh was not, so a hardcoded pair is right by coincidence.
        """
        encoder = ENCODERS.get(str(config.get("variant", DEFAULT_VARIANT)))
        return encoder.normalization if encoder is not None else IMAGENET
