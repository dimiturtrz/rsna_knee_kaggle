"""The study-level model: a shared image encoder under a per-diagnosis attention head.

Replicates the public baseline's architecture exactly, because the point of a replication is to
reproduce the number before changing anything — and because the trained members published alongside it
load only into this shape. `MemberPackage` checks that faithfulness numerically (see `members.py`);
this module is what it checks.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import override

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor, nn

from core.data.corpus.anatomy import ANATOMY, SlotAffinity
from core.findings import TARGETS
from kneemri.models.backbone import ENCODERS, IMAGENET, SMALL, Normalization
from kneemri.models.encoder_surgery import NO_LAYER_DECAY, EncoderSurgery
from kneemri.models.fingerprint import FINGERPRINT
from kneemri.models.slice_pooling import CHANNELS, SLICE_POOLINGS
from kneemri.models.slice_read import SliceRead
from kneemri.models.slot_head import ATTENDED, SLOT_POOLINGS
from kneemri.models.token_pooling import CLS_MEAN, POOL_PARTS, TOKEN_POOLINGS

COMPILED_PREFIX = "_orig_mod."
"""What `torch.compile` inserts into a state-dict key when it is REBOUND over a module attribute.

Spelled ONCE, here, beside the loader that strips it — the five members written before
`compile_encoder` switched to in-place compilation carry it, and every arm that ran `--compile-encoder`
saved its encoder under it. `EncoderDrift` had its own copy; a comparison that strips a stale spelling
matches nothing and reports a drift of 0.000, which is the most reassuring possible way to write
"measured nothing at all" and is exactly how that module's copy was found the first time.
"""

ADAPTERS = "lora_adapters"
RECOMPUTING = "recompute_activations"
COMPILED_ENCODER = "compile_encoder"
OPENED_BLOCKS = "opened_blocks"
DEPTH_REGIONS = "depth_regions"
"""The keys `applied` reports under — named beside the model because the model is what they describe."""

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
        )


class StudyModel(nn.Module):
    """Encoder plus head over a study's bag of slot images.

    The bag is flattened for the encoder and folded back before the head, so the encoder never sees the
    study structure and the head never sees pixels. Byte scaling, the ImageNet normalisation and the
    resize all live inside `forward` on purpose: they are part of the function the weights were fitted
    to, and a member run through a different one of them loads cleanly and predicts from the wrong
    image.
    """

    mean: Tensor
    std: Tensor

    def __init__(
        self,
        backbone: nn.Module,
        dim: int,
        n_slot: int,
        design: StudyDesign | None = None,
        normalization: Normalization = IMAGENET,
    ) -> None:
        # `backbone` is any module callable as `backbone(pixel_values=x)` returning an object with
        # `last_hidden_state` of shape (n, 1 + patches, dim) — a DINOv2 from transformers in
        # production, a deterministic stand-in under test, and a medical-pretrained encoder when that
        # ablation runs. Kept structural rather than a named protocol so a substitution is a
        # constructor argument, not an edit here.
        super().__init__()
        self.design = design or StudyDesign()
        self.backbone = backbone
        # A PLAIN ATTRIBUTE, not a submodule: `EncoderSurgery` is not an `nn.Module`, so it registers
        # nothing and no state-dict key changes. It holds the same backbone object this line just
        # registered, which is what lets the surgery run against the module the optimiser will see.
        self.surgery = EncoderSurgery(backbone)
        self.pool = self.design.pool
        self.slice_pooling = self.design.slice_pooling
        self.n_slot = n_slot
        # The BACKBONE's own config decides these, not our registry, because the loaded weights are the
        # only authority on what they emit. DINOv2 has one class token and no registers; DINOv3 adds
        # four registers, which must be excluded from the patch pooling below.
        config = getattr(backbone, "config", None)
        # A PLAIN VALUE, not a submodule: `SliceRead` is not an `nn.Module`, so it registers nothing and
        # no state-dict key changes. It holds every fact settled before the first pixel moves — the
        # tubelet, the position grid, the prefix tokens and the through-plane window — and performs the
        # step that consumes all four. The normalisation buffers stay registered HERE, below, because
        # they ride in the state dict.
        self.reading = SliceRead.of(
            backbone,
            self.design.pool,
            self.design.slice_pooling,
            through_plane=self.design.through_plane,
        )
        width = int(getattr(config, "hidden_size", dim) or dim)
        if width != dim:
            # A registry that disagrees with the weights is a typo, and it surfaces otherwise as a shape
            # error deep inside the head - or not at all, if the head happens to accept the width.
            message = f"encoder emits {width}-wide features and the registry says {dim}"
            raise ValueError(message)
        prior = SlotAffinity(ANATOMY).requested(TARGETS, n_slot, wanted=self.design.anatomical_prior)
        # STORED as well as used, because it is the width of `Readout.context` and a loss term that
        # projects out of the representation has to be built from it BEFORE the first batch — asking the
        # tensor at step one is too late, since the optimiser was assembled at construction.
        self.head_width = dim * POOL_PARTS[self.pool]
        head_width = self.head_width
        # Built only for the design that uses it: an unused module would still land in the state dict
        # and every published member would fail to load against it.
        # The lookup, not just a read: an unknown pooling raises HERE, before weights are fetched and
        # before a manifest's stale name can be served as channel stacking.
        pooling = SLICE_POOLINGS[self.slice_pooling]
        self._encodes_each_slice = pooling.per_slice
        self.slices = pooling.aggregator(head_width) if pooling.aggregator is not None else None
        self.head = SLOT_POOLINGS[self.design.slot_pooling](head_width, n_slot, len(TARGETS), prior)
        # PERSISTENT, so they ride in the state dict — which is what makes fixing the default safe:
        # every member already on disk keeps the constants it was actually fitted under, whatever the
        # registry now says, and only a fresh fit moves.
        self.register_buffer("mean", torch.tensor(normalization.mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(normalization.std).view(1, 3, 1, 1))

    @classmethod
    def from_config(cls, config: Mapping[str, object], backbone: nn.Module, dim: int, n_slot: int) -> "StudyModel":
        """The model a recorded config describes, adapters included, with no weights loaded yet.

        THE ONE REBUILD. Whatever a config says about the network belongs here, so a switch added to
        the trainer reaches every path that reconstructs a member at once instead of reaching whichever
        of them the author happened to be editing.

        The adapters are applied BEFORE the caller loads anything, because a LoRA member's state dict
        carries adapter keys that only exist once `low_rank` has wrapped the encoder's linears.

        The normalisation is resolved from the config's `variant` for the same reason everything else
        here is: both rebuild paths go through this one method, so an encoder whose constants are not
        ImageNet is right in the trainer and in the kernel at once rather than in whichever was edited.
        A member's own buffers overwrite these the moment its weights load — this decides what an
        UNLOADED model normalises with, which is what a fresh fit starts from.
        """
        variant = str(config.get("variant", DEFAULT_VARIANT))
        encoder = ENCODERS.get(variant)
        normalization = encoder.normalization if encoder is not None else IMAGENET
        model = cls(backbone, dim, n_slot, StudyDesign.from_config(config), normalization)
        rank = StudyDesign.lora_rank_of(config)
        if rank:
            model.low_rank(rank)
        return model

    @property
    def per_slice(self) -> bool:
        """Whether each slice is encoded ALONE, so a slot's embedding does not depend on a window.

        The one place that knows which poolings read per-slice. It was three places: the encoder
        branched on `{"max", "attend"}`, the module built itself on `== "attend"`, and the trainer's
        whole-stack guard tested `!= "max"` — so adding attention pooling silently left the guard
        rejecting it, and the arm died at startup with 'whole-stack reading needs per-slice pooling,
        not attend' against a model that supports exactly that.
        """
        return self._encodes_each_slice

    def check_whole_stack(self, slices: int) -> None:
        """Refuse, at STARTUP, a whole-stack read this model cannot take — and say which reason.

        TWO DIFFERENT TRUNKS CAN TAKE ONE AND THEY TAKE IT FOR DIFFERENT REASONS. A per-slice pooling
        encodes each slice alone, so the window may be any width. A tubelet trunk consumes the window
        as FRAMES, so it may be any width that its tubelet divides. What can take no window at all is
        channel stacking, which has exactly three channels to stack into.

        The guard lives here rather than in the trainer because the trainer's version tested a PROXY —
        first `!= "max"`, which silently rejected attention pooling the model supported, then
        `per_slice`, which silently rejected the volumetric read whose window is the one thing it
        exists to widen. Both are the same bug: a caller deciding what a model can be handed. The
        model is the only thing that knows.
        """
        if self.reading.volume is not None:
            tubelet = self.reading.volume.tubelet
            if slices % tubelet:
                message = (
                    f"this trunk spans {tubelet} slices per token, so a {slices}-slice "
                    f"window leaves a partial tubelet; use a multiple of {tubelet}"
                )
                raise ValueError(message)
            return
        if not self.per_slice:
            message = f"whole-stack reading needs per-slice pooling or a tubelet trunk, not {self.slice_pooling!r}"
            raise ValueError(message)

    @override
    def forward(
        self,
        imgs: UInt8[Tensor, "b s g p q"],
        mask: Float[Tensor, "b s"],
        img_size: int | None = None,
    ) -> Float[Tensor, "b o"]:
        """Encode every slot image of every study in the batch, then aggregate per diagnosis."""
        return self.head(self.slot_features(imgs, img_size), mask)

    def slot_features(
        self,
        imgs: UInt8[Tensor, "b s g p q"],
        img_size: int | None = None,
    ) -> Float[Tensor, "b s *o d"]:
        """One embedding per acquisition slot — everything before the head reads them.

        Or TWELVE per slot, one per diagnosis, under `labelwise` pooling: that aggregator replaces the
        slice axis with a diagnosis axis instead of removing it. `SlotHead.per_diagnosis` is where the
        two shapes meet, so no caller here branches on which pooling ran.

        Split out of `forward` so a training step can take BOTH readouts from one encoder pass. The
        encoder is essentially the whole cost of a step, so computing it twice to also predict a report
        embedding would have doubled training time to add an auxiliary loss.
        """
        stacked = self.slice_features(imgs, img_size)
        if not self.per_slice:
            return stacked[:, :, 0]
        return self.slices(stacked) if self.slices is not None else self.over_depth(stacked)

    def over_depth(self, stacked: Float[Tensor, "b s g w"]) -> Float[Tensor, "b s w"]:
        """The slot's embedding from its slices: a maximum per depth region, averaged over regions.

        `tensor_split` rather than a reshape because the slice count need not divide the region count —
        20 slices over 4 regions is even, 20 over 3 is not, and a reshape would refuse the second
        rather than give it a 7/7/6 split. One region is a single `amax` over the whole depth, which is
        the historic pooling reproduced by the same code rather than branched around it.

        The count is capped at the depth on hand, because a region thinner than a slice is an EMPTY
        split and `amax` over nothing raises. The bag that first hit it was the fingerprint's — three
        synthetic slices against four regions — which is a real reading and not a test artefact: the
        fingerprint is taken on every save, so an uncapped split would have failed a twelve-epoch arm
        at its first checkpoint rather than in the smoke.
        """
        regions = torch.tensor_split(stacked, min(self.design.slice_regions, stacked.shape[2]), dim=2)
        return torch.stack([region.amax(dim=2) for region in regions]).mean(dim=0)

    def slice_features(
        self,
        imgs: UInt8[Tensor, "b s g p q"],
        img_size: int | None = None,
    ) -> Float[Tensor, "b s g w"]:
        """The encoder's output for each slice of each slot, BEFORE the slices are pooled together.

        This is where the whole cost of a step lives, and it is the last point at which the result
        still depends only on the pixels and the encoder — everything after it is head. Split out so
        the features can be computed ONCE and reused: pooling, slot attention, the prior and the head
        are all cheap functions of this tensor, and they are the questions we currently answer at 320 s
        an epoch (see the cached-embedding harness).

        Channel stacking has no per-slice axis by construction — the group IS the encoder's three
        channels — so it returns a single position, and pooling over it is the identity.
        """
        batch, slots, group = imgs.shape[:3]
        per_slice = self.per_slice
        x = self.reading.pixels(imgs, self.mean, self.std, img_size, per_slice=per_slice)
        tokens = self._tokens(x)
        self.reading.check_tokens(x, tokens)
        pooled = self._pool(tokens)
        if not per_slice:
            return pooled.reshape(batch, slots, 1, -1)
        return pooled.reshape(batch, slots, group, -1)

    def supervised(
        self,
        imgs: UInt8[Tensor, "b s g p q"],
        mask: Float[Tensor, "b s"],
        img_size: int | None = None,
    ) -> "Readout":
        """Everything a training step can be judged on, from one pass over the encoder.

        Training-only, and deliberately NOT what `forward` returns: the fingerprint, the kernel and
        every published member read `forward`, so the inference contract stays exactly as it was and a
        member fitted with report supervision loads into the old path unchanged.
        """
        features = self.slot_features(imgs, img_size)
        context = self.head.context(features, mask)
        return Readout(
            logits=self.head.logits(context),
            report=self.head.report_vector(features, mask),
            context=context,
        )

    def fingerprint(self, image_size: int, group: int, device: torch.device) -> NDArray[np.float32]:
        """This model's answer to the one seeded question, as a portable identity — see `Fingerprint`."""
        return FINGERPRINT.of(self, self.n_slot, image_size, self.reading.probe_frames(group), device)

    def load_trained(self, state: dict[str, Tensor]) -> None:
        """Load weights saved by any earlier version of this model.

        Every checkpoint written before the anatomical prior existed lacks `head.prior`, and a strict
        load rejects the lot — including the five members already on the leaderboard and the ones the
        Kaggle kernel mounts. A missing prior is not ambiguous: the parameter is zeros when the design
        does not use it, which is exactly the model this checkpoint was trained as.

        So keys the checkpoint does not carry keep their initialised value, and everything it does carry
        must still match in name and shape. That is deliberately narrower than `strict=False`, which
        would also swallow a checkpoint that had nothing to do with this model.

        COMPILE'S PREFIX IS STRIPPED rather than rejected, which recovers weights rather than merely
        explaining them. `compile_encoder` now compiles in place so nothing new carries it, but five
        members were already saved through the rebinding version and are worth real GPU hours. The
        substitution is a no-op on every checkpoint that never met `torch.compile`.
        """
        merged = self.state_dict() | {key.replace(COMPILED_PREFIX, ""): value for key, value in state.items()}
        self.load_state_dict(merged)

    def low_rank(self, rank: int, alpha: float = 1.0) -> int:
        """Adapt every encoder block along a rank-r subspace — see `EncoderSurgery.low_rank`.

        Spelled here rather than left to the caller because `model.surgery.low_rank(...)` is a call
        chained through a field into a stranger, and because adaptation is part of what a MODEL is: it
        changes which keys the state dict carries, which `from_config` and `MemberPackage` both depend
        on.
        """
        return self.surgery.low_rank(rank, alpha)

    def applied(self) -> dict[str, object]:
        """WHAT WAS ACTUALLY DONE to this model, read off the module tree rather than off a flag.

        The model answers this because the model is the authority: `low_rank` picks layers by structure
        and can find none, `recompute_activations` delegates to the backbone, and compilation happens
        IN PLACE so nothing but the module records it. A caller that read the request instead would
        agree with itself on an encoder that ignored all three — see `kneemri.training.landed`, which
        compares this against what the run asked for before a card is spent.

        `_compiled_call_impl` is the only place an in-place `compile()` leaves a mark, the module not
        being rebound to an `OptimizedModule` so that parameter names stay loadable.

        DEPTH_REGIONS IS THE COUNT `over_depth` WILL ACTUALLY SPLIT ON, which is not the design's field
        whenever another aggregator owns the slice axis: `slot_features` reaches `over_depth` only when
        no `SLICE_POOLINGS` aggregator was built and each slice is encoded alone, so `--slice-regions 4`
        beside `--slice-pooling attend` splits nothing and the run records a region count it never used.
        That was silent for the whole life of the flag — no error, no log line — and it is why
        `fuse_attend` and `fuse_regions` had to be two arms. Reporting the EFFECTIVE count is what lets
        `landed` refuse the combination before the card is spent instead of after.
        """
        aggregated_elsewhere = self.slices is not None or not self.per_slice
        return {
            ADAPTERS: self.surgery.adapters(),
            RECOMPUTING: bool(getattr(self.backbone, "is_gradient_checkpointing", False)),
            COMPILED_ENCODER: self.backbone._compiled_call_impl is not None,  # noqa: SLF001 - see above
            OPENED_BLOCKS: self.surgery.opened_blocks(),
            DEPTH_REGIONS: 1 if aggregated_elsewhere else self.design.slice_regions,
        }

    def context_width(self) -> int:
        """How wide one finding's reading is in `Readout.context` — the student side of a projection.

        A method rather than a caller reaching into the head, because that is what the coupling gate
        calls navigating into a stranger, and because the width is a fact about the READOUT.

        It is the head's OUTPUT width, not `self.head_width`, which is what the head is FED. Those two
        differ by the slot projection (2048 in, 256 out on the attend head), and returning the input one
        built every feature-distillation projection with the wrong student side: `nn.Linear(2048, 1024)`
        fed a 256-wide tensor, so `raptor_feat_pre` died at its first step with `mat1 and mat2 shapes
        cannot be multiplied (1x256 and 2048x1024)` and no distillation arm had ever run.
        """
        return self.head.hidden

    def parameter_groups(
        self,
        head_lr: float,
        backbone_lr: float,
        unfreeze_last: int,
        layer_decay: float = NO_LAYER_DECAY,
        attached: list[torch.nn.Parameter] | None = None,
    ) -> list[dict[str, object]]:
        """Freeze the encoder except its last blocks, and give what remains its learning rate — per DEPTH.

        The model owns this rather than the trainer: which blocks exist, and therefore which of them can
        be opened, is a fact about the encoder, not about the fitting procedure. It owns only the
        COMPOSITION — the readout at its own rate, then whatever the encoder says its groups are, which
        is `EncoderSurgery.backbone_groups` and carries the whole depth argument with it.

        `attached` are parameters owned by something OUTSIDE this module that must still be stepped at
        the head's rate — today only a distillation term's projection. They join the head group rather
        than a group of their own because `Trainer.encoder_rate_decay` reads `param_groups[1:]` as the
        backbone and `one_cycle` sizes its `max_lr` list off this list; a trailing group corrupts both.
        Composed here for the reason the rest of this method is: the group layout is one fact, and a
        caller assembling its own would be a second copy of it, free to disagree.
        """
        head: dict[str, object] = {"params": [*self._readout_parameters(), *(attached or [])], "lr": head_lr}
        return [head, *self.surgery.backbone_groups(backbone_lr, unfreeze_last, layer_decay)]

    def _readout_parameters(self) -> list[torch.nn.Parameter]:
        """Every parameter OUTSIDE the encoder — the slot head, and whatever pools the slice axis.

        Read this by EXCLUSION rather than by naming the modules, because naming them is what failed.
        This group was `self.head.parameters()` for as long as `SliceAttention` existed, and the
        aggregator is a SIBLING of the head, not a child of it — so its tensors sat in no param group
        and the optimiser never updated them. The gate is zero-initialised on purpose so that an
        untrained pooling is exactly the mean over slices, which is why the defect never showed up as a
        crash or even as a bad score: it showed up as `attend` scoring like a mean, because it WAS one.
        See knee-7gyw for which ledger entries that invalidates.

        Exclusion also means the next sibling module cannot go missing the same way — under LoRA the
        adapters live inside the backbone and are correctly left to the backbone group.
        """
        encoder = {id(parameter) for parameter in self.backbone.parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in encoder]

    def compile_encoder(self) -> None:
        """Fuse the encoder's kernels — see `EncoderSurgery.compile_encoder` for what and why."""
        self.surgery.compile_encoder()

    def recompute_activations(self) -> None:
        """Trade throughput for the encoder's activation memory — see `EncoderSurgery`."""
        self.surgery.recompute_activations()

    def _tokens(self, x: Float[Tensor, "n *pixels"]) -> Float[Tensor, "n t d"]:
        """The block's tokens the design asks for, in one forward pass.

        `output_hidden_states` returns every block's output, so reading an intermediate layer costs no
        second pass — the tensors already exist and the last one is exactly `last_hidden_state`. The
        default therefore takes the cheap path and produces byte-identical results to before.
        """
        # A VOLUMETRIC TRUNK TAKES A DIFFERENT KEYWORD, and passing the wrong one is not a silent
        # failure — transformers rejects an unknown keyword — but it is a failure deep inside a fetched
        # model rather than here, so the branch is stated where the shape is known.
        keyword = "pixel_values_videos" if self.reading.volume else "pixel_values"
        # AND A FIXED-GRID TRUNK TAKES ONE MORE, for the same reason and with the same failure mode one
        # step earlier — see `GridFit`, which reads it off the trunk rather than off the registry.
        extra = self.reading.extra()
        if self.design.layer == -1:
            return self.backbone(**{keyword: x}, **extra).last_hidden_state
        return self.backbone(**{keyword: x}, **extra, output_hidden_states=True).hidden_states[self.design.layer]

    def _pool(self, tokens: Float[Tensor, "n t d"]) -> Float[Tensor, "n f"]:
        """Reduce this read's tokens to one vector, by the design's named pooling."""
        # PREFIX TOKENS, not "the first one". DINOv2 puts a single class token in front of the patches;
        # DINOv3 adds four REGISTER tokens, so `tokens[:, 1:]` would fold them into the patch mean —
        # and registers exist precisely to absorb high-norm outliers, which is the worst thing to
        # average and the worst thing to hand a top-k. The count comes off the backbone's own config,
        # so a v2 member is unchanged (no registers, prefix of one) and nothing re-fingerprints.
        return TOKEN_POOLINGS[self.pool].reduce(tokens, self.reading.prefix_tokens)


@dataclass(frozen=True)
class Readout:
    """One training pass's three answers, which one encoder pass produces together.

    A TYPE rather than a widening tuple, because the tuple was already two long and every term a loss
    is allowed to read has to reach the trainer through here. The encoder is essentially the whole cost
    of a step, so recomputing it to obtain a second view would double training time to add an auxiliary
    loss — which is the argument `slot_features` was split out for, stated once and now carried by a
    field instead of by a position.

    Nothing on the INFERENCE path constructs one: `forward` still returns the logits alone, so the
    kernel, the fingerprint and every published member read exactly the contract they always did.
    """

    logits: Float[Tensor, "b o"]
    report: Float[Tensor, "b r"]
    context: Float[Tensor, "b o h"]
    """What each diagnosis read from the study before `out` collapsed it onto one direction.

    The representation, not the decision — the only tensor here with an axis a contrastive term can
    shape. It is DETACHED from nothing and carries a gradient, so a loss written against it trains the
    head and the encoder that produced it.
    """
