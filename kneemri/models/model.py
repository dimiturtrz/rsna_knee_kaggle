"""The study-level model: a shared image encoder under a per-diagnosis attention head.

Replicates the public baseline's architecture exactly, because the point of a replication is to
reproduce the number before changing anything — and because the trained members published alongside it
load only into this shape. `MemberPackage` checks that faithfulness numerically (see `members.py`);
this module is what it checks.
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import override

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor, nn

from kneemri.corpus.anatomy import ANATOMY, SlotAffinity
from kneemri.findings import TARGETS
from kneemri.models.depth_window import DepthWindow
from kneemri.models.fingerprint import FINGERPRINT
from kneemri.models.lora import LoRALinear
from kneemri.models.slice_pooling import CHANNELS, SLICE_POOLINGS
from kneemri.models.slot_head import ATTENDED, SLOT_POOLINGS
from kneemri.models.token_pooling import CLS_MEAN, POOL_PARTS, TOKEN_POOLINGS
from kneemri.models.volume_read import VolumeRead

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
"""The keys `applied` reports under — named beside the model because the model is what they describe."""

NO_LAYER_DECAY = 1.0
"""The decay that means "do not decay" — and, more importantly, "keep the two-group shape EXACTLY".

Every number in the ledger was measured with the head in one group and every opened block in a second.
A refactor that emits one group per block at a shared rate reproduces that only NUMERICALLY, and an
optimiser-state or weight-decay difference between the two shapes is not something a score can rule
out afterwards. So the flat case returns the old two groups untouched and the decay is a branch.
"""

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


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

    def __init__(self, backbone: nn.Module, dim: int, n_slot: int, design: StudyDesign | None = None) -> None:
        # `backbone` is any module callable as `backbone(pixel_values=x)` returning an object with
        # `last_hidden_state` of shape (n, 1 + patches, dim) — a DINOv2 from transformers in
        # production, a deterministic stand-in under test, and a medical-pretrained encoder when that
        # ablation runs. Kept structural rather than a named protocol so a substitution is a
        # constructor argument, not an edit here.
        super().__init__()
        self.design = design or StudyDesign()
        self.backbone = backbone
        self.pool = self.design.pool
        self.slice_pooling = self.design.slice_pooling
        self.n_slot = n_slot
        # The BACKBONE's own config decides these, not our registry, because the loaded weights are the
        # only authority on what they emit. DINOv2 has one class token and no registers; DINOv3 adds
        # four registers, which must be excluded from the patch pooling below.
        config = getattr(backbone, "config", None)
        # A VOLUMETRIC TRUNK ANNOUNCES ITSELF BY ITS TUBELET; `VolumeRead` is None for an ordinary 2D
        # one, and holds every fact the depth axis changes when it is not.
        self.volume = VolumeRead.of(config, self.design.pool, self.design.slice_pooling)
        # No class token to skip on a tubelet trunk, and no registers either — the first token is a
        # patch. Getting this wrong is silent: the head accepts the width and the score comes out
        # plausible, which is the failure `prefix_tokens` exists to prevent.
        self.prefix_tokens = 0 if self.volume else 1 + int(getattr(config, "num_register_tokens", 0) or 0)
        width = int(getattr(config, "hidden_size", dim) or dim)
        if width != dim:
            # A registry that disagrees with the weights is a typo, and it surfaces otherwise as a shape
            # error deep inside the head - or not at all, if the head happens to accept the width.
            message = f"encoder emits {width}-wide features and the registry says {dim}"
            raise ValueError(message)
        prior = SlotAffinity(ANATOMY).requested(TARGETS, n_slot, wanted=self.design.anatomical_prior)
        head_width = dim * POOL_PARTS[self.pool]
        # Built only for the design that uses it: an unused module would still land in the state dict
        # and every published member would fail to load against it.
        # The lookup, not just a read: an unknown pooling raises HERE, before weights are fetched and
        # before a manifest's stale name can be served as channel stacking.
        pooling = SLICE_POOLINGS[self.slice_pooling]
        self._encodes_each_slice = pooling.per_slice
        self.slices = pooling.aggregator(head_width) if pooling.aggregator is not None else None
        self.head = SLOT_POOLINGS[self.design.slot_pooling](head_width, n_slot, len(TARGETS), prior)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    @classmethod
    def from_config(cls, config: Mapping[str, object], backbone: nn.Module, dim: int, n_slot: int) -> "StudyModel":
        """The model a recorded config describes, adapters included, with no weights loaded yet.

        THE ONE REBUILD. Whatever a config says about the network belongs here, so a switch added to
        the trainer reaches every path that reconstructs a member at once instead of reaching whichever
        of them the author happened to be editing.

        The adapters are applied BEFORE the caller loads anything, because a LoRA member's state dict
        carries adapter keys that only exist once `low_rank` has wrapped the encoder's linears.
        """
        model = cls(backbone, dim, n_slot, StudyDesign.from_config(config))
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
        if self.volume is not None:
            if slices % self.volume.tubelet:
                message = (
                    f"this trunk spans {self.volume.tubelet} slices per token, so a {slices}-slice "
                    f"window leaves a partial tubelet; use a multiple of {self.volume.tubelet}"
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
    ) -> Float[Tensor, "b s d"]:
        """One embedding per acquisition slot — everything before the head reads them.

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
        # "channels": three consecutive slices become the encoder's three channels — one forward per
        # slot. "max": each slice is encoded ALONE (replicated across the three channels the encoder
        # expects) and the slot's embedding is the per-dimension maximum over its slices.
        #
        # The difference is not a preference. Channel stacking couples the group's PHYSICAL THICKNESS
        # to the slice count, which is measurably why raising inference slices from 12 to 32 cost
        # -0.018 here while the recipe that pools per-slice can do it freely. It also means a finding
        # is only visible if it falls inside a sampled three-slice group.
        per_slice = self.per_slice
        if per_slice and self.design.through_plane:
            x = DepthWindow.channels(imgs).reshape(batch * slots * group, 3, *imgs.shape[3:]).float().div(255.0)
        elif per_slice:
            x = imgs.reshape(batch * slots * group, 1, *imgs.shape[3:]).expand(-1, 3, -1, -1)
            x = x.float().div(255.0)
        elif self.volume:
            # THE WINDOW IS FRAMES, NOT CHANNELS, and that is the whole difference this trunk buys —
            # see `VolumeRead` for why no reshaping of a 2D stem reaches it. Each slice is greyscale
            # replicated to the three channels the trunk was pretrained on; only the depth axis is new,
            # and it is folded in after the resize and the normalisation so both lines are reused.
            x = imgs.reshape(batch * slots * group, 1, *imgs.shape[3:]).expand(-1, 3, -1, -1)
            x = x.float().div(255.0)
        else:
            x = imgs.reshape(batch * slots, *imgs.shape[2:]).float().div_(255.0)
        if img_size is not None and img_size != x.shape[-1]:
            # The cache is held at the highest resolution any configuration needs and the rest
            # downsample from it, so every configuration sees the same pixels through a different
            # sampling grid rather than a different crop.
            x = nn.functional.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        if self.volume:
            x = self.volume.frames(x, batch * slots)
        tokens = self._tokens(x)
        if self.volume:
            self.volume.check_tokens(x, tokens)
        pooled = self._pool(tokens)
        if not per_slice:
            return pooled.reshape(batch, slots, 1, -1)
        return pooled.reshape(batch, slots, group, -1)

    def supervised(
        self,
        imgs: UInt8[Tensor, "b s g p q"],
        mask: Float[Tensor, "b s"],
        img_size: int | None = None,
    ) -> tuple[Float[Tensor, "b o"], Float[Tensor, "b r"]]:
        """The diagnosis logits AND the predicted report embedding, from one pass over the encoder.

        Training-only, and deliberately NOT what `forward` returns: the fingerprint, the kernel and
        every published member read `forward`, so the inference contract stays exactly as it was and a
        member fitted with report supervision loads into the old path unchanged.
        """
        features = self.slot_features(imgs, img_size)
        return self.head(features, mask), self.head.report_vector(features, mask)

    def fingerprint(self, image_size: int, group: int, device: torch.device) -> NDArray[np.float32]:
        """This model's answer to the one seeded question, as a portable identity — see `Fingerprint`."""
        frames = self.volume.probe_frames(group) if self.volume else group
        return FINGERPRINT.of(self, self.n_slot, image_size, frames, device)

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

    def _blocks(self) -> list[nn.Module]:
        """The encoder's repeated transformer blocks, found by structure rather than by a path.

        `backbone.encoder.layer` was hardcoded, which is where a DINOv2 keeps them and NOT where a
        DINOv3 does — its tree is `embeddings / rope_embeddings / model.layer / norm`, with no
        `encoder` at all. A registry field naming the path per encoder would work and would be one more
        value that travels in a config while some consumer reads a default: the bug shape this project
        has now hit seven times.

        The blocks are instead the longest `ModuleList` in the tree, which is what "the layers a
        transformer repeats" means structurally in both trees and in any third one. Ambiguity raises
        rather than picking, because opening the wrong six modules trains something that is not the
        encoder's top and reports a clean number for it.
        """
        lists = [module for module in self.backbone.modules() if isinstance(module, nn.ModuleList)]
        if not lists:
            message = f"{type(self.backbone).__name__} has no ModuleList, so it has no blocks to unfreeze"
            raise ValueError(message)
        deepest = max(len(module) for module in lists)
        candidates = [module for module in lists if len(module) == deepest]
        if len(candidates) > 1:
            message = (
                f"{type(self.backbone).__name__} has {len(candidates)} ModuleLists of {deepest}, "
                "so its blocks are ambiguous"
            )
            raise ValueError(message)
        return list(candidates[0])

    def low_rank(self, rank: int, alpha: float = 1.0) -> int:
        """Adapt EVERY block along a rank-r subspace instead of adapting the last few completely.

        Every `nn.Linear` inside a block is wrapped, attention and MLP alike, chosen by STRUCTURE
        rather than by name: DINOv2 calls its projections query/key/value/dense and DINOv3 calls them
        q_proj/k_proj/v_proj/o_proj, and a name list would silently adapt nothing on the encoder it did
        not know about. That failure has already happened once here, with `encoder.layer`.

        Returns how many layers were adapted, because "silently adapted nothing" is the failure mode
        worth making impossible: an arm that trains a frozen encoder and reports a number looks exactly
        like an arm that worked.

        THE WHOLE BACKBONE IS FROZEN FIRST, and leaving that out cost two arms. `LoRALinear` freezes the
        Linear it wraps and nothing else, so everything that is not an `nn.Linear` stayed trainable:
        position embeddings (526k), the patch-embedding projection (226k), every norm, and DINOv3's
        LayerScale gammas — 781,440 parameters against the adapters' 663,552, i.e. MORE than half of
        what the optimiser opened was not an adapter at all. All of it then trained at the ADAPTER rate
        of 5e-4, sixty times the fine-tuning rate, which is not low-rank adaptation of a frozen encoder
        but a wrecking ball through the pretrained representation. Measured on the v3 arm: loss flat at
        1.23 over four epochs and 0.6867 against fine-tuned v3's 0.8057.

        THE ARM'S OWN PRE-RUN CHECK MISSED IT AND LOOKED LIKE IT PASSED: 663,552 + 781,440 = 1,444,992,
        which is the "1.44M trainable parameters, 6.4% of the encoder" the header quotes as validation.
        Counting the total without asking WHICH parameters is how a bug reports the expected number.
        """
        for parameter in self.backbone.parameters():
            # torch takes this positionally; ruff's boolean-trap rule and the C signature disagree.
            parameter.requires_grad_(False)  # noqa: FBT003
        adapted = 0
        for block in self._blocks():
            for module in list(block.modules()):
                for name, child in list(module.named_children()):
                    if isinstance(child, nn.Linear):
                        setattr(module, name, LoRALinear(child, rank, alpha))
                        adapted += 1
        if adapted == 0:
            message = f"{type(self.backbone).__name__} has no Linear layers inside its blocks to adapt"
            raise ValueError(message)
        return adapted

    def applied(self) -> dict[str, object]:
        """WHAT WAS ACTUALLY DONE to this model, read off the module tree rather than off a flag.

        The model answers this because the model is the authority: `low_rank` picks layers by structure
        and can find none, `recompute_activations` delegates to the backbone, and compilation happens
        IN PLACE so nothing but the module records it. A caller that read the request instead would
        agree with itself on an encoder that ignored all three — see `kneemri.training.landed`, which
        compares this against what the run asked for before a card is spent.

        `_compiled_call_impl` is the only place an in-place `compile()` leaves a mark, the module not
        being rebound to an `OptimizedModule` so that parameter names stay loadable.
        """
        return {
            ADAPTERS: sum(1 for module in self.modules() if isinstance(module, LoRALinear)),
            RECOMPUTING: bool(getattr(self.backbone, "is_gradient_checkpointing", False)),
            COMPILED_ENCODER: self.backbone._compiled_call_impl is not None,  # noqa: SLF001 - see above
            OPENED_BLOCKS: self._opened_blocks(),
        }

    def _opened_blocks(self) -> int:
        """How many encoder blocks hold a trainable BASE weight — what `unfreeze_last` actually did.

        BASE, not trainable-anything. Under LoRA every block carries an adapter whose parameters want
        a gradient, so counting those would report twelve blocks open on a run that froze the encoder
        completely. The adapters are reported under their own key; this answers the other question and
        answers ZERO under adaptation, which is exactly what `parameter_groups` does with the depth
        there — so the check over this needs no exception for LoRA rather than a third loose key.

        Read AFTER the freezing, which the ordering already guarantees: `Trainer.__init__` builds the
        optimiser through `parameter_groups`, and `landed` runs on the trainer that call returned.
        """
        # `next(..., None)`, not `any(...)`: the generator yields TENSORS, and `any` asks each for its
        # truth value — which a multi-element tensor refuses. Emptiness is the question here.
        return sum(1 for block in self._blocks() if next(self._base_parameters(block), None) is not None)

    @staticmethod
    def _base_parameters(block: nn.Module) -> Iterator[torch.nn.Parameter]:
        """The block's trainable parameters that are not part of an adapter."""
        adapted = {
            id(parameter)
            for module in block.modules()
            if isinstance(module, LoRALinear)
            for parameter in module.parameters()
        }
        return (
            parameter for parameter in block.parameters() if parameter.requires_grad and id(parameter) not in adapted
        )

    def parameter_groups(
        self, head_lr: float, backbone_lr: float, unfreeze_last: int, layer_decay: float = NO_LAYER_DECAY
    ) -> list[dict[str, object]]:
        """Freeze the encoder except its last blocks, and give what remains its learning rate — per DEPTH.

        The model owns this rather than the trainer: which blocks exist, and therefore which of them can
        be opened, is a fact about the encoder, not about the fitting procedure.

        UNDER LoRA the depth argument does not apply: every block carries an adapter and the base
        weights are all frozen, so what opens is exactly the adapters. Honouring `unfreeze_last` as well
        would open six blocks' base weights alongside them and quietly make the run something else.

        WHY THE DEPTH ARGUMENT EXISTS AT ALL. With ONE flat rate over every opened block, depth and rate
        can only be measured along a diagonal: raising the rate until the top blocks move necessarily
        applies that same rate to the earliest blocks, which hold the most general features and are the
        most damaging to disturb. `Adaptation` already records that entanglement and calls it unresolved
        — its answer was to raise the rate, which is the move that cannot separate them. A per-depth
        rate is the standard way the transfer literature separates them, and until now this repo did not
        have one: `--unfreeze-last 12` at 4e-5 was not a deeper version of the six-block arm, it was the
        six-block arm plus six of the encoder's most general layers being rewritten at full rate.

        `NO_LAYER_DECAY` keeps the two-group shape EXACTLY, rather than emitting per-block groups that
        all happen to share a rate. Every number in the ledger was measured under that shape, and a
        refactor that reproduces it only numerically is not one an equality check can confirm.
        """
        head: dict[str, object] = {"params": self._readout_parameters(), "lr": head_lr}
        if layer_decay == NO_LAYER_DECAY:
            flat: dict[str, object] = {"params": self._opened(unfreeze_last), "lr": backbone_lr}
            return [head, flat]
        if any(isinstance(module, LoRALinear) for module in self.backbone.modules()):
            # LOUD, for the reason `objective` refuses a pair memory with no ranking term: under LoRA
            # what trains is adapters spread over every block, so a per-depth rate over the BASE blocks
            # would be applied to nothing and the run would train the flat recipe under this arm's name.
            message = (
                f"--layer-decay {layer_decay} scales a rate by BLOCK DEPTH and this encoder carries "
                "LoRA adapters, so the base blocks it would scale are frozen"
            )
            raise ValueError(message)
        self._last_blocks(unfreeze_last)
        opened = self._open_blocks(unfreeze_last)
        # The TOP block keeps `backbone_lr` and everything below it is divided down, rather than the
        # bottom keeping it and the top scaled up: the rate in the ledger is the rate the top block was
        # measured at, so this holds that fixed and only ever slows the layers underneath.
        graded: list[dict[str, object]] = [
            {"params": list(block.parameters()), "lr": backbone_lr * layer_decay ** (len(opened) - 1 - depth)}
            for depth, block in enumerate(opened)
        ]
        return [head, *graded]

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

    def _open_blocks(self, unfreeze_last: int) -> list[nn.Module]:
        """The last `unfreeze_last` blocks, counted FROM THE FRONT so that zero means zero.

        Spelled once because two callers now need it, and the arithmetic is the one `_last_blocks`
        documents as having read `[-0:]` — the whole encoder — under a head-only name.
        """
        blocks = self._blocks()
        return blocks[len(blocks) - min(unfreeze_last, len(blocks)) :]

    def _opened(self, unfreeze_last: int) -> list[torch.nn.Parameter]:
        """The backbone parameters this run fits, having frozen the rest.

        WHAT IS OPEN DIFFERS BY ADAPTATION, what the optimiser is handed does not — the two branches
        used to build the same pair of groups apiece, which is one place for the head's learning rate
        to drift from the other's.
        """
        if any(isinstance(module, LoRALinear) for module in self.backbone.modules()):
            return [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]
        return self._last_blocks(unfreeze_last)

    def _last_blocks(self, unfreeze_last: int) -> list[torch.nn.Parameter]:
        """Everything outside the last `unfreeze_last` blocks frozen, and what is left returned.

        COUNTED FROM THE FRONT, and that is the whole point of the arithmetic. `blocks[-n:]` reads as
        "the last n" and is that for every n but ZERO, where `[-0:]` is `[0:]` — the entire encoder.
        A head-only run therefore opened all twelve blocks and was byte-identical to `--unfreeze-last
        12` (measured: 2,592 parameters either way), under a name saying the opposite.
        `unfreeze_depth.sh` takes its depth as an argument, so the head-only rung of a depth sweep was
        one invocation away from spending a queue slot on the deepest arm in the sweep.

        The `min` is the same edit's other half: asking for more blocks than the encoder has now opens
        every block rather than silently reading as a request for all of them, which is what
        `applied` reports back and `landed` compares.
        """
        opened = self._open_blocks(unfreeze_last)
        trainable = {id(parameter) for block in opened for parameter in block.parameters()}
        for parameter in self.backbone.parameters():
            if id(parameter) not in trainable:
                # torch takes this positionally; ruff's boolean-trap rule and the C signature disagree.
                parameter.requires_grad_(False)  # noqa: FBT003
        return [parameter for parameter in self.backbone.parameters() if id(parameter) in trainable]

    def compile_encoder(self) -> None:
        """Fuse the encoder's kernels. The ENCODER only, and SHAPE-SPECIALISED, both measured.

        Only the encoder: `supervised` is called directly rather than through `forward`, so compiling
        the module would leave the path training actually takes in eager mode. The encoder is the bulk
        of the arithmetic anyway — 864 images a step under whole-stack against one small head.

        NOT `dynamic=True`, which is what this did and which cannot work on this machine: inductor
        raises `InductorError: AssertionError` inside its own size reasoning, with or without gradient
        checkpointing, on every configuration tried.

        AND IT DOES NOT NEED TO BE. The ragged batch that motivated dynamic shapes was a CHOICE, not a
        property of the data: `Trainer.batches` now drops the partial last batch for training, so every
        training step has one shape and the encoder compiles ONCE. Evaluation pads its final batch to
        full width and trims the result, so the holdout stays exactly the holdout the ledger quotes.

        MEASURED 2026-08-12, batch 2 at 336 px, whole-stack, checkpointing on, single shape throughout:

            eager      steady 150 ms/step
            compiled   steady 119 ms/step        -21%, for 2.2 s of compilation

        The docstring that justified `dynamic=True` assumed a recompile per epoch. That was wrong twice
        over: shapes are compiled once per PROCESS, and the second shape did not have to exist at all.

        It needs Triton, which upstream ships for everything except Windows — `triton-windows` is
        pinned in pyproject for exactly this.
        """
        # IN PLACE, NOT REBOUND, and this is not a style preference. `self.backbone =
        # torch.compile(self.backbone)` replaces the attribute with an OptimizedModule that holds the
        # real module as `_orig_mod`, and that name lands in every state-dict key: a compiled run saves
        # `backbone._orig_mod.embeddings.cls_token`, which `load_state_dict` then refuses as unexpected.
        # FIVE checkpoints were written that way before anything tried to load one - schedule_8,
        # extended_ws and all three DINOv3 members, every GPU hour of them unloadable and none of it
        # visible, because training never reloads its own weights through the inference path.
        #
        # `nn.Module.compile()` compiles the forward and leaves the module, and therefore its
        # parameter names, exactly where they were.
        self.backbone.compile()

    def recompute_activations(self) -> None:
        """Trade 19% of throughput for nearly all of the encoder's activation memory.

        MEASURED HERE 2026-08-12, one epoch each way at batch 6 — the only batch that fits both with
        and without it: 798 img/s against 946, so the tax is 19%. The "about 30%" this docstring
        carried was inherited from the literature and never checked against this workload; it is the
        kind of number that decides whether a lever is worth pulling, so it should have been measured
        the day it was written.

        MEASURED NEED, 2026-08-11. Whole-stack reading puts 12 studies x 6 slots x 12 slices = 864
        images through the encoder in one step. At 336 px that is 577 tokens x 384 dims held for each
        of the six trainable blocks, roughly 14-20 GB of stored activations, and the arm pinned all
        32 GB of a 5090 and then THRASHED: Windows pages GPU memory over PCIe rather than failing, so
        utilisation still read 100% while power sat at 145 W of 575 and fifteen minutes produced no
        epoch at all. A card that is genuinely working draws its power budget.

        Checkpointing stores one tensor per block boundary and recomputes the rest in the backward
        pass. It is the option that changes NOTHING about the experiment - same batch, same loss, and
        in particular the same 66 within-batch pairs the ranking term scores. Splitting the batch
        instead would have quietly changed the objective, since pairs across micro-batches do not
        exist.

        Attention was checked first and is already fused (`_attn_implementation` is sdpa), so the
        memory is genuine activation volume rather than materialised attention matrices.
        """
        # pyrefly: ignore[not-callable] - the hook is transformers', not torch's, and nn.Module's
        # __getattr__ is typed as returning Tensor, so any model-specific method reads as uncallable.
        self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def _tokens(self, x: Float[Tensor, "n *pixels"]) -> Float[Tensor, "n t d"]:
        """The block's tokens the design asks for, in one forward pass.

        `output_hidden_states` returns every block's output, so reading an intermediate layer costs no
        second pass — the tensors already exist and the last one is exactly `last_hidden_state`. The
        default therefore takes the cheap path and produces byte-identical results to before.
        """
        # A VOLUMETRIC TRUNK TAKES A DIFFERENT KEYWORD, and passing the wrong one is not a silent
        # failure — transformers rejects an unknown keyword — but it is a failure deep inside a fetched
        # model rather than here, so the branch is stated where the shape is known.
        keyword = "pixel_values_videos" if self.volume else "pixel_values"
        if self.design.layer == -1:
            return self.backbone(**{keyword: x}).last_hidden_state
        return self.backbone(**{keyword: x}, output_hidden_states=True).hidden_states[self.design.layer]

    def _pool(self, tokens: Float[Tensor, "n t d"]) -> Float[Tensor, "n f"]:
        """Reduce this read's tokens to one vector, by the design's named pooling."""
        # PREFIX TOKENS, not "the first one". DINOv2 puts a single class token in front of the patches;
        # DINOv3 adds four REGISTER tokens, so `tokens[:, 1:]` would fold them into the patch mean —
        # and registers exist precisely to absorb high-norm outliers, which is the worst thing to
        # average and the worst thing to hand a top-k. The count comes off the backbone's own config,
        # so a v2 member is unchanged (no registers, prefix of one) and nothing re-fingerprints.
        return TOKEN_POOLINGS[self.pool].reduce(tokens, self.prefix_tokens)
