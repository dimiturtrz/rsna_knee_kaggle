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

from kneemri.corpus.anatomy import ANATOMY, SlotAffinity
from kneemri.corpus.slots import SLOTS
from kneemri.models.lora import LoRALinear
from kneemri.models.slice_attention import SliceAttention
from kneemri.models.slot_head import ATTENDED, SLOT_POOLINGS

TARGETS: tuple[str, ...] = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
"""The twelve findings, in the column order the submission and the trained members both use."""

CLS_MEAN = "cls_mean"
"""Class token and patch mean — what every member on disk was fitted with."""

CLS_MEAN_FOCAL = "cls_mean_focal"
"""Those two plus the per-channel upper tail, for findings that occupy a small part of the field."""

POOL_PARTS: dict[str, int] = {CLS_MEAN: 2, CLS_MEAN_FOCAL: 3}
"""How many `dim`-wide parts each pooling scheme concatenates, which fixes the head's input width.

THE VOCABULARY as well as the widths: a name here is a name `_pool` builds parts for, and the count
must match what it appends or the head is built for a width the forward pass does not produce. The two
strings had five more copies — the design's default, the branch in `_pool`, two argv defaults and the
manifest's — and the manifest's was a LITERAL, so a run's own pooling could not reach the field the
serve path rebuilds the design from."""

ATTEND = "attend"
"""The one slice pooling that carries a learned module, so it is the one spelled in more than one place."""

CHANNELS = "channels"
"""Three consecutive slices become the encoder's three channels — one forward per slot, and the group's
physical thickness is coupled to the slice count."""

MAXIMUM = "max"
"""Each slice encoded ALONE, the slot taking the per-dimension maximum. Three forwards, no coupling."""

SLICE_POOLINGS: dict[str, bool] = {CHANNELS: False, MAXIMUM: True, ATTEND: True}
"""The vocabulary, mapped to whether a slice is encoded ALONE — the fact three call sites each decided.

An unknown name USED to mean channel stacking, because `per_slice` asked `in {"max", "attend"}` and
anything else fell through. `members.py` and `holdout_predictions.py` read this string out of a
manifest and hand it straight to `StudyDesign`, so a stale or misspelled name rebuilt a different
function on different pixels and loaded without complaint. The lookup now happens where the model is
BUILT, so it raises there instead."""

COMPILED_PREFIX = "_orig_mod."
"""What `torch.compile` inserts into a state-dict key when it is REBOUND over a module attribute.

Spelled once, and read only by `load_trained`, which strips it so the five members written before
`compile_encoder` switched to in-place compilation can still be loaded.
"""

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

FINGERPRINT_SEED = 2026
"""The seed of the synthetic bag. A constant of the question, not of the member that answers it."""

FINGERPRINT_BATCH = 2
"""Two studies, so the masked branch of the softmax is exercised alongside the unmasked one."""

_FOCAL_FRACTION = 8
"""The focal pooling keeps the top 1/8 of each channel's patch responses."""


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
        self.prefix_tokens = 1 + int(getattr(config, "num_register_tokens", 0) or 0)
        width = int(getattr(config, "hidden_size", dim) or dim)
        if width != dim:
            # A registry that disagrees with the weights is a typo, and it surfaces otherwise as a shape
            # error deep inside the head - or not at all, if the head happens to accept the width.
            message = f"encoder emits {width}-wide features and the registry says {dim}"
            raise ValueError(message)
        # The prior is defined over the canonical six slots, so asking for it with a different slot
        # count is a mistake rather than something to silently truncate — a (12, 6) table broadcast
        # against a (12, 2) attention map would either throw deep inside the head or, worse, not.
        if self.design.anatomical_prior and n_slot != len(SLOTS):
            message = f"anatomical prior covers {len(SLOTS)} slots, model has {n_slot}"
            raise ValueError(message)
        prior = SlotAffinity(ANATOMY).matrix(TARGETS, SLOTS) if self.design.anatomical_prior else None
        head_width = dim * POOL_PARTS[self.pool]
        # Built only for the design that uses it: an unused module would still land in the state dict
        # and every published member would fail to load against it.
        # The lookup, not just a read: an unknown pooling raises HERE, before weights are fetched and
        # before a manifest's stale name can be served as channel stacking.
        self._encodes_each_slice = SLICE_POOLINGS[self.slice_pooling]
        self.slices = SliceAttention(head_width) if self.slice_pooling == ATTEND else None
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
        return self.slices(stacked) if self.slices is not None else stacked.amax(dim=2)

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
        if per_slice:
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
        tokens = self._tokens(x)
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
        """This model's output on a fixed synthetic bag, as a portable identity.

        Weights read through the wrong preprocessing produce predictions, not errors: the submission is
        well formed, the log says nothing, and the difference is a number no output of the run reveals.
        So the input is generated from a seed rather than read — identical on any machine — and pushed
        through the whole forward path: byte scaling, ImageNet normalisation, resize, encoder, slot
        attention. Any of those differing moves the output by order one.

        float32 throughout: autocast would make the value depend on which device happened to run it,
        and the point of the number is that it does not.
        """
        generator = torch.Generator().manual_seed(FINGERPRINT_SEED)
        imgs = torch.randint(
            0,
            256,
            (FINGERPRINT_BATCH, self.n_slot, group, image_size, image_size),
            generator=generator,
            dtype=torch.uint8,
        ).to(device)
        mask = torch.ones(FINGERPRINT_BATCH, self.n_slot, device=device)
        mask[1, -1] = 0.0
        was_training = self.training
        self.eval()
        with torch.no_grad():
            out = self(imgs, mask, image_size).float().cpu().numpy()
        if was_training:
            self.train()
        return out

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

    def parameter_groups(self, head_lr: float, backbone_lr: float, unfreeze_last: int) -> list[dict[str, object]]:
        """Freeze the encoder except its last blocks, and give what remains its own learning rate.

        The model owns this rather than the trainer: which blocks exist, and therefore which of them can
        be opened, is a fact about the encoder, not about the fitting procedure.

        UNDER LoRA the depth argument does not apply: every block carries an adapter and the base
        weights are all frozen, so what opens is exactly the adapters. Honouring `unfreeze_last` as well
        would open six blocks' base weights alongside them and quietly make the run something else.
        """
        return [
            {"params": list(self.head.parameters()), "lr": head_lr},
            {"params": self._opened(unfreeze_last), "lr": backbone_lr},
        ]

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
        """Everything outside the last `unfreeze_last` blocks frozen, and what is left returned."""
        trainable = {id(parameter) for block in self._blocks()[-unfreeze_last:] for parameter in block.parameters()}
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

    def _tokens(self, x: Float[Tensor, "n c h w"]) -> Float[Tensor, "n t d"]:
        """The block's tokens the design asks for, in one forward pass.

        `output_hidden_states` returns every block's output, so reading an intermediate layer costs no
        second pass — the tensors already exist and the last one is exactly `last_hidden_state`. The
        default therefore takes the cheap path and produces byte-identical results to before.
        """
        if self.design.layer == -1:
            return self.backbone(pixel_values=x).last_hidden_state
        return self.backbone(pixel_values=x, output_hidden_states=True).hidden_states[self.design.layer]

    def _pool(self, tokens: Float[Tensor, "n t d"]) -> Float[Tensor, "n f"]:
        """Reduce patch tokens to one vector: the class token, the patch mean, optionally the tail."""
        # PREFIX TOKENS, not "the first one". DINOv2 puts a single class token in front of the patches;
        # DINOv3 adds four REGISTER tokens, so `tokens[:, 1:]` would fold them into the patch mean —
        # and registers exist precisely to absorb high-norm outliers, which is the worst thing to
        # average and the worst thing to hand a top-k. The count comes off the backbone's own config,
        # so a v2 member is unchanged (no registers, prefix of one) and nothing re-fingerprints.
        patch = tokens[:, self.prefix_tokens :]
        parts = [tokens[:, 0], patch.mean(1)]
        if self.pool == CLS_MEAN_FOCAL:
            # The upper tail of each channel over the patch grid, taken per channel rather than by
            # selecting whole patches: a finding occupies a small part of the field, so a plain mean
            # over the patches dilutes it, and this keeps the top eighth of each channel's responses.
            k = max(1, patch.shape[1] // _FOCAL_FRACTION)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        return torch.cat(parts, dim=1)
