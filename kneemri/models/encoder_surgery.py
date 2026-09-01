"""What is DONE TO an encoder before it is fitted: adapters, freezing, compilation, recompute.

None of this is the forward pass. `StudyModel` used to hold both, which made `model.py` the largest
file in the tree and put two unrelated reasons to edit it in one class: a change to how a study's slots
are pooled and a change to how the encoder's blocks are opened would both land here.

EVERY RULE IS STRUCTURAL, never a name list or a config path. A DINOv2 keeps its blocks at
`backbone.encoder.layer` and calls its projections query/key/value/dense; a DINOv3 keeps them at
`model.layer` and calls them q_proj/k_proj/v_proj/o_proj. Both name-based rules have already failed
here, and each failed the same way: it adapted or opened NOTHING, and the arm reported a clean number
for an encoder it never touched.
"""

import logging
from collections import defaultdict
from collections.abc import Iterator

import torch
from torch import nn

from kneemri.models.lora import LoRALinear

logger = logging.getLogger(__name__)

NO_LAYER_DECAY = 1.0
"""The decay that means "do not decay" — and, more importantly, "keep the two-group shape EXACTLY".

Every number in the ledger was measured with the head in one group and every opened block in a second.
A refactor that emits one group per block at a shared rate reproduces that only NUMERICALLY, and an
optimiser-state or weight-decay difference between the two shapes is not something a score can rule
out afterwards. So the flat case returns the old single group untouched and the decay is a branch.
"""


class EncoderSurgery:
    """One backbone, and every operation that changes what the optimiser will be handed."""

    def __init__(self, backbone: nn.Module) -> None:
        self.backbone = backbone

    def blocks(self) -> list[nn.Module]:
        """The encoder's repeated transformer blocks, found by structure rather than by a path.

        `backbone.encoder.layer` was hardcoded, which is where a DINOv2 keeps them and NOT where a
        DINOv3 does — its tree is `embeddings / rope_embeddings / model.layer / norm`, with no
        `encoder` at all. A registry field naming the path per encoder would work and would be one more
        value that travels in a config while some consumer reads a default: the bug shape this project
        has now hit seven times.

        The blocks are instead every module of the encoder's most repeated ELEMENT TYPE, in tree order.
        "The longest single `ModuleList`" was the earlier rule, and it is correct for a flat trunk and
        wrong for a staged one: Swin holds its 24 `SwinLayer` as 2/2/18/2 across four stages, so the
        longest list is stage 2's eighteen and `--unfreeze-last 6` opened `layers.2.blocks.12-17` while
        stage 3 — the widest, highest-level blocks in the network — stayed frozen for the whole run.
        ConvNeXt has the same shape (27 of 36). Both families were measured that way, and the docstring
        above had already named the cost: a clean number for an encoder whose top was never touched.

        Grouping by element type finds the same objects in the same order for a flat trunk, where one
        list holds all twelve, so no number already in the ledger moves. Ambiguity — two types tied on
        total count — still raises rather than picking, for the reason it always did.
        """
        families: dict[str, list[nn.Module]] = defaultdict(list)
        for holder in self.backbone.modules():
            members = self._repeated(holder)
            if members:
                families[type(members[0]).__name__].extend(members)
        if not families:
            message = f"{type(self.backbone).__name__} has no module container, so it has no blocks to unfreeze"
            raise ValueError(message)
        deepest = max(len(members) for members in families.values())
        candidates = sorted(name for name, members in families.items() if len(members) == deepest)
        if len(candidates) > 1:
            message = (
                f"{type(self.backbone).__name__} has {len(candidates)} block types of {deepest} "
                f"({', '.join(candidates)}), so its blocks are ambiguous"
            )
            raise ValueError(message)
        return families[candidates[0]]

    @staticmethod
    def _repeated(holder: nn.Module) -> list[nn.Module]:
        """The repeated children of a bare container, or nothing for a module that computes.

        Both container types are here because a `ModuleList` is where a transformer keeps its blocks and
        a `ModuleDict` is where a torchvision DenseNet keeps its dense layers — `_DenseBlock` subclasses
        `ModuleDict`, so a DenseNet-121 holds 58 `_DenseLayer` in four dicts and NO `ModuleList` at all,
        which read as "no blocks to unfreeze" and refused the whole arm at construction.

        `Sequential` is deliberately not a third case. It is a container that also defines a forward, so
        it wraps stages, stems and single layers as readily as repeated blocks, and counting its children
        would make the most-repeated-type vote turn on packaging rather than on structure.
        """
        if isinstance(holder, nn.ModuleDict):
            return list(holder.values())
        if isinstance(holder, nn.ModuleList):
            return list(holder)
        return []

    def adapters(self) -> int:
        """How many LoRA adapters this encoder currently carries — zero when it was never adapted.

        The question every branch here really asks. Under adaptation the base weights are all frozen,
        so a depth argument applies to nothing, and a caller that decided by reading the REQUEST rather
        than the module would agree with itself about an encoder that ignored it.
        """
        return sum(1 for module in self.backbone.modules() if isinstance(module, LoRALinear))

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
        for block in self.blocks():
            for module in list(block.modules()):
                for name, child in list(module.named_children()):
                    if isinstance(child, nn.Linear):
                        setattr(module, name, LoRALinear(child, rank, alpha))
                        adapted += 1
        if adapted == 0:
            message = f"{type(self.backbone).__name__} has no Linear layers inside its blocks to adapt"
            raise ValueError(message)
        return adapted

    def opened_blocks(self) -> int:
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
        return sum(1 for block in self.blocks() if next(self._base_parameters(block), None) is not None)

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

    def open_blocks(self, unfreeze_last: int) -> list[nn.Module]:
        """The last `unfreeze_last` blocks, counted FROM THE FRONT so that zero means zero.

        Spelled once because two callers now need it, and the arithmetic is the one `last_blocks`
        documents as having read `[-0:]` — the whole encoder — under a head-only name.
        """
        blocks = self.blocks()
        return blocks[len(blocks) - min(unfreeze_last, len(blocks)) :]

    def opened(self, unfreeze_last: int) -> list[torch.nn.Parameter]:
        """The backbone parameters this run fits, having frozen the rest.

        WHAT IS OPEN DIFFERS BY ADAPTATION, what the optimiser is handed does not — the two branches
        used to build the same pair of groups apiece, which is one place for the head's learning rate
        to drift from the other's.
        """
        if self.adapters():
            return [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]
        return self.last_blocks(unfreeze_last)

    def last_blocks(self, unfreeze_last: int) -> list[torch.nn.Parameter]:
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
        opened = self.open_blocks(unfreeze_last)
        trainable = {id(parameter) for block in opened for parameter in block.parameters()}
        for parameter in self.backbone.parameters():
            if id(parameter) not in trainable:
                # torch takes this positionally; ruff's boolean-trap rule and the C signature disagree.
                parameter.requires_grad_(False)  # noqa: FBT003
        return [parameter for parameter in self.backbone.parameters() if id(parameter) in trainable]

    def backbone_groups(
        self, backbone_lr: float, unfreeze_last: int, layer_decay: float = NO_LAYER_DECAY
    ) -> list[dict[str, object]]:
        """The optimiser groups the ENCODER contributes: what is open, at which rate, per DEPTH.

        The encoder owns this rather than the model: which blocks exist, which of them can be opened,
        and what a depth-scaled rate means are all facts about the encoder rather than about the
        readout stacked on top of it. The model composes it with the head's own group.

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

        `NO_LAYER_DECAY` keeps the flat shape EXACTLY, rather than emitting per-block groups that all
        happen to share a rate. Every number in the ledger was measured under that shape, and a
        refactor that reproduces it only numerically is not one an equality check can confirm.
        """
        if layer_decay == NO_LAYER_DECAY:
            return [{"params": self.opened(unfreeze_last), "lr": backbone_lr}]
        if self.adapters():
            # LOUD, for the reason `objective` refuses a pair memory with no ranking term: under LoRA
            # what trains is adapters spread over every block, so a per-depth rate over the BASE blocks
            # would be applied to nothing and the run would train the flat recipe under this arm's name.
            message = (
                f"--layer-decay {layer_decay} scales a rate by BLOCK DEPTH and this encoder carries "
                "LoRA adapters, so the base blocks it would scale are frozen"
            )
            raise ValueError(message)
        self.last_blocks(unfreeze_last)
        opened = self.open_blocks(unfreeze_last)
        # The TOP block keeps `backbone_lr` and everything below it is divided down, rather than the
        # bottom keeping it and the top scaled up: the rate in the ledger is the rate the top block was
        # measured at, so this holds that fixed and only ever slows the layers underneath.
        return [
            {"params": list(block.parameters()), "lr": backbone_lr * layer_decay ** (len(opened) - 1 - depth)}
            for depth, block in enumerate(opened)
        ]

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
