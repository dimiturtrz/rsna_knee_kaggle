"""What is done TO an encoder, asked of a backbone built here rather than of a whole StudyModel."""

import pytest
import torch
from torch import nn

from kneemri.models.encoder_surgery import (
    ADAPTERS,
    FROZEN_TOP,
    NO_OFFSET,
    OPENED_BLOCKS,
    STEM_OPEN,
    BlockWindow,
    EncoderSurgery,
)
from kneemri.models.lora import LoRALinear

DIM = 8
BLOCKS = 12
STAGE_DEPTHS = (2, 2, 18, 2)
SEED = 2026


class _Backbone(nn.Module):
    """Blocks under `encoder.layer`, each holding three Linears — attention's two and an MLP.

    The shape is DINOv2's, and it is the one every rule here must find WITHOUT reading those names:
    the deepest ModuleList is the blocks, and a Linear is a Linear whatever its parent calls it.
    """

    def __init__(self, depth: int = BLOCKS) -> None:
        super().__init__()
        self.embeddings = nn.Parameter(torch.zeros(1, DIM))
        self.encoder = nn.Module()
        blocks = []
        for _ in range(depth):
            attention = nn.Module()
            attention.add_module("query", nn.Linear(DIM, DIM))
            attention.add_module("value", nn.Linear(DIM, DIM))
            block = nn.Module()
            block.add_module("attention", attention)
            block.add_module("mlp", nn.Linear(DIM, DIM))
            blocks.append(block)
        self.encoder.layer = nn.ModuleList(blocks)
        self.is_gradient_checkpointing = False
        self.checkpointing: dict[str, object] | None = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, object]) -> None:
        """Stand-in for the transformers hook, recording how it was asked."""
        self.checkpointing = gradient_checkpointing_kwargs
        self.is_gradient_checkpointing = True


def _surgery(depth: int = BLOCKS) -> EncoderSurgery:
    torch.manual_seed(SEED)
    return EncoderSurgery(_Backbone(depth))


class _StagedBlock(nn.Module):
    """One block of a staged trunk, given its own type because that type is what groups them."""

    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Linear(DIM, DIM)


class _Stage(nn.Module):
    """A stage holding its own `blocks` list — Swin's `layers.N.blocks`, ConvNeXt's `stages.N.layers`."""

    def __init__(self, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_StagedBlock() for _ in range(depth)])


class _StagedBackbone(nn.Module):
    """Swin's shape: 24 blocks as 2/2/18/2, so no single list holds more than eighteen of them."""

    def __init__(self, depths: tuple[int, ...] = STAGE_DEPTHS) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Stage(depth) for depth in depths])


def test_slice_shift():
    """One gate pair per shifted block, hooked on the LAST `count` blocks and on no other."""
    surgery = _surgery()
    shift = surgery.slice_shift(2, DIM)
    assert tuple(shift.gates.shape) == (2, 2, DIM)
    assert not shift.gates.any(), "zero gates, so a shifted run starts at the control's function"
    hooked = [bool(block._forward_pre_hooks) for block in surgery.blocks()]
    assert hooked == [False] * (BLOCKS - 2) + [True, True]


def test_blocks():
    """The most repeated element type, wherever it hangs — not `encoder.layer`, which is one path."""
    surgery = _surgery()
    assert len(surgery.blocks()) == BLOCKS

    moved = _Backbone()
    moved.model = nn.Module()
    moved.model.layer = moved.encoder.layer
    del moved.encoder
    assert len(EncoderSurgery(moved).blocks()) == BLOCKS, "a DINOv3 tree keeps its blocks somewhere else"
    assert EncoderSurgery(moved).blocks() == list(moved.model.layer), (
        "a flat trunk must come back byte-identical: every ledger number was measured under the old rule"
    )

    staged = _StagedBackbone()
    found = EncoderSurgery(staged).blocks()
    assert len(found) == sum(STAGE_DEPTHS), "the longest list holds 18 of 24; the other three stages count"
    assert found[-1] is staged.layers[-1].blocks[-1], (
        "and the LAST block must be the last stage's — opening the top is the whole point of unfreeze_last"
    )


class _TimmShaped(nn.Module):
    """timm's ViT shape: the blocks live in a `Sequential`, and the tree owns no list or dict at all."""

    def __init__(self) -> None:
        super().__init__()
        self.patch_embed = nn.Sequential(nn.Conv2d(1, DIM, 1), nn.BatchNorm2d(DIM), nn.GELU())
        self.blocks = nn.Sequential(*[_StagedBlock() for _ in range(BLOCKS)])


def test_blocks_reads_a_homogeneous_sequential():
    """A timm trunk keeps twelve blocks in a Sequential; the blanket refusal failed every timm arm."""
    shaped = _TimmShaped()
    found = EncoderSurgery(shaped).blocks()
    assert found == list(shaped.blocks), "the Sequential holding one repeated type IS the block stack"
    assert all(not isinstance(module, nn.Conv2d) for module in found), (
        "and the mixed stem beside it is packaging, which is the reason Sequential was refused outright"
    )


def test_blocks_refuses_what_it_cannot_read():
    """Nothing to find and TWO things to find are both refusals: opening the wrong six is silent."""
    with pytest.raises(ValueError, match="no module container"):
        EncoderSurgery(nn.Linear(DIM, DIM)).blocks()

    tied = _Backbone(depth=2)
    tied.other = nn.ModuleList([nn.Linear(DIM, DIM), nn.Linear(DIM, DIM)])
    with pytest.raises(ValueError, match="ambiguous"):
        EncoderSurgery(tied).blocks()


def test_adapters():
    """Read off the tree, so a caller cannot agree with itself about an encoder that ignored the ask."""
    surgery = _surgery()
    assert surgery.adapters() == 0, "an encoder nobody adapted carries none"
    surgery.low_rank(rank=2)
    assert surgery.adapters() == 3 * BLOCKS


def test_low_rank():
    """Every Linear in every block wrapped, and everything else FROZEN — the two arms this cost."""
    surgery = _surgery()
    assert surgery.low_rank(rank=2) == 3 * BLOCKS, "three Linears in each block, attention and MLP alike"

    trainable = [name for name, parameter in surgery.backbone.named_parameters() if parameter.requires_grad]
    assert trainable, "the adapters must train"
    assert all(".down." in name or ".up." in name for name in trainable), (
        "everything that is not an nn.Linear - embeddings, norms - is invisible to LoRALinear and must "
        "be frozen by hand, or it trains at sixty times the fine-tuning rate"
    )
    assert any(isinstance(module, LoRALinear) for module in surgery.backbone.modules())


def test_low_rank_refuses_a_backbone_with_no_linears():
    """Adapting nothing and reporting a number looks exactly like adapting something and working."""
    empty = _Backbone(depth=2)
    empty.encoder.layer = nn.ModuleList([nn.Identity(), nn.Identity()])
    with pytest.raises(ValueError, match="no Linear layers"):
        EncoderSurgery(empty).low_rank(rank=2)


def test_opened_blocks():
    """BASE weights, so an adapted encoder answers ZERO rather than the twelve its adapters would give."""
    surgery = _surgery()
    assert surgery.opened_blocks() == BLOCKS, "nothing frozen yet, so every block holds a trainable weight"

    surgery.last_blocks(surgery.window(2))
    assert surgery.opened_blocks() == 2, "what `unfreeze_last` actually did, read after the freezing"

    adapted = _surgery()
    adapted.low_rank(rank=2)
    assert adapted.opened_blocks() == 0, "an adapter is not an opened base weight"


def test_open_blocks():
    """The last n counted FROM THE FRONT, so zero means zero and more than there are means all."""
    surgery = _surgery()
    assert len(surgery.open_blocks(surgery.window(0))) == 0, "`blocks[-0:]` is the whole encoder - the bug this fixes"
    assert len(surgery.open_blocks(surgery.window(2))) == 2
    assert surgery.open_blocks(surgery.window(2)) == surgery.blocks()[-2:]
    assert len(surgery.open_blocks(surgery.window(99))) == BLOCKS, "asking for more than exist opens every block"

    # An INTERIOR window opens the same NUMBER of blocks somewhere else entirely, which is the whole
    # point of the offset: `unfreeze_last` alone cannot say WHERE.
    assert surgery.open_blocks(surgery.window(2, 4)) == surgery.blocks()[-6:-4]


def test_opened():
    """One list either way: the adapters when adapted, the last blocks when not."""
    surgery = _surgery()
    assert len(surgery.opened(surgery.window(2))) == 2 * 3 * 2, "two blocks, three Linears, a weight and a bias apiece"

    adapted = _surgery()
    adapted.low_rank(rank=2)
    handed = adapted.opened(adapted.window(6))
    assert handed and all(parameter.requires_grad for parameter in handed)
    assert len(handed) == 2 * 3 * BLOCKS, "the depth is IGNORED under adaptation - every block's two adapter tensors"


def test_last_blocks():
    """It FREEZES as it reads: what comes back is trainable and everything outside it is not."""
    surgery = _surgery()
    opened = surgery.last_blocks(surgery.window(2))
    assert all(parameter.requires_grad for parameter in opened)
    frozen = [parameter for parameter in surgery.backbone.parameters() if not parameter.requires_grad]
    assert len(frozen) == 2 * 3 * (BLOCKS - 2) + 1, "every earlier block's tensors, plus the embedding"

    assert surgery.last_blocks(surgery.window(0)) == [], "a head-only run opens NOTHING, whatever `[-0:]` said"


def test_backbone_groups():
    """Flat stays ONE group at one rate; a decay emits one per opened block, the TOP one undecayed."""
    surgery = _surgery()
    flat = surgery.backbone_groups(backbone_lr=8e-6, window=surgery.window(2))
    assert len(flat) == 1 and flat[0]["lr"] == 8e-6, "the shape every ledger number was measured under"

    graded_surgery = _surgery()
    graded = graded_surgery.backbone_groups(backbone_lr=8e-6, window=graded_surgery.window(3), layer_decay=0.5)
    assert [group["lr"] for group in graded] == [2e-6, 4e-6, 8e-6], (
        "the top block keeps the rate the ledger quotes and the layers under it are divided down"
    )


def test_backbone_groups_refuse_a_depth_rate_under_adapters():
    """The base blocks a decay would scale are frozen, so the run would be the flat recipe renamed."""
    surgery = _surgery()
    surgery.low_rank(rank=2)
    with pytest.raises(ValueError, match="LoRA adapters"):
        surgery.backbone_groups(backbone_lr=8e-6, window=surgery.window(6), layer_decay=0.75)


def test_compile_encoder():
    """IN PLACE: the attribute is the same module afterwards, so no state-dict key gains a prefix."""
    surgery = _surgery()
    before = dict(surgery.backbone.named_parameters())
    surgery.compile_encoder()
    assert surgery.backbone._compiled_call_impl is not None, "the only mark an in-place compile leaves"
    assert set(dict(surgery.backbone.named_parameters())) == set(before), (
        "five checkpoints were saved under `_orig_mod.` keys and none of them could be loaded back"
    )


def test_recompute_activations():
    """Delegated to the encoder's own hook, non-reentrant, and readable back off the module."""
    surgery = _surgery()
    surgery.recompute_activations()
    assert surgery.backbone.checkpointing == {"use_reentrant": False}
    assert surgery.backbone.is_gradient_checkpointing, "the public flag `applied` reads, not the call alone"


class _NormBackbone(nn.Module):
    """A frozen BatchNorm prefix and one trainable block, which is the shape the defect needs.

    DenseNet's, reduced: everything before the opened block is norm, and the norms inside the opened
    block move with the weights beside them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.prefix = nn.Sequential(nn.BatchNorm1d(DIM), nn.BatchNorm1d(DIM))
        self.block = nn.Sequential(nn.BatchNorm1d(DIM), nn.Linear(DIM, DIM))
        self.unparameterised = nn.BatchNorm1d(DIM, affine=False)


def test_pin_frozen_norms():
    """A frozen norm stops tracking; an OPEN one keeps tracking; and `train()` must not undo either."""
    backbone = _NormBackbone()
    for parameter in backbone.prefix.parameters():
        parameter.requires_grad_(False)
    surgery = EncoderSurgery(backbone)
    backbone.train()

    assert surgery.pin_frozen_norms() == 2, "the two frozen prefix norms, and not the open block's"
    assert not backbone.prefix[0].training, "a frozen norm rewriting its buffers is the whole defect"
    assert backbone.block[0].training, "statistics frozen under moving weights is the same bug reversed"
    assert backbone.unparameterised.training, "affine=False owns no parameter, so frozen is undecidable"


def test_pin_frozen_norms_counts_nothing_on_a_layernorm_trunk():
    """Every ViT, Swin and ConvNeXt here keeps no running statistics and legitimately reads zero."""
    assert EncoderSurgery(nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, DIM))).pin_frozen_norms() == 0


def test_of():
    """The window as a half-open slice, and the equality every ledger number was measured under.

    `open_blocks` was a strict suffix before 2026-09-05. Offset zero has to reproduce it EXACTLY - not
    approximately, and not for the middle of the range only - or every score on record silently changes
    what recipe it describes.
    """
    for last in (0, 1, 6, 12, 99):
        window = BlockWindow.of(BLOCKS, last, NO_OFFSET)
        suffix = list(range(BLOCKS))[BLOCKS - min(last, BLOCKS) :]
        assert list(range(window.lo, window.hi)) == suffix, "offset zero IS the old suffix, at every n"

    interior = BlockWindow.of(BLOCKS, 3, 4)
    assert (interior.lo, interior.hi) == (5, 8), "three blocks ending four short of the top"

    assert BlockWindow.of(BLOCKS, 6, 99) == BlockWindow(lo=0, hi=0), "an offset past the trunk opens nothing"
    assert BlockWindow.of(BLOCKS, -1, -1) == BlockWindow(lo=BLOCKS, hi=BLOCKS), "negatives clamp, they do not wrap"


def test_count():
    """How many blocks the window holds - what `landed` compares against the recipe."""
    assert BlockWindow.of(BLOCKS, 3, NO_OFFSET).count() == 3
    assert BlockWindow.of(BLOCKS, 3, 4).count() == 3, "an interior window opens the same NUMBER elsewhere"
    assert BlockWindow.of(BLOCKS, 99, NO_OFFSET).count() == BLOCKS
    assert BlockWindow.of(BLOCKS, 3, BLOCKS).count() == 0


def test_window():
    """Resolved against THIS encoder's depth, which is the only place the depth is known."""
    surgery = _surgery()
    assert surgery.window(2) == BlockWindow(lo=BLOCKS - 2, hi=BLOCKS)
    assert surgery.window(2, 3) == BlockWindow(lo=BLOCKS - 5, hi=BLOCKS - 3)
    assert surgery.window(99).count() == BLOCKS

    shallow = _surgery(depth=4)
    assert shallow.window(2) == BlockWindow(lo=2, hi=4), "the same request lands elsewhere on a shorter trunk"


def test_frozen_top():
    """WHERE the window sat, read off the weights - the half `opened_blocks` cannot see.

    An offset that reached nothing opens the same NUMBER of blocks in the wrong place, so a count alone
    would read a mechanism that never ran as a null.
    """
    surgery = _surgery()
    surgery.last_blocks(surgery.window(3, 4))
    assert surgery.opened_blocks() == 3, "the count says how many"
    assert surgery.frozen_top() == 4, "and this says where"

    suffix = _surgery()
    suffix.last_blocks(suffix.window(3))
    assert suffix.frozen_top() == NO_OFFSET, "every arm before 2026-09-05 ends at the top"

    adapted = _surgery()
    adapted.low_rank(rank=2)
    assert adapted.frozen_top() == NO_OFFSET, "no base block is open at all under LoRA"


def test_backbone_groups_anchor_at_the_top_of_the_window():
    """The ladder anchors at the TOP OPENED block, which under an interior window is not the last one.

    Otherwise an early-block arm runs every one of its blocks at `backbone_lr * decay^k` for a k the
    ledger has never seen, and its number would not be comparable with anything.
    """
    surgery = _surgery()
    graded = surgery.backbone_groups(backbone_lr=8e-6, window=surgery.window(3, 4), layer_decay=0.5)
    assert [group["lr"] for group in graded] == [2e-6, 4e-6, 8e-6], "the same ladder, one rung per opened block"


def test_backbone_groups_put_the_stem_at_the_front_of_the_ladder():
    """The ladder stays ordered by depth WITH the stem in it, because it is read positionally.

    `parameter_groups` documents the layout the readers depend on: head at 0, backbone at 1:, and a
    trailing group corrupts both the anchor and the per-rung ratio. The stem is a rung below the lowest
    open block and earlier than every one of them in the forward pass, so the front is where it keeps
    that ordering true. Appended instead, `OptimiserWitness` reads the ladder's bottom as its top and
    refuses the run: measured at `--layer-decay 0.75 --unfreeze-stem`, which reported the stem's
    7.12e-06 as the asked-for 4e-05 and the total 5.62x span as the asked-for 0.75.
    """
    surgery = _surgery()
    graded = surgery.backbone_groups(backbone_lr=8e-6, window=surgery.window(3, unfreeze_stem=True), layer_decay=0.5)
    rates = [group["lr"] for group in graded]

    assert rates == [1e-6, 2e-6, 4e-6, 8e-6], "one rung per opened block, and the stem one below the lowest"
    assert rates[-1] == 8e-6, "the anchor is the LAST group, which is what the trainer captures"
    assert rates[-2] / rates[-1] == 0.5, "and the last pair is the per-rung ratio the witness measures"


def test_stem_parameters():
    """The COMPLEMENT of the block set, so a mis-identified trunk fails loudly rather than opening nothing."""
    surgery = _surgery()
    inside = {id(parameter) for block in surgery.blocks() for parameter in block.parameters()}
    stem = surgery.stem_parameters()

    assert stem, "a ViT reads pixels somewhere, and that somewhere is not a block"
    assert not any(id(parameter) in inside for parameter in stem), "nothing in the window is in the stem"
    assert len(stem) + len(inside) == len(list(surgery.backbone.parameters())), "the two partition the trunk"


def test_stem_open():
    """Whether a BASE pre-block weight trains — the axis neither block count nor offset can express."""
    frozen = _surgery()
    frozen.last_blocks(frozen.window(3))
    assert not frozen.stem_open(), "every arm before 2026-09-16: the trainable set is built out of blocks"

    opened = _surgery()
    opened.last_blocks(opened.window(3, unfreeze_stem=True))
    assert opened.stem_open(), "and the switch is what reaches it"

    adapted = _surgery()
    adapted.low_rank(rank=2)
    assert not adapted.stem_open(), "adapters never wrap the stem, and low_rank freezes every base weight"


def test_applied():
    """Every key is a fact about the same freezing, so they are read together rather than one by one."""
    surgery = _surgery()
    surgery.last_blocks(surgery.window(3, 4, unfreeze_stem=True))
    assert surgery.applied() == {ADAPTERS: 0, OPENED_BLOCKS: 3, FROZEN_TOP: 4, STEM_OPEN: True}

    adapted = _surgery()
    adapted.low_rank(rank=2)
    assert adapted.applied() == {ADAPTERS: 36, OPENED_BLOCKS: 0, FROZEN_TOP: NO_OFFSET, STEM_OPEN: False}
