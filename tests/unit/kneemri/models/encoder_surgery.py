"""What is done TO an encoder, asked of a backbone built here rather than of a whole StudyModel."""

import pytest
import torch
from torch import nn

from kneemri.models.encoder_surgery import EncoderSurgery
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

    surgery.last_blocks(2)
    assert surgery.opened_blocks() == 2, "what `unfreeze_last` actually did, read after the freezing"

    adapted = _surgery()
    adapted.low_rank(rank=2)
    assert adapted.opened_blocks() == 0, "an adapter is not an opened base weight"


def test_open_blocks():
    """The last n counted FROM THE FRONT, so zero means zero and more than there are means all."""
    surgery = _surgery()
    assert len(surgery.open_blocks(0)) == 0, "`blocks[-0:]` is the whole encoder - the bug this arithmetic fixes"
    assert len(surgery.open_blocks(2)) == 2
    assert surgery.open_blocks(2) == surgery.blocks()[-2:]
    assert len(surgery.open_blocks(99)) == BLOCKS, "asking for more than exist opens every block"


def test_opened():
    """One list either way: the adapters when adapted, the last blocks when not."""
    surgery = _surgery()
    assert len(surgery.opened(2)) == 2 * 3 * 2, "two blocks, three Linears, a weight and a bias apiece"

    adapted = _surgery()
    adapted.low_rank(rank=2)
    handed = adapted.opened(6)
    assert handed and all(parameter.requires_grad for parameter in handed)
    assert len(handed) == 2 * 3 * BLOCKS, "the depth is IGNORED under adaptation - every block's two adapter tensors"


def test_last_blocks():
    """It FREEZES as it reads: what comes back is trainable and everything outside it is not."""
    surgery = _surgery()
    opened = surgery.last_blocks(2)
    assert all(parameter.requires_grad for parameter in opened)
    frozen = [parameter for parameter in surgery.backbone.parameters() if not parameter.requires_grad]
    assert len(frozen) == 2 * 3 * (BLOCKS - 2) + 1, "every earlier block's tensors, plus the embedding"

    assert surgery.last_blocks(0) == [], "a head-only run opens NOTHING, whatever `[-0:]` would have said"


def test_backbone_groups():
    """Flat stays ONE group at one rate; a decay emits one per opened block, the TOP one undecayed."""
    surgery = _surgery()
    flat = surgery.backbone_groups(backbone_lr=8e-6, unfreeze_last=2)
    assert len(flat) == 1 and flat[0]["lr"] == 8e-6, "the shape every ledger number was measured under"

    graded = _surgery().backbone_groups(backbone_lr=8e-6, unfreeze_last=3, layer_decay=0.5)
    assert [group["lr"] for group in graded] == [2e-6, 4e-6, 8e-6], (
        "the top block keeps the rate the ledger quotes and the layers under it are divided down"
    )


def test_backbone_groups_refuse_a_depth_rate_under_adapters():
    """The base blocks a decay would scale are frozen, so the run would be the flat recipe renamed."""
    surgery = _surgery()
    surgery.low_rank(rank=2)
    with pytest.raises(ValueError, match="LoRA adapters"):
        surgery.backbone_groups(backbone_lr=8e-6, unfreeze_last=6, layer_decay=0.75)


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
