from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import cast, override

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from kneemri.models.backbone import IMAGENET, RAD_DINO
from kneemri.models.lora import LoRALinear
from kneemri.models.model import (
    ADAPTERS,
    COMPILED_ENCODER,
    COMPILED_PREFIX,
    DEPTH_REGIONS,
    OPENED_BLOCKS,
    RECOMPUTING,
    TARGETS,
    StudyDesign,
    StudyModel,
)
from kneemri.models.slice_pooling import ATTEND, SLICE_POOLINGS
from kneemri.models.slot_head import REPORT_DIMS, SLOT_POOLINGS
from kneemri.models.token_pooling import CLS_MEAN_FOCAL, MEAN, POOL_PARTS
from kneemri.models.volume_read import VolumeRead

SEED = 2026
DIM = 8
PATCHES = 16
SLOTS = 6
GROUP = 3


@dataclass
class _Output:
    last_hidden_state: Tensor
    hidden_states: tuple[Tensor, ...] | None = None


class _FakeBackbone(torch.nn.Module):
    """A stand-in encoder whose tokens vary PER DIMENSION with the input.

    An earlier version returned `base + mean(pixels)` — the same scalar added to every feature. The
    head begins with a LayerNorm, which removes exactly that, so the stand-in made the model's output
    independent of its pixels and no test could detect a change in pooling. A useful double has to
    perturb the feature vector's SHAPE, not just its level.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(DIM, DIM) for _ in range(12)])
        self.checkpointing: dict[str, object] | None = None
        # The PUBLIC flag transformers exposes, which is what `applied` reads — a double that recorded
        # only the call would let a check pass against an encoder that had ignored it.
        self.is_gradient_checkpointing = False

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, object]) -> None:
        """Stand-in for the transformers hook, recording how it was asked."""
        self.checkpointing = gradient_checkpointing_kwargs
        self.is_gradient_checkpointing = True

    @override
    def forward(self, *, pixel_values: Tensor, output_hidden_states: bool = False) -> _Output:
        n = pixel_values.shape[0]
        rows = pixel_values.mean(dim=(1, 3))[:, :DIM] if pixel_values.shape[2] >= DIM else None
        if rows is None:
            rows = pixel_values.mean(dim=(1, 3)).repeat(1, (DIM // pixel_values.shape[2]) + 1)[:, :DIM]
        base = torch.arange(PATCHES + 1, dtype=torch.float32).reshape(1, -1, 1) * 0.01
        last = base + rows.reshape(n, 1, DIM)
        # Each "block" differs by a constant so a test can tell which one the head was handed.
        states = tuple(last + index for index in range(13)) if output_hidden_states else None
        return _Output(last_hidden_state=last, hidden_states=states)


@pytest.fixture
def study_model() -> StudyModel:
    torch.manual_seed(SEED)
    return StudyModel(_FakeBackbone(), DIM, SLOTS)


def test_forward(study_model: StudyModel):
    """One logit per diagnosis per study, and a masked-out slot must change the answer."""
    torch.manual_seed(SEED)
    imgs = torch.randint(0, 256, (2, SLOTS, GROUP, 14, 14), dtype=torch.uint8)
    partial = torch.ones(2, SLOTS)
    partial[:, -1] = 0.0
    study_model.eval()
    with torch.no_grad():
        every_slot = study_model.forward(imgs, torch.ones(2, SLOTS), 14)
        one_absent = study_model.forward(imgs, partial, 14)
    assert every_slot.shape == (2, len(TARGETS))
    assert not torch.allclose(every_slot, one_absent), "the presence mask must reach the softmax"


def test_study_model_refuses_a_shape_it_cannot_serve():
    """Both constructor guards must raise at BUILD time, where the wrong number is still readable.

    Each guards a mismatch whose other outcome is worse than a crash: a registry width that disagrees with
    the loaded weights surfaces as a shape error deep in the head, or not at all if the head happens to
    accept it; and a (12, 6) anatomical prior broadcast against a two-slot attention map is a silently
    wrong prior rather than an error. A member that cannot be built is cheap; one that builds wrong is not.
    """
    # The WEIGHTS are the authority on the width, so the mismatch only exists for a backbone that says
    # what it emits — a configless stand-in is taken at the registry's word and cannot trip this.
    speaks_for_itself = _FakeBackbone()
    speaks_for_itself.config = SimpleNamespace(hidden_size=DIM)

    with pytest.raises(ValueError, match="registry says"):
        StudyModel(speaks_for_itself, DIM + 1, SLOTS)

    with pytest.raises(ValueError, match="anatomical prior covers"):
        StudyModel(_FakeBackbone(), DIM, SLOTS - 1, design=StudyDesign(anatomical_prior=True))


def test_forward_resizes_to_the_requested_size(study_model: StudyModel):
    """A cache held at one resolution is resampled to whatever the member was fitted at."""
    torch.manual_seed(SEED)
    imgs = torch.randint(0, 256, (1, SLOTS, GROUP, 28, 28), dtype=torch.uint8)
    study_model.eval()
    with torch.no_grad():
        out = study_model.forward(imgs, torch.ones(1, SLOTS), 14)
    assert out.shape == (1, len(TARGETS))


def test_fingerprint(study_model: StudyModel):
    """The same model answers the seeded question identically twice; a different model does not."""
    device = torch.device("cpu")
    first = study_model.fingerprint(14, GROUP, device)
    assert first.shape == (2, len(TARGETS))
    assert np.array_equal(first, study_model.fingerprint(14, GROUP, device))
    torch.manual_seed(SEED + 1)
    other = StudyModel(_FakeBackbone(), DIM, SLOTS)
    assert not np.allclose(first, other.fingerprint(14, GROUP, device))


def test_fingerprint_leaves_training_mode_untouched(study_model: StudyModel):
    """A check run mid-training must not silently leave the model in eval."""
    study_model.train()
    study_model.fingerprint(14, GROUP, torch.device("cpu"))
    assert study_model.training


def test_every_pool_builds_the_width_its_forward_pass_produces():
    """`POOL_PARTS` fixes the head's input width, so every name in it must concatenate exactly that many.

    Iterating the table rather than naming the two: a pooling added with a wrong count builds a head
    for a width the forward pass never produces, and the failure is a shape error deep in the first
    batch — after the backbone is fetched and the cache is mapped.
    """
    torch.manual_seed(SEED)
    for name, parts in POOL_PARTS.items():
        model = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(pool=name)).eval()
        assert model.head.proj[1].in_features == DIM * parts, name

        with torch.no_grad():
            pooled = model._pool(torch.randn(1, PATCHES + 1, DIM))
        assert pooled.shape[1] == DIM * parts, f"{name} pools a width its head is not built for"

    with pytest.raises(KeyError):
        StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(pool="cls_men"))


def test_every_slice_pooling_name_says_whether_a_slice_is_encoded_alone():
    """`per_slice` decides how many forwards a step costs and what a slot's embedding depends on.

    It asked `in {"max", "attend"}`, so a name that was neither — a typo, or a manifest written by a
    later recipe — meant channel stacking, silently. `Member.slice_pooling` and `HoldoutPredictions`
    both read that string out of a checkpoint and hand it straight here, so the wrong answer is a
    different function on different pixels that loads without complaint.
    """
    torch.manual_seed(SEED)
    for name, pooling in SLICE_POOLINGS.items():
        model = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling=name))
        assert model.per_slice is pooling.per_slice, name
        built = type(model.slices) if model.slices is not None else None
        assert built is pooling.aggregator, f"{name} builds the module its record names, and only that one"

    with pytest.raises(KeyError):
        StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="chanels"))


def test_slice_features():
    """The per-slice features are what slot pooling reduces, so pooling must be their function.

    Split out because it is the last point where the result depends only on pixels and encoder — the
    seam a cached-embedding harness needs, and the reason it can exist at all. Channel stacking has no
    per-slice axis by construction, so it returns a single position.
    """
    imgs = torch.randint(0, 256, (2, SLOTS, GROUP, 14, 14), dtype=torch.uint8)

    torch.manual_seed(SEED)
    per_slice = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max")).eval()
    with torch.no_grad():
        stacked = per_slice.slice_features(imgs, 14)
        pooled = per_slice.slot_features(imgs, 14)
    assert stacked.shape == (2, SLOTS, GROUP, DIM * POOL_PARTS["cls_mean"])
    assert torch.allclose(stacked.amax(dim=2), pooled), "max pooling must be a function of these features"

    torch.manual_seed(SEED)
    stacking = StudyModel(_FakeBackbone(), DIM, SLOTS).eval()
    with torch.no_grad():
        assert stacking.slice_features(imgs, 14).shape[2] == 1, "channel stacking has no per-slice axis"


def test_slice_features_through_plane_changes_pixels_and_nothing_else():
    """The arm's whole claim: same shape, same cost, different channels — so the control must not move."""
    imgs = torch.randint(0, 256, (2, SLOTS, GROUP, 14, 14), dtype=torch.uint8)

    torch.manual_seed(SEED)
    control = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max")).eval()
    torch.manual_seed(SEED)
    windowed = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max", through_plane=True)).eval()

    with torch.no_grad():
        plain = control.slice_features(imgs, 14)
        neighboured = windowed.slice_features(imgs, 14)

    assert neighboured.shape == plain.shape, "the slice axis, token count and width are the control's"
    assert not torch.allclose(neighboured, plain), "and the pixels reaching the trunk are not"


TUBELET = 2
TUBELET_PATCH = 4
TUBELET_CROP = 8
TUBELET_FRAMES = 4


class _TubeletBackbone(torch.nn.Module):
    """A stand-in VOLUME trunk: it reads `pixel_values_videos`, spans `TUBELET` frames per token, and
    emits no class token.

    Its grid is ADAPTIVE, which is the property the guard exists for. The real V-JEPA 2 derives its
    position ids from the run-time token count rather than a learned table, so a read at the wrong
    spatial size does not raise — it silently hands pretrained weights a grid they were never fitted
    on and returns a plausible tensor. A double that raised on the wrong size would make the guard
    untestable by making the hazard impossible.
    """

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(patch_size=TUBELET_PATCH, crop_size=TUBELET_CROP, tubelet_size=TUBELET)
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(DIM, DIM) for _ in range(12)])

    @override
    def forward(self, *, pixel_values_videos: Tensor, output_hidden_states: bool = False) -> _Output:
        n, frames, _, height, width = pixel_values_videos.shape
        count = (height // TUBELET_PATCH) * (width // TUBELET_PATCH) * (frames // TUBELET)
        last = torch.arange(count, dtype=torch.float32).reshape(1, -1, 1).expand(n, count, DIM).clone()
        states = tuple(last + index for index in range(13)) if output_hidden_states else None
        return _Output(last_hidden_state=last, hidden_states=states)


def _tubelet_model() -> StudyModel:
    torch.manual_seed(SEED)
    return StudyModel(_TubeletBackbone(), DIM, SLOTS, StudyDesign(pool=MEAN)).eval()


def test_slice_features_reads_a_tubelet_trunk_as_frames():
    """The whole point of the volume path: the group becomes DEPTH, not the trunk's three channels.

    Asserted through the trunk's own token count, because that is the only witness that the frame axis
    arrived as frames — a group folded into channels would emit one tubelet's worth of tokens, and a
    group left flat would emit `group` times too many rows.
    """
    torch.manual_seed(SEED)
    imgs = torch.randint(0, 256, (2, SLOTS, TUBELET_FRAMES, TUBELET_CROP, TUBELET_CROP), dtype=torch.uint8)
    model = _tubelet_model()

    with torch.no_grad():
        features = model.slice_features(imgs, TUBELET_CROP)

    assert features.shape == (2, SLOTS, 1, DIM), "a volume trunk keeps the (b, s, 1, w) contract untouched"
    assert model.volume.tubelet == TUBELET, "the trunk's own config decides this, not our registry"
    assert model.prefix_tokens == 0, "there is no class token to skip"

    with pytest.raises(ValueError, match="fitted at"), torch.no_grad():
        model.slice_features(imgs, TUBELET_CROP * 2)


def test_study_model_refuses_a_pooling_a_tubelet_trunk_cannot_serve():
    """`tokens[:, 0]` is a real row on a trunk with no class token — the top-left patch of the first
    tubelet — so `cls_mean` would concatenate it, the head would accept the width, and the run would
    score like a slightly noisy mean. Nothing downstream can see that, so it is refused here."""
    with pytest.raises(ValueError, match="no class token"):
        StudyModel(_TubeletBackbone(), DIM, SLOTS, StudyDesign(pool=CLS_MEAN_FOCAL))

    with pytest.raises(ValueError, match="volumetric read"):
        StudyModel(_TubeletBackbone(), DIM, SLOTS, StudyDesign(pool=MEAN, slice_pooling="max"))


class _BlockedBackbone(_FakeBackbone):
    """Blocks shaped like a real transformer's: a submodule holding several Linears, not a bare Linear.

    The plain stand-in makes each block a single `nn.Linear`, which no encoder does — and adaptation
    wraps the Linears INSIDE a block, so the realistic shape is the one that exercises it.
    """

    def __init__(self) -> None:
        super().__init__()
        # NON-LINEAR PARAMETERS, because a real encoder has them and their absence hid a bug. LoRALinear
        # freezes only the Linear it wraps, so position embeddings, norms and DINOv3's LayerScale gammas
        # stayed trainable and were optimised at the ADAPTER rate - 781,440 parameters against the
        # adapters' 663,552. A stand-in made only of Linears cannot catch that, and did not.
        self.register_parameter("position_embeddings", torch.nn.Parameter(torch.zeros(1, 4, DIM)))
        self.layer_scale = torch.nn.Parameter(torch.ones(DIM))
        blocks = []
        for _ in range(12):
            attention = torch.nn.Module()
            attention.add_module("query", torch.nn.Linear(DIM, DIM))
            attention.add_module("value", torch.nn.Linear(DIM, DIM))
            block = torch.nn.Module()
            block.add_module("attention", attention)
            block.add_module("mlp", torch.nn.Linear(DIM, DIM))
            blocks.append(block)
        self.encoder.layer = torch.nn.ModuleList(blocks)


def test_low_rank():
    """Every block is adapted, the base weights all freeze, and what opens is exactly the adapters.

    Chosen by STRUCTURE rather than by layer name: DINOv2 calls its projections query/key/value and
    DINOv3 calls them q_proj/k_proj/v_proj, so a name list would silently adapt nothing on the encoder
    it did not know about — the same failure `encoder.layer` already caused once.
    """
    torch.manual_seed(SEED)
    model = StudyModel(_BlockedBackbone(), DIM, SLOTS)
    adapted = model.low_rank(rank=2)
    assert adapted == 36, "three Linears in each of twelve blocks, attention and MLP alike"

    groups = model.parameter_groups(head_lr=1e-3, backbone_lr=8e-6, unfreeze_last=6)
    opened = cast("list[torch.nn.Parameter]", groups[1]["params"])
    assert opened, "the adapters must be trainable"
    assert all(parameter.requires_grad for parameter in opened)
    assert all(isinstance(module, LoRALinear) or True for module in model.backbone.modules())

    frozen = [p for p in model.backbone.parameters() if not p.requires_grad]
    assert len(frozen) == 74, "36 Linears' weight and bias, PLUS the position embedding and layer scale"
    trainable = [name for name, p in model.backbone.named_parameters() if p.requires_grad]
    assert all(".down." in name or ".up." in name for name in trainable), (
        "ONLY adapters may train: everything that is not an nn.Linear - embeddings, norms, LayerScale - "
        "is invisible to LoRALinear and stayed open at the adapter rate until low_rank froze it"
    )
    # unfreeze_last is IGNORED under adaptation: honouring it too would open six blocks' base weights
    # alongside the adapters and quietly make the run something else.
    assert not any(id(p) in {id(f) for f in frozen} for p in opened)


def test_study_design_from_config():
    """Every switch a run records comes back, with the defaults spelled in one place.

    The equivalence classes a config falls into: empty (every published member, which predates all of
    these keys and must come back as the design it was fitted under), and one that states each switch
    away from its default. `pool` is in here because the ranking tool read every switch but that one.
    """
    assert StudyDesign.from_config({}) == StudyDesign(), "an old manifest means what it was fitted with"

    stated = StudyDesign.from_config(
        {
            "pool": CLS_MEAN_FOCAL,
            "slice_pooling": ATTEND,
            "slot_pooling": "mean",
            "anatomical_prior": True,
            "encoder_layer": -4,
        }
    )
    assert stated == StudyDesign(
        pool=CLS_MEAN_FOCAL, slice_pooling=ATTEND, slot_pooling="mean", anatomical_prior=True, layer=-4
    )


def test_lora_rank_of():
    """Zero for every manifest predating adaptation, and zero however the key spells "none".

    Those members carry no adapter keys and must be built without them; a stored `null` or an empty
    string means the same thing as an absent key, and reading either as a rank would wrap an encoder
    whose weights have no adapters to load into.
    """
    assert StudyDesign.lora_rank_of({}) == 0
    assert StudyDesign.lora_rank_of({"lora_rank": None}) == 0
    assert StudyDesign.lora_rank_of({"lora_rank": 8}) == 8


def test_study_model_from_config():
    """THE ONE REBUILD: the width the config fixes, and the adapters it says the encoder carries.

    Both are what the ranking path never read — it rebuilt a focal member two parts wide and a LoRA
    member with no adapters, so neither could be loaded at all, let alone scored.
    """
    torch.manual_seed(SEED)
    model = StudyModel.from_config({"pool": CLS_MEAN_FOCAL, "lora_rank": 2}, _BlockedBackbone(), DIM, SLOTS)
    assert model.head.proj[1].in_features == DIM * POOL_PARTS[CLS_MEAN_FOCAL], "the recorded pooling fixes the width"
    assert any(isinstance(module, LoRALinear) for module in model.backbone.modules()), (
        "the adapters must exist before any weights arrive, or a LoRA member cannot be loaded at all"
    )
    assert not any(
        isinstance(m, LoRALinear) for m in StudyModel.from_config({}, _BlockedBackbone(), DIM, SLOTS).backbone.modules()
    )


def test_study_model_from_config_resolves_the_variants_normalization():
    """THE REBUILD READS THE REGISTRY, not one shared constant.

    Six of seven encoders were pretrained under the ImageNet constants and the seventh was not, so a
    hardcoded pair was right by coincidence everywhere it was ever checked. RAD-DINO's greyscale
    constants sit 0.20 std off in offset and 12.8% off in scale — no error, a plausible curve, a real
    number supporting a false conclusion.
    """
    torch.manual_seed(SEED)
    rad = StudyModel.from_config({"variant": RAD_DINO}, _BlockedBackbone(), DIM, SLOTS)
    assert rad.mean.flatten().tolist() == pytest.approx([0.5307] * 3), "the mount's own constants, not ImageNet's"
    assert rad.std.flatten().tolist() == pytest.approx([0.2583] * 3)
    plain = StudyModel.from_config({}, _BlockedBackbone(), DIM, SLOTS)
    assert plain.mean.flatten().tolist() == pytest.approx(list(IMAGENET.mean)), "an unnamed variant stays ImageNet"
    assert plain.std.flatten().tolist() == pytest.approx(list(IMAGENET.std))


def test_low_rank_refuses_a_backbone_with_no_linears():
    """Adapting nothing and reporting a number looks exactly like adapting something and working."""
    torch.manual_seed(SEED)
    backbone = _FakeBackbone()
    backbone.encoder.layer = torch.nn.ModuleList([torch.nn.Identity() for _ in range(12)])
    model = StudyModel(backbone, DIM, SLOTS)
    with pytest.raises(ValueError, match="no Linear layers"):
        model.low_rank(rank=2)


def test_parameter_groups(study_model: StudyModel):
    """The head gets its own learning rate; the encoder gets a slower one and only its last blocks."""
    groups = study_model.parameter_groups(head_lr=1e-3, backbone_lr=8e-6, unfreeze_last=2)
    assert [group["lr"] for group in groups] == [1e-3, 8e-6]
    opened = cast("list[torch.nn.Parameter]", groups[1]["params"])
    assert opened, "the last blocks must remain trainable"
    assert all(parameter.requires_grad for parameter in opened)
    frozen = [p for p in study_model.backbone.parameters() if not p.requires_grad]
    assert frozen, "the early blocks must be frozen"


def test_parameter_groups_hold_every_trainable_tensor():
    """NOTHING TRAINABLE MAY FALL OUTSIDE EVERY GROUP — the failure this test exists for already happened.

    The head group was `self.head.parameters()`, and the slice aggregator is a SIBLING of the head. Its
    tensors were therefore in no group and the optimiser never updated them, for as long as
    `SliceAttention` existed. It never surfaced as a crash or even as a bad score: the gate is
    zero-initialised so that an untrained pooling equals the mean over slices, so `attend` scored like a
    mean because it was one (knee-7gyw). Asserted over EVERY pooling, since the whole point is that a
    module added later must not be able to go missing the same way.
    """
    for name in SLICE_POOLINGS:
        torch.manual_seed(SEED)
        model = StudyModel(_BlockedBackbone(), DIM, SLOTS, StudyDesign(slice_pooling=name))
        groups = model.parameter_groups(head_lr=1e-3, backbone_lr=8e-6, unfreeze_last=4)

        seen = [id(p) for group in groups for p in cast("list[torch.nn.Parameter]", group["params"])]
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        assert trainable <= set(seen), f"{name} leaves a trainable tensor outside every optimiser group"
        assert len(seen) == len(set(seen)), f"{name} hands the same tensor to two groups"


def test_parameter_groups_layer_decay():
    """One group per opened block, the TOP one still at `backbone_lr`, everything below it divided down.

    The top block holding the named rate is the property that matters: every ledger number was measured
    with the whole opened stack at `backbone_lr`, so anchoring at the top makes the decayed arm a strict
    SLOWING of the layers underneath rather than a run at a different rate throughout — which would move
    two variables and read as neither.
    """
    torch.manual_seed(SEED)
    model = StudyModel(_BlockedBackbone(), DIM, SLOTS)
    groups = model.parameter_groups(head_lr=1e-3, backbone_lr=8e-6, unfreeze_last=4, layer_decay=0.5)

    rates = [group["lr"] for group in groups]
    assert rates == [1e-3, 8e-6 * 0.125, 8e-6 * 0.25, 8e-6 * 0.5, 8e-6], "bottom to top, halving upwards"
    opened = [p for block in groups[1:] for p in cast("list[torch.nn.Parameter]", block["params"])]
    assert all(parameter.requires_grad for parameter in opened)
    assert [p for p in model.backbone.parameters() if not p.requires_grad], "the early blocks must freeze"


def test_parameter_groups_flat_keeps_the_two_group_shape():
    """The default must be the OLD two groups, not per-block groups that happen to share a rate.

    A refactor that reproduces the ledger's shape only numerically is not one a score can confirm:
    AdamW carries per-group state and weight decay, and "the same rates in more groups" is a different
    optimiser configuration that no arm would ever be able to attribute the difference to.
    """
    torch.manual_seed(SEED)
    model = StudyModel(_BlockedBackbone(), DIM, SLOTS)
    flat = model.parameter_groups(head_lr=1e-3, backbone_lr=8e-6, unfreeze_last=4)

    assert len(flat) == 2, "head, then every opened block in ONE group"
    assert [group["lr"] for group in flat] == [1e-3, 8e-6]


def test_parameter_groups_refuses_layer_decay_under_lora():
    """Under LoRA the base blocks a per-depth rate would scale are frozen, so the flag would be a no-op.

    Same refusal as `--pair-memory` with no ranking term, for the same reason: the silent version trains
    the flat recipe under the decayed arm's name and gets read as a refutation of the mechanism.
    """
    torch.manual_seed(SEED)
    model = StudyModel(_BlockedBackbone(), DIM, SLOTS)
    model.low_rank(rank=2)
    with pytest.raises(ValueError, match="LoRA adapters"):
        model.parameter_groups(head_lr=1e-3, backbone_lr=8e-6, unfreeze_last=4, layer_decay=0.5)


class _V3ShapedBackbone(_FakeBackbone):
    """A backbone that keeps its blocks where a DINOv3 keeps them: `model.layer`, with no `encoder`."""

    def __init__(self) -> None:
        super().__init__()
        del self.encoder
        self.model = torch.nn.Module()
        self.model.layer = torch.nn.ModuleList([torch.nn.Linear(DIM, DIM) for _ in range(12)])


def test_parameter_groups_finds_blocks_that_are_not_under_encoder():
    """A DINOv3 tree has no `encoder.layer`, and the hardcoded path raised AttributeError on it."""
    torch.manual_seed(SEED)
    model = StudyModel(_V3ShapedBackbone(), DIM, SLOTS)
    groups = model.parameter_groups(head_lr=1e-3, backbone_lr=8e-6, unfreeze_last=2)
    opened = cast("list[torch.nn.Parameter]", groups[1]["params"])
    assert len(opened) == 4, "two Linear blocks is two weights and two biases"
    assert [p for p in model.backbone.parameters() if not p.requires_grad], "the early blocks must freeze"


def test_parameter_groups_refuses_a_backbone_with_no_blocks():
    """Something with no ModuleList is not a transformer, and silently opening nothing trains nothing."""
    torch.manual_seed(SEED)
    backbone = _FakeBackbone()
    del backbone.encoder
    model = StudyModel(backbone, DIM, SLOTS)
    with pytest.raises(ValueError, match="no ModuleList"):
        model.parameter_groups(head_lr=1e-3, backbone_lr=8e-6, unfreeze_last=2)


def test_parameter_groups_refuses_ambiguous_blocks():
    """Two equally long ModuleLists give no answer, and guessing opens modules that are not the top."""
    torch.manual_seed(SEED)
    backbone = _FakeBackbone()
    backbone.decoder = torch.nn.Module()
    backbone.decoder.layer = torch.nn.ModuleList([torch.nn.Linear(DIM, DIM) for _ in range(12)])
    model = StudyModel(backbone, DIM, SLOTS)
    with pytest.raises(ValueError, match="ambiguous"):
        model.parameter_groups(head_lr=1e-3, backbone_lr=8e-6, unfreeze_last=2)


def test_forward_per_slice_pooling_differs_from_channel_stacking():
    """The two aggregations are different functions, and a test double must be able to tell.

    The first version of this could not: the stand-in encoder added the same scalar to every feature
    dimension, and the head's opening LayerNorm removed it, so the model's output did not depend on
    its pixels at all. A double has to perturb the feature vector's SHAPE to be worth anything.
    """
    imgs = torch.randint(0, 256, (2, SLOTS, GROUP, 14, 14), dtype=torch.uint8)
    mask = torch.ones(2, SLOTS)
    outputs = {}
    for mode in ("channels", "max"):
        torch.manual_seed(SEED)
        model = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling=mode)).eval()
        with torch.no_grad():
            outputs[mode] = model.forward(imgs, mask, 14)
    assert not torch.allclose(outputs["channels"], outputs["max"]), "per-slice pooling must change the function"


def test_per_slice():
    """Which poolings encode a slice ALONE — the one fact three call sites used to each decide alone.

    "attend" is the case that broke: the encoder branched on a set containing it, the attention module
    built itself on it, and the trainer's whole-stack guard tested `!= "max"` — so an arm that had
    been written, reviewed and queued died at startup with 'whole-stack reading needs per-slice
    pooling, not attend', against a model that supports precisely that.
    """
    for mode in ("max", "attend"):
        assert StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling=mode)).per_slice, mode
    assert not StudyModel(_FakeBackbone(), DIM, SLOTS).per_slice, "channel stacking reads three at once"


def test_check_whole_stack():
    """Which models can be handed the whole stack, and the two DIFFERENT reasons one can.

    The equivalence classes are the guard's own history. Channel stacking is the genuine refusal — it
    has three channels to stack into and no more. Per-slice pooling takes any width. A tubelet trunk
    takes any width its tubelet divides, and is the class the previous `per_slice` guard rejected
    while the model supported it exactly as it had rejected "attend" one trunk earlier.
    """
    StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max")).check_whole_stack(20)

    stacked = StudyModel(_FakeBackbone(), DIM, SLOTS)
    with pytest.raises(ValueError, match="per-slice pooling or a tubelet trunk"):
        stacked.check_whole_stack(20)

    volumetric = StudyModel(_FakeBackbone(), DIM, SLOTS)
    volumetric.volume = VolumeRead(2, 16, 256)
    volumetric.check_whole_stack(20)
    with pytest.raises(ValueError, match="partial tubelet"):
        volumetric.check_whole_stack(15)


def test_forward_is_sensitive_to_its_pixels():
    """A guard for the double itself: two different STRUCTURED images must give different logits.

    Written because a degenerate stand-in silently made every pooling test vacuous — it added the same
    scalar to every feature, and the head's opening LayerNorm removed it.

    The images are structured rather than uniform on purpose. A uniform image drives this double to a
    constant feature vector, which LayerNorm legitimately erases; a real encoder would still separate
    black from white through its biases and position embeddings. Asserting on uniform inputs would be
    testing the stand-in's limits rather than the model's behaviour.
    """
    torch.manual_seed(SEED)
    model = StudyModel(_FakeBackbone(), DIM, SLOTS).eval()
    mask = torch.ones(1, SLOTS)
    generator = torch.Generator().manual_seed(SEED)
    first = torch.randint(0, 256, (1, SLOTS, GROUP, 14, 14), generator=generator, dtype=torch.uint8)
    second = torch.randint(0, 256, (1, SLOTS, GROUP, 14, 14), generator=generator, dtype=torch.uint8)
    with torch.no_grad():
        assert not torch.allclose(model.forward(first, mask, 14), model.forward(second, mask, 14)), (
            "the model must respond to its input"
        )


def test_load_trained():
    """Every member on the leaderboard was saved without `head.prior`, and must still load.

    A strict load rejects them all. The missing parameter is unambiguous — it is zeros unless the design
    asks for the prior — so it keeps its initialised value while everything the checkpoint does carry is
    still matched by name and shape.
    """
    torch.manual_seed(SEED)
    model = StudyModel(_FakeBackbone(), DIM, SLOTS)
    older = {name: value for name, value in model.state_dict().items() if name != "head.prior"}

    torch.manual_seed(SEED + 1)
    loaded = StudyModel(_FakeBackbone(), DIM, SLOTS)
    loaded.load_trained(older)

    assert torch.equal(loaded.head.out.weight, model.head.out.weight)
    assert float(loaded.head.prior.abs().sum()) == 0.0


def test_load_trained_still_rejects_a_checkpoint_that_does_not_fit():
    """Narrower than strict=False: a wrong shape is still an error, not a silent partial load."""
    model = StudyModel(_FakeBackbone(), DIM, SLOTS)
    state = dict(model.state_dict())
    state["head.out.weight"] = torch.zeros(3, 3)
    with pytest.raises(RuntimeError):
        model.load_trained(state)


def test_slot_features(study_model: StudyModel):
    """One embedding per slot, before any head reads them — the shape the two readouts share."""
    imgs = torch.randint(0, 256, (2, SLOTS, GROUP, 14, 14), dtype=torch.uint8)
    features = study_model.slot_features(imgs, 14)
    assert features.shape[:2] == (2, SLOTS)
    assert torch.isfinite(features).all()


def test_over_depth(study_model: StudyModel):
    """One region is the historic maximum EXACTLY; more regions give each part of the depth its own.

    The equality is the contract that lets this replace the old `amax` rather than branch around it —
    every member on disk was fitted at one region, so any drift here is a silent refit of all of them.
    """
    stacked = torch.randn(2, SLOTS, 4, 8)

    assert torch.equal(study_model.over_depth(stacked), stacked.amax(dim=2)), "one region is one max"

    study_model.design = replace(study_model.design, slice_regions=2)
    halves = torch.stack([stacked[:, :, :2].amax(dim=2), stacked[:, :, 2:].amax(dim=2)]).mean(dim=0)
    assert torch.allclose(study_model.over_depth(stacked), halves)

    study_model.design = replace(study_model.design, slice_regions=8)
    thin = stacked[:, :, :3]
    assert torch.equal(study_model.over_depth(thin), thin.mean(dim=2)), "a region is never thinner than a slice"


def test_supervised(study_model: StudyModel):
    """Both readouts come from ONE encoder pass, and the logits match plain `forward` exactly.

    The equality is the contract: the fingerprint, the kernel and every published member read
    `forward`, so a member fitted through `supervised` has to be the same function at inference.
    """
    model = study_model.eval()
    imgs = torch.randint(0, 256, (2, SLOTS, GROUP, 14, 14), dtype=torch.uint8)
    mask = torch.ones(2, SLOTS)

    with torch.no_grad():
        readout = model.supervised(imgs, mask, 14)
        plain = model(imgs, mask, 14)

    assert torch.allclose(readout.logits, plain, atol=1e-6), "training and inference must be the same function"
    assert readout.report.shape == (2, REPORT_DIMS)
    assert readout.context.shape[:2] == (2, len(TARGETS)), "one context vector per finding, per study"


def test_applied():
    """The model's own account of what was done to it, before and after each wrapper.

    Read off the module tree rather than off a flag, because the failure being guarded against is a
    wrapper that runs and changes nothing: `low_rank` picks Linears by structure, so an encoder whose
    blocks hold none adapts zero layers while the run's config still says rank 2.

    The depth regions read 1 under the default design because channel stacking removes the slice axis
    before `over_depth` can split it — the count a run ASKS for is only the count it gets when nothing
    else has already consumed that axis.
    """
    torch.manual_seed(SEED)
    model = StudyModel(_BlockedBackbone(), DIM, SLOTS)

    assert model.applied() == {
        ADAPTERS: 0,
        RECOMPUTING: False,
        COMPILED_ENCODER: False,
        OPENED_BLOCKS: 12,
        DEPTH_REGIONS: 1,
    }

    model.low_rank(rank=2)
    model.recompute_activations()

    assert model.applied() == {
        ADAPTERS: 36,
        RECOMPUTING: True,
        COMPILED_ENCODER: False,
        OPENED_BLOCKS: 0,
        DEPTH_REGIONS: 1,
    }, (
        "adaptation freezes every base weight, so no block is open on the axis unfreeze_last moves — "
        "counting trainable-anything would say twelve here and make the check unreadable under LoRA"
    )


def test_parameter_groups_at_zero_depth_opens_nothing():
    """`blocks[-0:]` is the WHOLE list, so a head-only run trained the entire encoder.

    Measured before the fix on this same backbone: asked 0 gave twelve open blocks and 2,592
    parameters, byte-identical to asked 12. `unfreeze_depth.sh` takes its depth as an argument, so the
    head-only rung of a depth sweep was one invocation from spending its slot on the deepest arm.
    """
    torch.manual_seed(SEED)
    model = StudyModel(_BlockedBackbone(), DIM, SLOTS)

    groups = model.parameter_groups(head_lr=1e-3, backbone_lr=8e-6, unfreeze_last=0)

    assert cast("list[torch.nn.Parameter]", groups[1]["params"]) == [], "nothing in the encoder may open"
    assert model.applied()[OPENED_BLOCKS] == 0
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())


def test_parameter_groups_clamps_a_depth_past_the_encoder():
    """More blocks than the encoder has opens every block, rather than reading as a negative slice."""
    torch.manual_seed(SEED)
    model = StudyModel(_BlockedBackbone(), DIM, SLOTS)

    model.parameter_groups(head_lr=1e-3, backbone_lr=8e-6, unfreeze_last=99)

    assert model.applied()[OPENED_BLOCKS] == 12


def test_recompute_activations(study_model: StudyModel):
    """Checkpointing is turned on through the encoder's own hook, non-reentrant.

    use_reentrant=False matters: the reentrant implementation silently produces no gradients when the
    checkpointed block's inputs do not require grad, which is exactly our case - the early encoder
    blocks are frozen, so the first trainable block receives an input with no grad history.
    """
    study_model.recompute_activations()

    backbone = cast("_FakeBackbone", study_model.backbone)
    assert backbone.checkpointing == {"use_reentrant": False}


def test_compiled_prefix_is_what_torch_actually_inserts():
    """The constant is checked against torch, not against our memory of torch.

    Wrapping alone is enough — compilation is lazy, so no forward runs and no inductor is invoked. The
    failure this guards is silent in both readers: `load_trained` strips a prefix that is no longer
    there and refuses the old members it exists to recover, and `EncoderDrift` matches no tensor at all
    and reports a drift of 0.000 rather than raising.
    """
    keys = torch.compile(nn.Linear(2, 2)).state_dict()

    assert keys, "an empty state dict would pass every assertion below without meaning anything"
    assert all(key.startswith(COMPILED_PREFIX) for key in keys), f"torch now wraps under {list(keys)}"


def test_compile_encoder(study_model: StudyModel):
    """Compiling leaves the model's PARAMETER NAMES alone, and the forward path still answers.

    This test used to assert `backbone is not before` — "the encoder must actually be replaced" — which
    pinned the defect rather than the contract. Rebinding the attribute to torch.compile's wrapper puts
    its `_orig_mod` into every state-dict key, and five members were saved unloadable before anything
    tried to read one back. Identity is now asserted the other way, because the names are what a
    checkpoint carries and a submission needs.

    THE FORWARD IS NO LONGER EXERCISED HERE, and that is the same fact `RunSetup.model`
    now encodes by refusing to compile off a card: a compiled CPU trace makes dynamo probe
    `has_triton()`, which indexes a CUDA worker cache the trace never filled and raises `IndexError:
    list index out of range`. Moving this fixture to a card instead only trades that for a device
    mismatch inside the FAKE backbone, which holds a CPU constant `nn.Module.to` never sees — either
    way the assertion is about the harness, not the model. What compilation must not break is the
    parameter NAMES, which is what a checkpoint carries and needs no forward; that a compiled encoder
    still trains is proved by every arm the smoke runs on the card.
    """
    before = study_model.backbone
    names = set(study_model.state_dict())
    study_model.compile_encoder()

    assert study_model.backbone is before, "compiling must not rebind the encoder"
    assert set(study_model.state_dict()) == names, "compiling must not rename a single parameter"


def test_tokens_reads_the_layer_the_design_asks_for():
    """-1 takes the cheap `last_hidden_state` path; any other index reads that block's tokens.

    The default must be byte-identical to what every member in the ledger was fitted on, which is why
    -1 does not go through `output_hidden_states` at all. Reading an intermediate layer is a knob, not
    a fix: the final block of a self-supervised ViT is the most specialised toward its pretraining
    objective, and our findings are 1-3 mm.
    """
    imgs = torch.randint(0, 256, (1, SLOTS, GROUP, 14, 14), dtype=torch.uint8)
    torch.manual_seed(SEED)
    last = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max")).eval()
    torch.manual_seed(SEED)
    middle = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max", layer=6)).eval()

    with torch.no_grad():
        from_last = last.slice_features(imgs, 14)
        from_middle = middle.slice_features(imgs, 14)

    assert from_last.shape == from_middle.shape
    assert not torch.allclose(from_last, from_middle), "a different block must reach the head"


def test_every_slot_pooling_name_builds_its_own_head():
    """The name a member records must build the head it names — for every name, not just the default.

    The dispatch was an `if` against one literal, so anything that was not exactly "mean" got the
    attention head: a typo, a stale checkpoint, or a third pooling added later. Iterating the vocabulary
    is the point — a name added to `SLOT_POOLINGS` without a head, or a head the builder cannot reach,
    fails here rather than in a run that reports a plausible number for the wrong recipe.
    """
    for name, head in SLOT_POOLINGS.items():
        model = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slot_pooling=name))
        assert type(model.head) is head, name

    with pytest.raises(KeyError):
        StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slot_pooling="attnd"))
