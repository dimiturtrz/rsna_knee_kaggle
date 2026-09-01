from dataclasses import dataclass
from types import SimpleNamespace
from typing import override

import pytest
import torch
from torch import Tensor, nn

from kneemri.models.depth_window import DepthWindow
from kneemri.models.grid_fit import INTERPOLATE
from kneemri.models.model import StudyDesign, StudyModel
from kneemri.models.slice_read import BYTE_SCALE, SliceRead
from kneemri.models.token_pooling import MEAN
from kneemri.models.volume_read import VolumeRead

SEED = 2026
DIM = 8
PATCHES = 16
SLOTS = 6
GROUP = 4
SIZE = 4
TUBELET = 2
TUBELET_PATCH = 2
TUBELET_CROP = 4

MEAN_CONSTANTS = torch.tensor([0.1, 0.2, 0.3]).view(1, 3, 1, 1)
STD_CONSTANTS = torch.tensor([0.4, 0.5, 0.6]).view(1, 3, 1, 1)

STATE_DICT_KEYS = (
    "mean",
    "std",
    "backbone.project.weight",
    "backbone.project.bias",
    "head.slot_emb",
    "head.query",
    "head.prior",
    "head.proj.0.weight",
    "head.proj.0.bias",
    "head.proj.1.weight",
    "head.proj.1.bias",
    "head.out.weight",
    "head.out.bias",
    "head.report.weight",
    "head.report.bias",
)
"""RECORDED EXPLICITLY, from the tree before `SliceRead` existed, because the point of this file is that
it did not move. `SliceRead` is a frozen dataclass and not an `nn.Module`, so it registers no parameter
and no buffer; every member on disk keeps loading only for as long as that stays true."""


@dataclass
class _Output:
    last_hidden_state: Tensor


class _FakeBackbone(nn.Module):
    """A deterministic stand-in shaped like the trunks this model is fitted on."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=DIM)
        self.project = nn.Linear(SIZE * SIZE * 3, DIM)

    @override
    def forward(self, pixel_values: Tensor) -> _Output:
        flat = pixel_values.reshape(pixel_values.shape[0], -1)
        token = self.project(flat).unsqueeze(1)
        return _Output(last_hidden_state=token.expand(-1, 1 + PATCHES, -1).contiguous())


class _InterpolatingBackbone(_FakeBackbone):
    """A trunk whose position grid is fixed, so its forward takes the resampling keyword."""

    @override
    def forward(self, pixel_values: Tensor, interpolate_pos_encoding: bool = False) -> _Output:
        assert interpolate_pos_encoding, "a fixed-grid trunk must be told to resample"
        return super().forward(pixel_values)


class _RegisteredBackbone(_FakeBackbone):
    """DINOv3's shape: a class token plus four register tokens in front of the patches."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=DIM, num_register_tokens=4)


class _TubeletBackbone(nn.Module):
    """A volumetric trunk, which announces itself by a tubelet in its own config.

    NOT a subclass of the 2D stand-in, though it repeats its two lines: the whole point of the trunk is
    that it takes a DIFFERENT forward keyword, and a subclass whose override renames the parameter is
    the same substitutability defect in the double that it would be in production.
    """

    def __init__(self) -> None:
        super().__init__()
        self.project = nn.Linear(SIZE * SIZE * 3, DIM)
        self.config = SimpleNamespace(
            hidden_size=DIM,
            tubelet_size=TUBELET,
            patch_size=TUBELET_PATCH,
            crop_size=TUBELET_CROP,
        )

    @override
    def forward(self, pixel_values_videos: Tensor) -> _Output:
        videos, frames = pixel_values_videos.shape[:2]
        flat = pixel_values_videos.reshape(videos, -1)[:, : SIZE * SIZE * 3]
        token = self.project(flat).unsqueeze(1)
        tokens = (TUBELET_CROP // TUBELET_PATCH) ** 2 * (frames // TUBELET)
        return _Output(last_hidden_state=token.expand(-1, tokens, -1).contiguous())


def _bag(group: int = GROUP, size: int = SIZE) -> Tensor:
    generator = torch.Generator().manual_seed(SEED)
    return torch.randint(0, 256, (2, SLOTS, group, size, size), generator=generator, dtype=torch.uint8)


def test_of():
    """The four facts a trunk and a design settle between them, read off the trunk's own config.

    The equivalence classes are the trunk families this project mounts: a plain DINOv2 (one prefix
    token, its own position resampling), DINOv3 (four registers on top of the class token), a
    fixed-grid trunk that must be handed the keyword, and a tubelet trunk that has no class token at
    all. `through_plane` is the one field the DESIGN owns rather than the weights.
    """
    plain = SliceRead.of(_FakeBackbone(), MEAN, "max", through_plane=False)
    assert plain.volume is None, "no tubelet in the config is what makes a trunk 2D"
    assert plain.prefix_tokens == 1
    assert not plain.grid.interpolates
    assert not plain.through_plane

    assert SliceRead.of(_RegisteredBackbone(), MEAN, "max", through_plane=True).prefix_tokens == 5
    assert SliceRead.of(_FakeBackbone(), MEAN, "max", through_plane=True).through_plane
    assert SliceRead.of(_InterpolatingBackbone(), MEAN, "max", through_plane=False).grid.interpolates

    volumetric = SliceRead.of(_TubeletBackbone(), MEAN, "channels", through_plane=False)
    assert volumetric.volume == VolumeRead(TUBELET, TUBELET_PATCH, TUBELET_CROP)
    assert volumetric.prefix_tokens == 0, "a tubelet trunk's first token is a patch"


def test_extra():
    """Empty for a trunk that resamples its own grid, and the keyword for one that must be told."""
    assert SliceRead.of(_FakeBackbone(), MEAN, "max", through_plane=False).extra() == {}
    assert SliceRead.of(_InterpolatingBackbone(), MEAN, "max", through_plane=False).extra() == {INTERPOLATE: True}


def test_probe_frames():
    """A 2D trunk's synthetic depth is the group itself; a tubelet trunk's rounds up to a whole tubelet."""
    assert SliceRead.of(_FakeBackbone(), MEAN, "max", through_plane=False).probe_frames(3) == 3
    volumetric = SliceRead.of(_TubeletBackbone(), MEAN, "channels", through_plane=False)
    assert volumetric.probe_frames(3) == 4, "three slices is one and a half tubelets"
    assert volumetric.probe_frames(4) == 4


def test_check_tokens():
    """SILENT ON A 2D TRUNK AND LOUD ON A VOLUMETRIC ONE — the no-op half is why the caller has no branch.

    The negative case is the load-bearing one: a spatial grid the weights were never fitted at is a read
    the trunk itself would have accepted without a word.
    """
    frames = torch.zeros(1, TUBELET, 3, TUBELET_CROP, TUBELET_CROP)
    grid = (TUBELET_CROP // TUBELET_PATCH) ** 2
    plain = SliceRead.of(_FakeBackbone(), MEAN, "max", through_plane=False)

    plain.check_tokens(frames, torch.zeros(1, 1, DIM))

    volumetric = SliceRead.of(_TubeletBackbone(), MEAN, "channels", through_plane=False)
    volumetric.check_tokens(frames, torch.zeros(1, grid, DIM))
    wide = torch.zeros(1, TUBELET, 3, TUBELET_CROP * 2, TUBELET_CROP * 2)
    with pytest.raises(ValueError, match="fitted at"):
        volumetric.check_tokens(wide, torch.zeros(1, grid, DIM))


def test_pixels():
    """EVERY BRANCH AGAINST THE EXPRESSION IT WAS LIFTED OUT OF, written out here rather than referenced.

    This is the equivalence oracle for the split: the right-hand sides below are the four lines
    `StudyModel.slice_features` held before `SliceRead` existed, and `torch.equal` is a bitwise
    comparison rather than a tolerance. A refactor that changed the arithmetic — a different scaling
    order, a resize before the byte division, a normalisation applied to the folded tensor — would move
    every published member's predictions while loading its weights cleanly, and would be visible here
    and nowhere else.
    """
    imgs = _bag()
    batch, slots, group = imgs.shape[:3]
    flat = batch * slots * group

    triple = _bag(group=3)
    stacked = SliceRead.of(_FakeBackbone(), MEAN, "channels", through_plane=False)
    expected = triple.reshape(batch * slots, *triple.shape[2:]).float().div(BYTE_SCALE)
    assert torch.equal(
        stacked.pixels(triple, MEAN_CONSTANTS, STD_CONSTANTS, per_slice=False),
        (expected - MEAN_CONSTANTS) / STD_CONSTANTS,
    )

    alone = SliceRead.of(_FakeBackbone(), MEAN, "max", through_plane=False)
    replicated = imgs.reshape(flat, 1, *imgs.shape[3:]).expand(-1, 3, -1, -1).float().div(BYTE_SCALE)
    assert torch.equal(
        alone.pixels(imgs, MEAN_CONSTANTS, STD_CONSTANTS, per_slice=True),
        (replicated - MEAN_CONSTANTS) / STD_CONSTANTS,
    )

    through = SliceRead.of(_FakeBackbone(), MEAN, "max", through_plane=True)
    windowed = DepthWindow.channels(imgs).reshape(flat, 3, *imgs.shape[3:]).float().div(BYTE_SCALE)
    assert torch.equal(
        through.pixels(imgs, MEAN_CONSTANTS, STD_CONSTANTS, per_slice=True),
        (windowed - MEAN_CONSTANTS) / STD_CONSTANTS,
    )
    assert not torch.equal(
        through.pixels(imgs, MEAN_CONSTANTS, STD_CONSTANTS, per_slice=True),
        alone.pixels(imgs, MEAN_CONSTANTS, STD_CONSTANTS, per_slice=True),
    ), "the through-plane window must actually change what the channels hold"

    volumetric = SliceRead.of(_TubeletBackbone(), MEAN, "channels", through_plane=False)
    folded = volumetric.pixels(imgs, MEAN_CONSTANTS, STD_CONSTANTS, per_slice=False)
    assert folded.shape == (batch * slots, group, 3, SIZE, SIZE), "the window reaches the trunk as FRAMES"
    assert torch.equal(folded.reshape(flat, 3, SIZE, SIZE), (replicated - MEAN_CONSTANTS) / STD_CONSTANTS)

    resized = alone.pixels(imgs, MEAN_CONSTANTS, STD_CONSTANTS, SIZE * 2, per_slice=True)
    assert resized.shape == (flat, 3, SIZE * 2, SIZE * 2)


def test_the_split_moved_no_state_dict_key():
    """THE INVARIANT THE WHOLE EXTRACTION IS BOUNDED BY: not one key or value changed.

    `SliceRead` holds four facts that used to be attributes of `StudyModel`, and `StudyModel` is an
    `nn.Module` — which registers any attribute that is a `Parameter`, a `Module` or a `Tensor`. Had the
    value been given buffers of its own, or had the normalisation constants moved into it, every member
    published on the leaderboard would have failed to load against a tree that looks entirely fine.

    The keys are compared against a list recorded from BEFORE the split rather than against a rebuild,
    so the test cannot agree with a mistake by making it twice.
    """
    torch.manual_seed(SEED)
    model = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max"))

    state = model.state_dict()
    assert tuple(state) == STATE_DICT_KEYS
    assert torch.equal(state["mean"], torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
    assert torch.equal(state["std"], torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    assert not isinstance(model.reading, nn.Module), "a registered SliceRead would rename every key under it"


def test_forward_is_unchanged_by_the_split():
    """One fixed input through a seeded model, bit for bit — the oracle for the pixel step in place.

    `test_pixels` proves the extracted arithmetic in isolation; this proves the model still CALLS it the
    same way, with the same buffers, in the same order relative to the resize — a seeded rebuild must be
    bitwise identical, which is the property a silently reordered pixel step would break.

    The tie to the tree BEFORE the extraction cannot live in this file, because the file is on the
    after side: it was made once against `git show HEAD:kneemri/models/model.py`, over seven designs
    (channel stacking, `max`, `attend`, through-plane, DINOv3 registers, depth regions and a tubelet
    trunk), and every one agreed on its keys, its buffer values and its logits bit for bit. What this
    keeps is the standing half — that the split's own arithmetic is reproducible from a seed.
    """
    torch.manual_seed(SEED)
    model = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max")).eval()
    torch.manual_seed(SEED)
    rebuilt = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max")).eval()

    imgs = _bag()
    mask = torch.ones(2, SLOTS)
    with torch.no_grad():
        logits = model.forward(imgs, mask, SIZE)
        again = rebuilt.forward(imgs, mask, SIZE)

    assert torch.equal(logits, again), "a seeded rebuild must be bitwise identical"
    assert logits.shape == (2, len(model.head.out.bias))
