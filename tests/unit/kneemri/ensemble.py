import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import override

import numpy as np
import pytest
import torch
from torch import Tensor

from kneemri.ensemble import Ensemble
from kneemri.members import MemberConfig, MemberPackage
from kneemri.model import TARGETS, StudyDesign, StudyModel

SEED = 2026
DIM = 8
PATCHES = 16
SLOTS = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]


@dataclass
class _Output:
    last_hidden_state: Tensor


class _FakeBackbone(torch.nn.Module):
    """A stand-in encoder whose tokens vary PER DIMENSION with the input.

    An earlier version returned `base + mean(pixels)` — the same scalar added to every feature. The
    head begins with a LayerNorm, which removes exactly that, so the stand-in made the model's output
    independent of its pixels and no test could detect a change in pooling. A useful double has to
    perturb the feature vector's SHAPE, not just its level.
    """

    @override
    def forward(self, *, pixel_values: Tensor) -> _Output:
        n = pixel_values.shape[0]
        rows = pixel_values.mean(dim=(1, 3))[:, :DIM] if pixel_values.shape[2] >= DIM else None
        if rows is None:
            rows = pixel_values.mean(dim=(1, 3)).repeat(1, (DIM // pixel_values.shape[2]) + 1)[:, :DIM]
        base = torch.arange(PATCHES + 1, dtype=torch.float32).reshape(1, -1, 1) * 0.01
        return _Output(last_hidden_state=base + rows.reshape(n, 1, DIM))


def _ensemble(tmp_path: Path, *, whole_stack: bool = False, slice_pooling: str = "channels") -> Ensemble:
    device = torch.device("cpu")
    torch.manual_seed(SEED)
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = StudyModel(_FakeBackbone(), DIM, len(SLOTS), StudyDesign(slice_pooling=slice_pooling))
    config = {
        "img": 14,
        "whole_stack": whole_stack,
        "slice_pooling": slice_pooling,
        "slices": 6,
        "group": 3,
        "crop_mm": 130.0,
        "band": [0.2, 0.8],
        "slots": SLOTS,
        "rules": {"order": "normal", "lat": "centre", "slot_fallback": False, "decode_fill": "nearest"},
        "variant": "small",
        "pool": "cls_mean",
    }
    torch.save(model.state_dict(), tmp_path / "m.pt")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "members": [
                    {
                        "id": "m",
                        "file": "m.pt",
                        "fold": 0,
                        "config": config,
                        "fingerprint": model.fingerprint(14, 3, device).tolist(),
                        "holdout": 0.8,
                        "annot": 0.8,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    package = MemberPackage(tmp_path)
    return Ensemble(package, {"m": model}, device)


def test_windows():
    """Overlapping windows are every consecutive run; disjoint ones tile the stack without overlap."""
    ensemble = Ensemble.__new__(Ensemble)
    assert ensemble.windows(12, 3, overlap=True) == list(range(10))
    assert ensemble.windows(12, 3, overlap=False) == [0, 3, 6, 9]
    assert ensemble.windows(2, 3, overlap=True) == [0], "a stack thinner than a window still yields one"


def test_predict_member(tmp_path: Path):
    """A member's probabilities are averaged over its windows and land in [0, 1]."""
    ensemble = _ensemble(tmp_path)
    member = ensemble.package.members[0]
    images = torch.randint(0, 256, (2, len(SLOTS), 6, 14, 14), dtype=torch.uint8)
    mask = torch.ones(2, len(SLOTS))
    predictions = ensemble.predict_member(member, images, mask)
    assert predictions.shape == (2, len(TARGETS))
    assert float(predictions.min()) >= 0.0
    assert float(predictions.max()) <= 1.0


def test_combine():
    """Members combine by RANK per label, because rank order is all the metric reads."""
    ensemble = Ensemble.__new__(Ensemble)
    first = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
    # A wildly over-confident second member that AGREES on the ordering. Averaging probabilities would
    # let its 0.999 swamp the first member's 0.9; averaging ranks keeps the ordering both agree on.
    second = np.array([[0.999, 0.001], [0.998, 0.002], [0.997, 0.003]])
    combined = ensemble.combine([first, second])
    assert combined.shape == first.shape
    assert combined[:, 0].argsort().tolist() == [2, 1, 0], "the shared ordering of column 0 survives"
    assert combined.max() <= 1.0, "ranks are normalised, so no member's scale leaks through"


def test_combine_returns_a_single_member_untouched():
    """One member has nothing to combine with, and ranking it would discard its values for no gain."""
    ensemble = Ensemble.__new__(Ensemble)
    only = np.array([[0.4, 0.6]])
    assert np.array_equal(ensemble.combine([only]), only)


def test_submission():
    """The table is the study id followed by the twelve targets, in the order the metric expects."""
    ensemble = Ensemble.__new__(Ensemble)
    table = ensemble.submission(["a", "b"], np.zeros((2, len(TARGETS))))
    assert list(table)[0] == "StudyInstanceUID"
    assert list(table)[1:] == list(TARGETS)
    assert len(table["ACL"]) == 2


def test_predict_chunks(tmp_path: Path):
    """Chunks are consumed lazily and every member sees every chunk before the table is combined."""
    ensemble = _ensemble(tmp_path)
    images = torch.randint(0, 256, (2, len(SLOTS), 6, 14, 14), dtype=torch.uint8)
    mask = torch.ones(2, len(SLOTS))
    consumed = 0

    def source():
        nonlocal consumed
        for _ in range(3):
            consumed += 1
            yield images, mask

    combined = ensemble.predict_chunks(source(), overlap=False)
    assert consumed == 3, "the generator must be drained, not indexed"
    assert combined.shape == (6, len(TARGETS)), "three chunks of two studies"


def test_slices(tmp_path: Path):
    """The decode depth is the deepest stack any member was fitted on, never a default.

    A member trained on twenty slices and fed twelve raises nothing - the window loop just reads a
    shallower stack than the weights ever saw, and the fingerprint covers image size and group rather
    than depth. So the number has to come from the members, which is what this pins.
    """
    ensemble = _ensemble(tmp_path)
    assert ensemble.slices() == 6, "the fixture member declares six, so six is what gets decoded"

    ensemble.package.members[0].config["slices"] = 20
    assert ensemble.slices() == 20, "a deeper member must raise the decode depth, not be truncated to it"


def test_predict_member_reads_a_whole_stack_member_in_one_pass(tmp_path: Path):
    """A whole-stack member sees its ENTIRE stack at once, not `group` slices at a time.

    Its slot embedding is the elementwise maximum over the full stack, so windowing it by three would
    hand it maxima over a quarter of the stack — a different function of the same pixels. Nothing
    raises when that happens and the holdout cannot expose it, because holdout scoring reads the same
    window training did. The leaderboard did: two members at 0.8481 holdout scored 0.882 while five at
    0.8269 scored 0.888.

    The stack here is deeper than the member wants, which is the real case — a chunk is decoded to the
    deepest depth ANY member needs, so a member fitted on six must be handed the middle six of eight,
    exactly as `Trainer.predict` takes the middle window.
    """
    ensemble = _ensemble(tmp_path / "whole", whole_stack=True, slice_pooling="max")
    member = ensemble.package.members[0]
    images = torch.randint(0, 256, (2, len(SLOTS), 8, 14, 14), dtype=torch.uint8)
    mask = torch.ones(2, len(SLOTS))

    predictions = ensemble.predict_member(member, images, mask, overlap=True)

    model = ensemble.models[member.identifier]
    model.eval()
    with torch.no_grad():
        expected = torch.sigmoid(model(images[:, :, 1:7], mask, member.read_size).float()).numpy()
    assert np.allclose(predictions, expected), "one pass over the middle six slices, not ten windows of three"


def test_predict_member_still_windows_a_member_that_is_not_whole_stack(tmp_path: Path):
    """The published members were fitted windowed and must stay windowed — the two paths differ."""
    windowed = _ensemble(tmp_path / "win", whole_stack=False, slice_pooling="max")
    member = windowed.package.members[0]
    images = torch.randint(0, 256, (2, len(SLOTS), 8, 14, 14), dtype=torch.uint8)
    mask = torch.ones(2, len(SLOTS))

    model = windowed.models[member.identifier]
    model.eval()
    with torch.no_grad():
        single = torch.sigmoid(model(images[:, :, 1:7], mask, member.read_size).float()).numpy()
    assert not np.allclose(windowed.predict_member(member, images, mask, overlap=True), single), (
        "a windowed member averaged over its windows is not one pass over the stack"
    )


def test_decoded_size(tmp_path: Path):
    """The decode resolution comes from the members, and disagreement is refused rather than resolved.

    A single value, not a maximum: depth can be windowed down at inference but resolution cannot, so
    two members wanting different sizes genuinely need two decodes. Silently picking one would hand the
    other an image it never saw — 336 px upsampled to 392 where training downsampled from 448.
    """
    ensemble = _ensemble(tmp_path / "one")
    assert ensemble.decoded_size() == 14

    member = ensemble.package.members[0]
    coarser: MemberConfig = {**member.config, "img": 28}
    ensemble.package.members = [member, replace(member, identifier="other", config=coarser)]
    with pytest.raises(ValueError, match="two decodes"):
        ensemble.decoded_size()
