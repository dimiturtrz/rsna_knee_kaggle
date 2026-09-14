"""Boxes drawn from known tissue, crops of known pixels, and a whole ROI build over a cache the test writes."""

from pathlib import Path

import numpy as np
import pytest
import torch

from core.data.corpus.slots import ROI_SCHEME, ROI_SLOTS, SLOTS
from core.data.corpus.study_cache import StudyCache
from kneemri.embeddings.roi_cache import LOCATED_SLICES, SAGITTAL, BoxRule, RoiBox, RoiCache
from kneemri.models.locator import Locator

SIDE = 28
SLICES = 4
CPU = torch.device("cpu")
RULE = BoxRule(margin=1, minimum=8, size=SIDE)


def _drawn(top: int, left: int, extent: int, slices: int = SLICES) -> torch.Tensor:
    drawn = torch.zeros(SLICES, SIDE, SIDE, dtype=torch.long)
    drawn[:slices, top : top + extent, left : left + extent] = 1
    return drawn


def _caches(directory: Path, studies: int) -> RoiCache:
    source = StudyCache(directory, slices=SLICES, image_size=SIDE)
    source.prepare()
    np.save(source.images_path, np.full(source.shape(studies), 7, dtype=np.uint8))
    source.record([f"s{k}" for k in range(studies)])
    mask = np.ones((studies, len(SLOTS)), dtype=np.float32)
    mask[0, SAGITTAL[1]] = 0
    np.save(source.mask_path, mask)
    return RoiCache(source, source.under(ROI_SCHEME), Locator.__new__(Locator), CPU)


def test_blank() -> None:
    assert RoiBox(0, 0, SIDE, LOCATED_SLICES - 1, 0).blank
    assert not RoiBox(0, 0, SIDE, LOCATED_SLICES, 0).blank


def test_row() -> None:
    row = RoiBox(7, 0, 14, LOCATED_SLICES, 9).row("s0", SAGITTAL[2], SIDE)
    assert (row["slot"], row["top_share"], row["side_share"], row["blank"]) == (SLOTS[SAGITTAL[2]].name, 0.25, 0.5, 0)


def test_at() -> None:
    rule = BoxRule.at(336)
    assert (rule.margin, rule.minimum, rule.size) == (23, 207, 336)


def test_box() -> None:
    box = RULE.box(_drawn(10, 12, 10))
    assert box.side == 11
    assert not box.blank
    assert abs(box.top + box.side / 2 - 14.5) <= 1


def test_box_floor_and_blank() -> None:
    assert RULE.box(_drawn(10, 10, 5)).side == RULE.minimum
    assert RULE.box(_drawn(10, 10, 5, slices=LOCATED_SLICES - 1)).blank


def test_placed() -> None:
    assert RULE.placed(2.0, 10) == 0
    assert RULE.placed(27.0, 10) == SIDE - 10
    assert RULE.placed(14.0, 10) == 9


def test_crop() -> None:
    stack = torch.zeros(2, SIDE, SIDE, dtype=torch.uint8)
    stack[:, :14, :14] = 200
    cropped = RoiCache.crop(stack, RoiBox(0, 0, 14, LOCATED_SLICES, 0))
    assert cropped.shape == stack.shape
    assert int(cropped.min()) == 200


def test_located(tiny_locator: Locator) -> None:
    builder = RoiCache(StudyCache(Path()), StudyCache(Path()), tiny_locator, CPU)
    box = builder.located(torch.zeros(SLICES, SIDE, SIDE, dtype=torch.uint8), RULE)
    assert 0 < box.side <= SIDE


def test_table(tmp_path: Path) -> None:
    assert _caches(tmp_path, 1).table.name == f"roi_boxes_{SIDE}px_{SLICES}sl_{ROI_SCHEME}.csv"


def test_build(tmp_path: Path, tiny_locator: Locator, monkeypatch: pytest.MonkeyPatch) -> None:
    builder = _caches(tmp_path, 2)
    builder = RoiCache(builder.source, builder.destination, tiny_locator, CPU)
    monkeypatch.setattr(BoxRule, "at", staticmethod(lambda size: BoxRule(1, 8, size)))
    monkeypatch.setattr(BoxRule, "box", lambda self, drawn: RoiBox(2, 2, 14, LOCATED_SLICES, 1))
    rows = builder.build()
    images, mask = np.load(builder.destination.images_path), np.load(builder.destination.mask_path)
    assert images.shape == (2, len(ROI_SLOTS), SLICES, SIDE, SIDE)
    assert len(rows) == 2 * len(SAGITTAL) - 1
    assert mask[0, len(SLOTS) + 1] == 0
    assert mask[1, len(SLOTS) :].tolist() == [1, 1, 1]
    assert int(images[1, len(SLOTS)].min()) == 7
    with pytest.raises(FileExistsError):
        builder.build()


def test_write(tmp_path: Path) -> None:
    builder = _caches(tmp_path, 1)
    builder.write([RoiBox(0, 0, SIDE, 0, 0).row("s0", 0, SIDE)])
    assert builder.table.read_text().splitlines()[0].startswith("study,slot,plane")


def test_misses() -> None:
    rows = [RoiBox(0, 0, SIDE, 0, 0).row("s0", 0, SIDE), RoiBox(0, 0, 9, LOCATED_SLICES, 5).row("s1", 0, SIDE)]
    lines = RoiCache.misses(rows, 3)
    assert lines[0].startswith(f"{SLOTS[0].name}: blank 1/2")
    assert lines[-1].endswith("1/3")
