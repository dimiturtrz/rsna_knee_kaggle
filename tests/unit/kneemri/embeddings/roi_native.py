"""Box scaling, table reading and the re-cut of known pixels."""

from pathlib import Path

import torch

from kneemri.embeddings.roi_native import NativeBox

SIZE = 12


def test_scaled() -> None:
    assert NativeBox("SAG_T1", 1, 2, 5, blank=False).scaled(2) == NativeBox("SAG_T1", 2, 4, 10, blank=False)


def test_read(tmp_path: Path) -> None:
    table = tmp_path / "boxes.csv"
    table.write_text("study,slot,top,left,side,blank\ns0,SAG_T1,1,2,8,0\ns0,SAG_FLUID_FS,0,0,8,1\n")
    boxes = NativeBox.read(table)
    assert boxes["s0"] == [NativeBox("SAG_T1", 1, 2, 8, blank=False), NativeBox("SAG_FLUID_FS", 0, 0, 8, blank=True)]


def test_recut() -> None:
    stack = torch.zeros(3, 2 * SIZE, 2 * SIZE, dtype=torch.uint8)
    stack[:, 4:16, 4:16] = 200
    cropped = NativeBox("SAG_T1", 2, 2, 6, blank=False).recut(stack, SIZE)
    assert cropped.shape == (3, SIZE, SIZE)
    assert int(cropped.min()) == 200
