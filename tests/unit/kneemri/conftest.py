"""A `Locator` over a CPU-sized fake trunk and a slice corpus, shared by the model, fit and audit mirror tests."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import override

import numpy as np
import pandas as pd
import pytest
import torch
from torch import Tensor, nn

from core.data.corpus.skmtea_boxes import SCAN_COLUMN
from core.data.corpus.skmtea_slices import IMAGES, LABELS, SLICE_TABLE, SPLIT_COLUMN
from kneemri.models.backbone import ENCODERS
from kneemri.models.locator import LAYERS, PATCH, Locator, SliceCorpus
from kneemri.models.variants import SMALL

WIDTH = 8
CORPUS_SIDE = 28


class _PatchTrunk(nn.Module):
    """The HuggingFace ViT contract the locator reads: `hidden_states`, a class token first, per depth."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Conv2d(3, WIDTH, PATCH, stride=PATCH)

    @override
    def forward(self, pixel_values: Tensor, *, output_hidden_states: bool) -> SimpleNamespace:
        tokens = self.embed(pixel_values).flatten(2).transpose(1, 2)
        sequence = torch.cat([tokens.mean(1, keepdim=True), tokens], dim=1)
        return SimpleNamespace(hidden_states=(sequence,) * (max(LAYERS) + 1) if output_hidden_states else ())


@pytest.fixture
def tiny_locator() -> Locator:
    torch.manual_seed(0)
    return Locator(_PatchTrunk(), replace(ENCODERS[SMALL], dim=WIDTH))


@pytest.fixture
def slice_directory(tmp_path: Path) -> Path:
    """Eight random slices of two echoes: four train, two val, two test."""
    rng = np.random.default_rng(0)
    splits = ["train"] * 4 + ["val"] * 2 + ["test"] * 2
    np.save(tmp_path / IMAGES, rng.integers(0, 256, (len(splits), 2, CORPUS_SIDE, CORPUS_SIDE), dtype=np.uint8))
    np.save(tmp_path / LABELS, rng.integers(0, 7, (len(splits), CORPUS_SIDE, CORPUS_SIDE), dtype=np.uint8))
    table = pd.DataFrame({SCAN_COLUMN: ["MTR_0"] * len(splits), SPLIT_COLUMN: splits})
    table.to_csv(tmp_path / SLICE_TABLE, index=False)
    return tmp_path


@pytest.fixture
def slice_corpus(slice_directory: Path) -> SliceCorpus:
    return SliceCorpus.read(slice_directory)
