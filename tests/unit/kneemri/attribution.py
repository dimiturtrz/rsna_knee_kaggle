from typing import cast, override

import numpy as np
import torch
from torch import nn

from kneemri.attribution import Attribution, Holdout
from kneemri.holdout_predictions import HoldoutPredictions
from kneemri.model import TARGETS, StudyModel

SEED = 2026
STUDIES = 8
SLOTS = 6
SLICES = 6
SIZE = 16
CARRIER = 2
"""The slot the fake model actually reads — the one a slot diagnostic has to single out."""


class _SlotReader(nn.Module):
    """A stand-in that reads ONE slot's CENTRE, so both occlusions have a knowable right answer."""

    @override
    def forward(self, imgs: torch.Tensor, mask: torch.Tensor, img_size: int | None = None) -> torch.Tensor:
        centre = imgs[:, CARRIER, :, 4:12, 4:12].float().mean(dim=(1, 2, 3)) / 255.0
        present = mask[:, CARRIER]
        return ((centre * present) * 10.0 - 5.0).unsqueeze(1).expand(-1, len(TARGETS))


def _holdout(starts: list[int] | None = None) -> Holdout:
    """The fixture as the object the diagnostics take."""
    images, mask, truth = _fixture()
    return Holdout(images, mask, truth, starts or [0], SIZE)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Studies whose label is written into the centre of the carrier slot, and noise everywhere else."""
    generator = np.random.default_rng(SEED)
    images = generator.integers(0, 60, (STUDIES, SLOTS, SLICES, SIZE, SIZE), dtype=np.uint8)
    truth = np.zeros((STUDIES, len(TARGETS)), dtype=np.float32)
    positive = np.arange(STUDIES) % 2 == 0
    images[positive, CARRIER, :, 4:12, 4:12] = 250
    truth[positive] = 1.0
    return images, np.ones((STUDIES, SLOTS), dtype=np.float32), truth


def _attribution() -> Attribution:
    return Attribution(HoldoutPredictions(torch.device("cpu")))


def test_slot_importance():
    """Withholding the slot that carries the signal costs the score; the others cost nothing."""
    drops = _attribution().slot_importance(cast(StudyModel, _SlotReader()), _holdout())

    assert drops[str(CARRIER)] > 0.2, "removing the carrier must cost real score"
    others = [drop for slot, drop in drops.items() if slot != str(CARRIER)]
    assert all(abs(drop) < 1e-6 for drop in others), "a slot the model ignores must cost nothing"


def test_region_importance():
    """Keeping only the centre preserves the score; keeping only the border destroys it."""
    regions = _attribution().region_importance(cast(StudyModel, _SlotReader()), _holdout())

    assert regions["centre_only"] >= regions["whole"] - 1e-6
    assert regions["border_only"] < regions["centre_only"], "the border holds no signal in this fixture"


def test_depth_profile():
    """Every window is scored on its own, so the profile has one entry per start."""
    starts = [0, 1, 2]
    depth = _attribution().depth_profile(cast(StudyModel, _SlotReader()), _holdout(starts))

    assert sorted(depth) == sorted(str(start) for start in starts)
    assert all(np.isfinite(value) for value in depth.values())


def test_report():
    """The report states the verdict each diagnostic supports rather than leaving numbers bare."""
    lines = _attribution().report(
        {"0": 0.01, "2": 0.30}, {"whole": 0.83, "centre_only": 0.82, "border_only": 0.51}, {"0": 0.80, "3": 0.83}
    )
    text = "\n".join(lines)
    assert "spread 0.2900" in text
    assert "border-only should be near chance" in text
    assert "slice 3" in text, "the best window has to be named, not left to be read off"


def test_targets():
    """The diagnostics average over the competition's twelve labels, in their order."""
    assert _attribution().targets() == list(TARGETS)
