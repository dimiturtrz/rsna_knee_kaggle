from typing import cast, override

import numpy as np
import pytest
import torch
from torch import nn

from core.data.corpus.slot_reader import GROUP
from core.findings import TARGETS
from kneemri.evaluation.attribution import Attribution, AttributionReport, Diagnostics, Holdout
from kneemri.inference.holdout_predictions import HoldoutPredictions, Window
from kneemri.models.model import StudyModel

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


def _holdout(starts: list[int] | None = None, width: int = GROUP) -> Holdout:
    """The fixture as the object the diagnostics take.

    The WIDTH travels inside the window rather than being supplied by the scorer. It used to be the
    constant GROUP there, so a whole-stack member - whose window is its full depth - was occluded three
    slices at a time in every diagnostic this module produces.
    """
    images, mask, truth = _fixture()
    return Holdout(images, mask, truth, Window(starts=starts or [0], width=width, size=SIZE))


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


def _report(
    slots: dict[str, float] | None = None,
    regions: dict[str, float] | None = None,
    rings: dict[str, float] | None = None,
    depth_by_label: dict[str, dict[str, float]] | None = None,
    slices: dict[str, dict[str, float]] | None = None,
) -> AttributionReport:
    """A report over stated diagnostics, the rest filled with a shape each section can render.

    Every field defaults because each section reads its own: a depth test that has to spell out slot
    and ring numbers is stating things the assertion does not depend on.
    """
    return AttributionReport(
        Diagnostics(
            slots=slots or {"0": 0.01, "2": 0.30},
            regions=regions or {"whole": 0.83, "centre_only": 0.82, "border_only": 0.51},
            rings=rings or {"0.05": 0.52, "0.25": 0.74},
            depth_by_label=depth_by_label or {"0": dict.fromkeys(TARGETS, 0.80), "3": dict.fromkeys(TARGETS, 0.83)},
            slices=slices or {"0": dict.fromkeys(TARGETS, 0.02), "1": dict.fromkeys(TARGETS, 0.01)},
        )
    )


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


def test_patch_importance():
    """The map has to put its mass where the fixture wrote the signal, not spread it over the field.

    The fixture's carrier writes into `4:12` of a 16 px field, which on a four-cell grid is exactly the
    inner two-by-two. The stand-in reads nothing else, so blanking an outer cell must cost precisely
    nothing — that zero is what separates a real localisation from a map that merely looks peaked.
    """
    maps = _attribution().patch_importance(cast(StudyModel, _SlotReader()), _holdout(), grid=4)

    assert maps.shape == (STUDIES, len(TARGETS), 4, 4)
    total = np.abs(maps).sum()
    centre = np.abs(maps[..., 1:3, 1:3]).sum()
    assert centre / total > 0.9, "a diffuse read would leave a quarter of the mass in the centre cells"


def test_depth():
    """The profile is the per-label table meaned over labels, not a second pass over the model."""
    by_label = {"0": {"ACL": 0.80, "MCL": 0.90}, "3": {"ACL": 0.70, "MCL": 0.70}}
    averaged = Diagnostics(slots={}, regions={}, rings={}, depth_by_label=by_label, slices={}).depth

    assert sorted(averaged) == ["0", "3"]
    assert averaged["0"] == pytest.approx(0.85)
    assert averaged["3"] == pytest.approx(0.70)


def test_ring_importance():
    """A ring that holds none of the signal must fall toward chance as it narrows.

    The fixture writes its label into the centre of one slot, so every ring is blind to it. This is
    the control the whole diagnostic rests on: if a ring scored well HERE, the measurement would be
    reading something other than the pixels it kept.
    """
    rings = _attribution().ring_importance(cast(StudyModel, _SlotReader()), _holdout(), widths=(0.125, 0.25))

    assert sorted(rings) == ["0.12", "0.25"]
    assert rings["0.12"] < 0.6, "a ring holding no signal must be near chance"


def test_depth_by_label():
    """Each window is scored per diagnosis, so two labels can be seen to prefer different depths."""
    starts = [0, 1]
    by_label = _attribution().depth_by_label(cast(StudyModel, _SlotReader()), _holdout(starts))

    assert sorted(by_label) == ["0", "1"]
    assert list(by_label["0"]) == list(TARGETS), "every label keeps its own score at every depth"


def test_slice_importance():
    """A stack whose every slice carries the same signal must cost nothing when one is replaced.

    The fixture writes the label into EVERY slice of the carrier slot, so this is the redundant case,
    and it is the control the diagnostic rests on: a non-zero drop here would be the occlusion
    disturbing the model rather than removing information.
    """
    by_label = _attribution().slice_importance(cast(StudyModel, _SlotReader()), _holdout(width=SLICES))

    assert sorted(by_label, key=int) == [str(index) for index in range(SLICES)]
    assert list(by_label["0"]) == list(TARGETS), "every label keeps its own cost at every depth"
    assert all(abs(drop) < 1e-6 for label in by_label.values() for drop in label.values())


def test_slice_cost_names_the_verdict():
    """One slice holding most of the cost, with every label agreeing, is the actionable combination."""
    concentrated = {"0": dict.fromkeys(TARGETS, 0.30), "1": dict.fromkeys(TARGETS, 0.01)}
    lines = "\n".join(_report(slices=concentrated)._slice_cost())

    assert "slice 0 carries 97% of the total cost over 2 slices" in lines
    assert "1 distinct depths over 12 labels" in lines
    assert "still has something to collect" in lines


def test_slice_cost_when_no_slice_costs_anything():
    """A redundant stack is reported as such rather than dividing by a zero total."""
    lines = "\n".join(_report(slices={"0": dict.fromkeys(TARGETS, 0.0)})._slice_cost())

    assert "carries the finding redundantly" in lines


def test_lines():
    """The report states the verdict each diagnostic supports rather than leaving numbers bare."""
    lines = _report(
        slots={"0": 0.01, "2": 0.30},
        regions={"whole": 0.83, "centre_only": 0.82, "border_only": 0.51},
        rings={"0.05": 0.52, "0.25": 0.74},
        depth_by_label={"0": dict.fromkeys(TARGETS, 0.80), "3": dict.fromkeys(TARGETS, 0.83)},
        slices={"0": dict.fromkeys(TARGETS, 0.02), "1": dict.fromkeys(TARGETS, 0.01)},
    ).lines()
    text = "\n".join(lines)
    assert "spread 0.2900" in text
    assert "ring 0.05: 0.5200" in text
    assert "the narrowest ring holds no joint" in text
    assert "slice 3" in text, "the best window has to be named, not left to be read off"
    assert "1 distinct depths over 12 labels" in text, "agreement across labels is what gates acting on depth"


def test_depth_refuses_a_single_window():
    """One window is not a depth profile, and the section used to report it as its strongest finding.

    With `len(depth) == 1` the best window is the whole reading, the range is `x to x`, and every
    label prefers the only start on offer — which printed as "one shared depth means a re-weighting
    of the sampled windows helps every label". A licence to act, from an n of one.
    """
    lines = "\n".join(
        _report(
            depth_by_label={"0": dict.fromkeys(TARGETS, 0.8558)},
        )._depth()
    )

    assert "not measured" in lines, "the axis says what it could not ask"
    assert "1 window(s)" in lines, "and names the reason"
    assert "distinct depths" not in lines, "the agreement verdict n=1 cannot support is absent"
    assert "best single window" not in lines, "so is the best-window claim"
