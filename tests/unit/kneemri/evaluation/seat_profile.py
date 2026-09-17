"""The per-member, per-finding profile: what one member scores, and which member leads a finding."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from core.findings import TARGETS
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.seat_profile import SeatProfile
from kneemri.runs.holdout_record import HoldoutRecord

STUDIES = ["s1", "s2", "s3", "s4"]
TRUTH_COLUMN = [1.0, 0.0, 1.0, 0.0]
"""Two positives against two negatives, so every column has both classes and no AUC is undefined."""

LEADER, LAGGARD = TARGETS[0], TARGETS[1]
PERFECT, INVERTED = 1.0, 0.0


def _truth() -> NDArray[np.float32]:
    """The same alternating column repeated across all twelve findings."""
    return np.tile(np.array(TRUTH_COLUMN, dtype=np.float32)[:, None], (1, len(TARGETS)))


def _member(directory: Path, name: str, right: str) -> Path:
    """A checkpoint whose record ranks `right` perfectly and every other finding backwards."""
    truth = _truth()
    predictions = 1.0 - truth.astype(np.float64)
    predictions[:, TARGETS.index(right)] = truth[:, TARGETS.index(right)]
    path = directory / f"{name}.pt"
    HoldoutRecord(path).write(STUDIES, predictions, truth, epoch=3)
    return path


def test_scores_defaults_to_ledger75() -> None:
    """The unspecified cut is LEDGER75, so a per-finding leader is ordered on the ruler the board agrees with."""
    assert SeatProfile().scores.name() == "ledger75"
    assert LabelScores().name() != "ledger75", "the midpoint has to stay reachable and distinguishable"


def test_reading(tmp_path: Path) -> None:
    """The kept arrays come back under the study names and the epoch they were written with."""
    path = _member(tmp_path, "m1", LEADER)
    reading = SeatProfile().reading(path)
    assert reading.studies == STUDIES
    assert reading.epoch == 3
    assert reading.predictions.shape == (len(STUDIES), len(TARGETS))


def test_reading_refuses_a_member_with_no_record(tmp_path: Path) -> None:
    """A `--fold -1` member held nothing out, and the profile must say so rather than score nothing."""
    with pytest.raises(FileNotFoundError, match="no holdout record"):
        SeatProfile().reading(tmp_path / "dark.pt")


def test_scored(tmp_path: Path) -> None:
    """One member's twelve AUCs, at the endpoints the fixture was built to land on."""
    scored = SeatProfile().scored(_member(tmp_path, "m1", LEADER))
    assert set(scored) == set(TARGETS)
    assert scored[LEADER] == PERFECT
    assert scored[LAGGARD] == INVERTED


def test_table(tmp_path: Path) -> None:
    """Every member keyed by its checkpoint stem, which is the name the ledger quotes."""
    table = SeatProfile().table([_member(tmp_path, "m1", LEADER), _member(tmp_path, "m2", LAGGARD)])
    assert sorted(table) == ["m1", "m2"]
    assert table["m2"][LAGGARD] == PERFECT


def test_leaders(tmp_path: Path) -> None:
    """The per-finding leader is the member that scores highest THERE, not the one with the best mean.

    Both members here hold the same mean, so a mean-ranked answer could not distinguish them — which is
    the whole reason this module exists beside `seat_value`.
    """
    profile = SeatProfile()
    table = profile.table([_member(tmp_path, "m1", LEADER), _member(tmp_path, "m2", LAGGARD)])
    leaders = profile.leaders(table)
    assert leaders[LEADER] == ("m1", PERFECT)
    assert leaders[LAGGARD] == ("m2", PERFECT)


def test_spread(tmp_path: Path) -> None:
    """Best minus worst per finding — wide where the pool disagrees, and never negative."""
    profile = SeatProfile()
    table = profile.table([_member(tmp_path, "m1", LEADER), _member(tmp_path, "m2", LAGGARD)])
    spread = profile.spread(table)
    assert spread[LEADER] == pytest.approx(PERFECT - INVERTED)
    assert min(spread.values()) >= 0.0
