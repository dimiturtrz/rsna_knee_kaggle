"""The per-band transfer reading, on a board file and a pool this test builds itself.

The disk half is real — a row is rebuildable only when every member it names has a checkpoint — so the
board file and the checkpoints are written into `tmp_path`. The PREDICTION half is stubbed, because
reading real holdout records would make this a test about the data root rather than about the arithmetic
of cutting a holdout into bands.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.transfer_strata import Stratum, StratumReading, TransferStrata
from kneemri.inference.vote import LogitAverage

STUDIES = 40
LABELS = len(TARGETS)
"""The full finding vocabulary, not a convenient three: `LabelScores.per_label` scores every target by
position, so a narrower table indexes past its own last column."""


def _board(directory: Path) -> Path:
    """A board file holding one unrebuildable row for every reason a row can be unrebuildable."""
    rows = [
        {"id": "low", "scope": "ensemble", "offline": 0.80, "board": 0.920, "members_named": ["a", "b"]},
        {"id": "mid", "scope": "ensemble", "offline": 0.81, "board": 0.928, "members_named": ["a", "c"]},
        {"id": "high", "scope": "ensemble", "offline": 0.82, "board": 0.936, "members_named": ["b", "c"]},
        {"id": "dark", "scope": "ensemble", "offline": 0.83, "board": 0.941, "members_named": ["a", "missing"]},
        {"id": "solo", "scope": "member_mean", "offline": 0.79, "board": 0.910, "members_named": ["a"]},
        {"id": "unscored", "scope": "ensemble", "offline": None, "board": 0.900, "members_named": ["a"]},
    ]
    record = directory / "board.jsonl"
    record.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    for name in ("a", "b", "c"):
        (directory / f"{name}.pt").write_bytes(b"")
    return record


STRENGTH: dict[tuple[str, ...], float] = {("a", "b"): 0.30, ("a", "c"): 0.55, ("b", "c"): 0.80}
"""How closely each entry's members track the truth. Keyed on the MEMBERS so two entries differ by more
than the board's resolving floor — a fixture where every entry scores the same resolves no pairs, and a
test asserting over zero pairs asserts nothing."""


def _pool(names: list[str]) -> Pool:
    """Members that agree on the first half of the studies and disagree on the second.

    Disagreement is what the module cuts on, so it is built in rather than left to chance: the second
    half is drawn independently per member, the first half is not.
    """
    generator = np.random.default_rng(7)
    truth = (generator.random((STUDIES, LABELS)) > 0.5).astype(np.float64)
    strength = STRENGTH.get(tuple(names), 0.5)
    predictions = []
    for _ in names:
        member = truth * strength + generator.random((STUDIES, LABELS)) * (1.0 - strength)
        member[STUDIES // 2 :] = generator.random((STUDIES - STUDIES // 2, LABELS))
        predictions.append(np.clip(member, 0.0, 1.0))
    return Pool(list(names), predictions, truth, LogitAverage())


@pytest.fixture
def strata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TransferStrata:
    """A reading whose board file is real and whose predictions are the fixture pool above."""
    reading = TransferStrata(models=tmp_path, record=_board(tmp_path), bands=2)
    monkeypatch.setattr(reading, "_pool", _pool)
    return reading


def test_rows(strata: TransferStrata):
    """Every recorded line comes back, including the ones no reading can price."""
    assert [row["id"] for row in strata.rows()] == ["low", "mid", "high", "dark", "solo", "unscored"]


def test_honest(strata: TransferStrata):
    """A row is kept only with both numbers, this scope, and every member on disk.

    `dark` is the case that matters: a member with no checkpoint is a member that held nothing out, so
    the row cannot be scored on our holdout at all and must not be scored on the part it can reach.
    """
    assert [row["id"] for row in strata.honest()] == ["low", "mid", "high"]


def test_members(strata: TransferStrata):
    """The reference pool is the union over rebuildable rows, each member once and in first-seen order."""
    assert strata.members() == ["a", "b", "c"]


def test_disagreement(strata: TransferStrata):
    """One number per study, and it is larger where the fixture made the members disagree."""
    spread = strata.disagreement()
    assert spread.shape == (STUDIES,)
    assert spread[STUDIES // 2 :].mean() > spread[: STUDIES // 2].mean()


def test_strata(strata: TransferStrata):
    """Equal-count bands, easiest first, together covering every study exactly once."""
    bands = strata.strata()
    assert [band.band for band in bands] == [0, 1]
    assert sorted(np.concatenate([band.rows for band in bands]).tolist()) == list(range(STUDIES))
    assert bands[0].difficulty < bands[1].difficulty


def test_named():
    """Bands are numbered from one in the report, because the report is read by people."""
    assert Stratum(band=0, rows=np.array([0], dtype=np.int64), difficulty=0.1).named() == "Q1"


def test_readings(strata: TransferStrata):
    """Every rebuildable entry is scored in every band, carrying the board number it was given."""
    found = strata.readings()
    assert {(reading.entry, reading.band) for reading in found} == {
        (entry, band) for entry in ("low", "mid", "high") for band in (0, 1)
    }
    assert all(reading.scoreable <= LABELS for reading in found)


def test_compression(strata: TransferStrata):
    """The median board-per-holdout ratio over the pairs the holdout separated.

    A band that separates no pair reports NaN rather than a number: zero resolved pairs is a statement
    about the band, and averaging it in as a slope of zero would read as saturation that was never
    measured.
    """
    ladder = [
        StratumReading(entry="low", band=0, offline=0.80, board=0.920, scoreable=LABELS),
        StratumReading(entry="mid", band=0, offline=0.81, board=0.930, scoreable=LABELS),
        StratumReading(entry="high", band=0, offline=0.82, board=0.940, scoreable=LABELS),
    ]
    assert strata.compression(ladder) == pytest.approx(1.0)
    assert np.isnan(strata.compression(ladder[:1]))
    assert np.isnan(strata.compression([]))


def test_attribution(strata: TransferStrata):
    """Both halves are priced off the pairs the holdout resolved, and an unresolvable set says so.

    The fixture's three entries differ by hundredths, so what is asserted is the SHAPE of the answer —
    two finite medians over a counted set of pairs — not a direction. A direction asserted on synthetic
    predictions would be a test of the fixture.
    """
    found = strata.attribution()
    assert found.pairs > 0
    assert np.isfinite(found.hard) and np.isfinite(found.easy)


def test_lines(strata: TransferStrata):
    """The report names every band and every entry it priced."""
    lines = strata.lines()
    assert any(line.startswith("Q1") for line in lines)
    assert any(line.startswith("Q2") for line in lines)
    assert any("low" in line for line in lines)
