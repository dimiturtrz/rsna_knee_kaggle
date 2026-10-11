"""The per-member, per-finding profile: what one member scores, and which member leads a finding."""

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from core.findings import TARGETS
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.evaluation.seat_profile import PricedColumns, RankedColumns, SeatProfile
from kneemri.runs.holdout_record import SUFFIX, HoldoutRecord

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


def test_readings(tmp_path: Path) -> None:
    """The pool's kept arrays, keyed by stem and checked onto one ruler before anything is compared."""
    readings = SeatProfile().readings([_member(tmp_path, "m1", LEADER), _member(tmp_path, "m2", LAGGARD)])
    assert sorted(readings) == ["m1", "m2"]
    assert readings["m1"].studies == STUDIES


def test_profile(tmp_path: Path) -> None:
    """An already-read pool scores exactly as `table` does, so the second caller pays no second read."""
    seat = SeatProfile()
    paths = [_member(tmp_path, "m1", LEADER), _member(tmp_path, "m2", LAGGARD)]
    assert seat.profile(seat.readings(paths)) == seat.table(paths)


def test_population(tmp_path: Path) -> None:
    """A member graded by another table answers another question, so it is dropped before anything ranks it."""
    ours = _member(tmp_path, "m1", LEADER)
    moved = _member(tmp_path, "moved", LEADER)
    kept = HoldoutRecord(moved).required()
    HoldoutRecord(moved).write(STUDIES, kept.predictions, _truth() * 0.5, epoch=3)
    assert SeatProfile().population([ours, moved], "m1") == [ours]


def test_margins(tmp_path: Path) -> None:
    """A finding's leader is taken from the pool MINUS the named member, and priced against it with an interval."""
    seat = SeatProfile()
    paths = [_member(tmp_path, "m1", LEADER), _member(tmp_path, "m2", LAGGARD)]
    readings = seat.readings(paths)
    margins = seat.margins(readings, seat.profile(readings), "m2")
    name, margin, (low, high) = margins[LEADER]
    assert name == "m1", "m2 is the baseline and may not be its own challenger"
    assert margin == pytest.approx(PERFECT - INVERTED)
    assert low <= margin <= high, "the interval brackets the margin, or the pair was passed in the wrong order"


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


PIN_UNRANKED = [f"{target:22s} m0                            0.0000  0.0000" for target in TARGETS[3:]]
PIN_MEMBERS = [
    "",
    "m0                                 mean 0.0833  weakest MCL 0.0000",
    "m1                                 mean 0.0833  weakest ACL 0.0000",
    "m2                                 mean 0.0833  weakest ACL 0.0000",
]
PIN_PLAIN = [
    "--- per finding @ledger75",
    "finding                leader                          best  spread",
    "ACL                    m0                            1.0000  1.0000",
    "MCL                    m1                            1.0000  1.0000",
    "Medial Meniscus        m2                            1.0000  1.0000",
    *PIN_UNRANKED,
    *PIN_MEMBERS,
]
PIN_AGAINST = [
    "--- per finding @cut0.5",
    "finding                leader                          best  spread    margin       95% interval  clears",
    "ACL                    m0                            1.0000  1.0000    1.0000 [  1.0000,  1.0000]  YES",
    "MCL                    m1                            1.0000  1.0000    1.0000 [  1.0000,  1.0000]  YES",
    "Medial Meniscus        m0                            0.0000  1.0000   -1.0000 [ -1.0000, -1.0000]  no",
    *[f"{line}    0.0000 [  0.0000,  0.0000]  no" for line in PIN_UNRANKED],
    *PIN_MEMBERS,
]
"""Both reports pinned verbatim off the module before main was split into its paths and its rows."""


def _two_columns(tmp_path: Path, against: str | None) -> RankedColumns:
    """The columns over two members, each right on a different finding, priced against `against` if named."""
    seat = SeatProfile(LabelScores(MIDPOINT))
    readings = seat.readings([_member(tmp_path, "m1", LEADER), _member(tmp_path, "m2", LAGGARD)])
    return seat.columns(readings, seat.profile(readings), against)


def test_paths(tmp_path: Path) -> None:
    """Named members pass through untouched; a holdouts directory yields one record path per pooled member."""
    seat = SeatProfile()
    named = [tmp_path / "x.pt"]
    assert seat.paths(named, None, tmp_path / "unused") == named
    _members(tmp_path)
    declines = tmp_path / "declines"
    declines.write_text("", encoding="utf-8")
    assert sorted(path.name for path in seat.paths(None, tmp_path, declines)) == [
        f"m{index}{SUFFIX}" for index in range(3)
    ]


def test_columns(tmp_path: Path) -> None:
    """No baseline gives the plain ranking; a named one gives the priced ranking, leaders drawn from the rest."""
    plain = _two_columns(tmp_path, None)
    priced = _two_columns(tmp_path, "m2")
    assert type(plain) is RankedColumns
    assert isinstance(priced, PricedColumns)
    assert (plain.leader[LAGGARD], priced.leader[LAGGARD]) == ("m2", "m1"), "the baseline never leads"


def test_ranked_columns_header(tmp_path: Path) -> None:
    """The plain header carries no margin columns."""
    assert _two_columns(tmp_path, None).header() == PIN_PLAIN[1]


def test_ranked_columns_row(tmp_path: Path) -> None:
    """A row names the finding, its leader, the leader's score and the pool's spread."""
    assert _two_columns(tmp_path, None).row(LEADER) == f"{LEADER:22s} m1                            1.0000  1.0000"


def test_priced_columns_header(tmp_path: Path) -> None:
    """The priced header appends margin, interval and verdict to the plain one."""
    assert _two_columns(tmp_path, "m2").header() == PIN_AGAINST[1]


def test_priced_columns_row(tmp_path: Path) -> None:
    """A margin whose interval sits above zero clears; a negative one does not."""
    columns = _two_columns(tmp_path, "m2")
    assert columns.row(LEADER) == PIN_AGAINST[2].replace("m0", "m1")
    assert columns.row(LAGGARD).endswith("  -1.0000 [ -1.0000, -1.0000]  no")


def test_summary(tmp_path: Path) -> None:
    """One member's mean over its twelve findings and the finding it reads worst."""
    assert _two_columns(tmp_path, None).summary("m1") == PIN_MEMBERS[1].replace("m0", "m1")


def _main_lines(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, *argv: str) -> list[str]:
    """Drive main with `argv` and return what it reported."""
    import logging  # noqa: PLC0415 - test-local

    from kneemri.evaluation import seat_profile  # noqa: PLC0415 - test-local

    monkeypatch.setattr("sys.argv", ["seat_profile", *argv])
    caplog.set_level(logging.INFO, logger=seat_profile.__name__)
    seat_profile.main()
    return [record.getMessage() for record in caplog.records if record.name == seat_profile.__name__]


def _members(directory: Path) -> list[str]:
    """Three members, each right on a different finding and backwards on the rest."""
    return [str(_member(directory, f"m{index}", TARGETS[index])) for index in range(3)]


def test_main_ranks_each_finding_and_each_member(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
):
    """Without a baseline the table carries no margin columns, and every member gets a summary row."""
    assert _main_lines(monkeypatch, caplog, "--members", *_members(tmp_path)) == PIN_PLAIN


def test_main_reads_a_holdouts_directory(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
):
    """A directory of records with nothing declined reads the same pool the member list names."""
    _members(tmp_path)
    declines = tmp_path / "declines"
    declines.write_text("", encoding="utf-8")
    assert _main_lines(monkeypatch, caplog, "--holdouts", str(tmp_path), "--declines", str(declines)) == PIN_PLAIN


def test_main_prices_each_leader_against_a_baseline(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
):
    """--against adds the margin, its interval and the verdict, with the leader drawn from the rest of the pool."""
    lines = _main_lines(monkeypatch, caplog, "--members", *_members(tmp_path), "--against", "m2", "--midpoint")
    assert lines == PIN_AGAINST
