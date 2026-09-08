"""A candidate is priced against the champion the pin names, not against a roster typed by hand."""

import json
from pathlib import Path

import numpy as np

from core.findings import TARGETS
from kneemri.evaluation.ledger75 import MIDPOINT_COLUMNS, RECUT
from kneemri.evaluation.scores import MIDPOINT
from kneemri.kaggle.champion_check import Champion
from kneemri.kaggle.seat_value import SeatValue
from kneemri.runs.holdout_record import HoldoutRecord

LABELS = 12
STUDIES = 40
RNG = np.random.default_rng(2026)


def _champion() -> Champion:
    """A pinned entry, whose members are BARE names — the pin stores `band_dense_s2026`, not a filename."""
    return Champion(entry="test", kernel="k", board=0.9, members=("shipped", "other"), kit="abc123")


def _member(path: Path, truth: np.ndarray, shift: float) -> None:
    """One member's record, and the checkpoint it was read from — a record alone is a dead run."""
    studies = [f"s{index}" for index in range(STUDIES)]
    predictions = truth * shift + RNG.random((STUDIES, LABELS))
    HoldoutRecord(path).write(studies, predictions.astype(np.float64), truth.astype(np.float32), epoch=6)
    path.write_bytes(b"")


def _board(tmp_path: Path) -> Path:
    """A board holding one fully scoreable entry, one that is not, and two rows that are not entries.

    The last two are the reason `entries` cannot simply map `from_board` over the file: the real record
    opens with prose notes carrying no `id`, and 31 of its rows record no members at all.
    """
    rows = [
        {"note": "the record's own preamble, which is not an entry"},
        {"id": "whole", "kernel": "k", "board": 0.933, "members_named": ["shipped", "spare"], "scope": "ensemble"},
        {"id": "dark", "kernel": "k", "board": 0.938, "members_named": ["shipped", "ghost"], "scope": "ensemble"},
        {"id": "ruined", "kernel": "k", "board": 0.94, "scope": "ensemble"},
    ]
    path = tmp_path / "board.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_roster(tmp_path: Path):
    """The members come from the pin, so an arm cannot price itself against a stale hand-typed four."""
    seats = SeatValue(_champion(), tmp_path)

    roster = seats.roster()

    assert [path.name for path in roster] == ["shipped.pt", "other.pt"], "each pinned name becomes a checkpoint"
    assert all(path.parent == tmp_path for path in roster), "resolved under the processed root, not the repo"


def test_report(tmp_path: Path):
    """The report states the roster's own score and what the candidate adds to it."""
    truth = (RNG.random((STUDIES, LABELS)) > 0.5).astype(np.float32)
    for name, shift in (("shipped.pt", 2.0), ("other.pt", 2.0), ("candidate.pt", 2.0)):
        _member(tmp_path / name, truth, shift)

    lines = SeatValue(_champion(), tmp_path).report([tmp_path / "candidate.pt"])

    assert lines[0].startswith("priced against all 2 champion members"), "the denominator leads, even whole"
    assert lines[0].endswith("macro read on the cut0.5 ruler"), "and it names the ruler it was read on"
    assert lines[1].startswith("roster of 2:"), "the base every delta is against is stated"
    assert "+candidate.pt" in lines[2], "the candidate is named with its marginal contribution"
    assert "shipped" not in lines[2], "a roster member is never also priced as a candidate"


def test_scoreable(tmp_path: Path):
    """A member that held nothing out is separated from the ones that can be priced, not silently dropped."""
    truth = (RNG.random((STUDIES, LABELS)) > 0.5).astype(np.float32)
    _member(tmp_path / "shipped.pt", truth, 2.0)
    seats = SeatValue(_champion(), tmp_path)

    priced, unscoreable = seats.scoreable(seats.roster())

    assert [path.name for path in priced] == ["shipped.pt"]
    assert [path.name for path in unscoreable] == ["other.pt"], "a fold -1 member is named, not discarded quietly"


def test_report_states_the_subset_it_priced_against(tmp_path: Path):
    """With an unscoreable member in the roster the report still prices, and says what it priced over."""
    truth = (RNG.random((STUDIES, LABELS)) > 0.5).astype(np.float32)
    for name in ("shipped.pt", "other.pt", "candidate.pt"):
        _member(tmp_path / name, truth, 2.0)
    champion = Champion(entry="test", kernel="k", board=0.9, members=("shipped", "other", "held"), kit="abc123")

    lines = SeatValue(champion, tmp_path).report([tmp_path / "candidate.pt"])

    assert lines[0].startswith("priced against 2 of 3 champion members; held unscoreable (fold -1)")
    assert lines[1].startswith("roster of 2:"), "the proxy roster is the scoreable subset, and it is labelled"


def test_named(tmp_path: Path):
    """A bare name is a member of the processed root, which is how every arm in this repo is spelled."""
    seats = SeatValue(_champion(), tmp_path)

    assert seats.named(Path("band_span_s2026.pt")) == tmp_path / "band_span_s2026.pt"
    assert seats.named(tmp_path / "x.pt") == tmp_path / "x.pt", "a path given in full is left alone"


def test_unpriceable(tmp_path: Path):
    """A roster of one has no mix to be marginal to, so the refusal carries its own denominator."""
    truth = (RNG.random((STUDIES, LABELS)) > 0.5).astype(np.float32)
    _member(tmp_path / "shipped.pt", truth, 2.0)
    _member(tmp_path / "spare.pt", truth, 2.0)
    seats = SeatValue(_champion(), tmp_path, _board(tmp_path))
    roster, unscoreable = seats.scoreable(seats.roster())

    lines = seats.unpriceable(roster, unscoreable)

    assert lines[0] == (
        "NO SEAT VALUE against test: 1 of 2 members carry a holdout record, and pricing a marginal "
        "contribution needs at least 2"
    )
    assert "other" in lines[1], "the members that can never be scored are named, not counted"
    assert "--against" in lines[2], "a refusal a reader cannot act on is half a refusal"


def test_report_refuses_a_roster_too_small_to_price(tmp_path: Path):
    """The refusal reaches the arm log in place of a number, rather than beside one."""
    truth = (RNG.random((STUDIES, LABELS)) > 0.5).astype(np.float32)
    _member(tmp_path / "shipped.pt", truth, 2.0)
    _member(tmp_path / "candidate.pt", truth, 2.0)

    lines = SeatValue(_champion(), tmp_path, _board(tmp_path)).report([tmp_path / "candidate.pt"])

    assert lines[0].startswith("NO SEAT VALUE against test:")
    assert not any("candidate" in line for line in lines), "no candidate is priced against a roster of one"


def test_report_prices_a_candidate_given_by_name(tmp_path: Path):
    """The equivalence class the existing test missed by always passing a full path.

    A candidate that does not resolve carries no holdout record, and a missing record is *skipped* with
    one log line rather than raised on — so an unpriced candidate reads exactly like one priced at
    nothing, and the roster figure beside it still looks like a complete answer.
    """
    truth = (RNG.random((STUDIES, LABELS)) > 0.5).astype(np.float32)
    for name, shift in (("shipped.pt", 2.0), ("other.pt", 2.0), ("candidate.pt", 2.0)):
        _member(tmp_path / name, truth, shift)

    lines = SeatValue(_champion(), tmp_path).report([Path("candidate.pt")])

    assert any("+candidate.pt" in line for line in lines), "a candidate named without a directory is priced"


def test_ruler():
    """Both rulers resolve by name, and `ledger75` is the MIXED cut rather than a scalar 0.75.

    The distinction is the whole reason this exists: `ledger75` recuts ten columns and leaves Medial
    Meniscus and Baker's at the midpoint, so a scalar `LabelScores(0.75)` is a different instrument
    wearing the same number.
    """
    midpoint = SeatValue.ruler("midpoint")
    recut = SeatValue.ruler("ledger75")

    assert midpoint.name() == "cut0.5"
    assert recut.name() == "ledger75"
    assert set(recut.column_cuts) == set(MIDPOINT_COLUMNS), "exactly the columns ledger75 leaves alone"
    assert all(cut == MIDPOINT for cut in recut.column_cuts.values())
    assert recut.cut_for(next(target for target in TARGETS if target not in MIDPOINT_COLUMNS)) == RECUT
