"""A candidate is priced against the champion the pin names, not against a roster typed by hand."""

from pathlib import Path

import numpy as np

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
    """One member's record: predictions that follow the truth by `shift`."""
    studies = [f"s{index}" for index in range(STUDIES)]
    predictions = truth * shift + RNG.random((STUDIES, LABELS))
    HoldoutRecord(path).write(studies, predictions.astype(np.float64), truth.astype(np.float32), epoch=6)


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

    assert lines[0].startswith("roster of 2:"), "the base every delta is against is stated"
    assert "+candidate.pt" in lines[1], "the candidate is named with its marginal contribution"
    assert "shipped" not in lines[1], "a roster member is never also priced as a candidate"


def test_named(tmp_path: Path):
    """A bare name is a member of the processed root, which is how every arm in this repo is spelled."""
    seats = SeatValue(_champion(), tmp_path)

    assert seats.named(Path("band_span_s2026.pt")) == tmp_path / "band_span_s2026.pt"
    assert seats.named(tmp_path / "x.pt") == tmp_path / "x.pt", "a path given in full is left alone"


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
