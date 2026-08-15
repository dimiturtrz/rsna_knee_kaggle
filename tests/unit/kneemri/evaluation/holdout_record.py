"""The per-study predictions file: what it stores, and the pairing it refuses."""

from pathlib import Path

import numpy as np
import pytest

from kneemri.evaluation.holdout_record import DERIVED, HoldoutReading, HoldoutRecord

LABELS = 12


def _reading(studies: list[str], offset: float = 0.0, truth: np.ndarray | None = None) -> HoldoutReading:
    positions = np.arange(len(studies), dtype=np.float64).reshape(-1, 1)
    scores = np.zeros((len(studies), LABELS), dtype=np.float32) if truth is None else truth
    return HoldoutReading(
        studies=studies,
        predictions=np.tile(positions / 10 + offset, (1, LABELS)),
        truth=scores,
        epoch=3,
    )


def test_read(tmp_path: Path):
    """Sixty-five runs already on disk predate this, and they must say so rather than raise."""
    assert HoldoutRecord(tmp_path / "member.pt").read() is None


def test_exists(tmp_path: Path):
    """Asked of the record rather than of a path, and the suffix does not accumulate.

    Both spellings are natural: the trainer passes its checkpoint path, an analysis passes the file it
    can see on disk. Appending blindly produced a `.holdout.holdout.npz` that read as a run which
    predicted nothing.
    """
    record = HoldoutRecord(tmp_path / "member.pt")
    assert not record.exists()

    record.write(["a"], np.zeros((1, LABELS)), np.zeros((1, LABELS), np.float32), 0)

    assert record.exists()
    assert HoldoutRecord(record.path).path == record.path


def test_write(tmp_path: Path):
    """Predictions survive the round trip under the NAMES of the studies they belong to.

    Positions mean nothing across runs — a different fold, a rebuilt corpus or a dropped study all
    reindex the array while leaving its shape plausible.
    """
    record = HoldoutRecord(tmp_path / "member.pt")
    predictions = np.linspace(0, 1, 3 * LABELS).reshape(3, LABELS)
    truth = np.zeros((3, LABELS), dtype=np.float32)

    studies = HoldoutRecord.named(["a", "b", "c", "d"], np.array([3, 1, 0], dtype=np.int64))
    record.write(studies, predictions, truth, epoch=5, source=DERIVED)
    stored = record.read()

    assert stored is not None
    assert stored.studies == ["d", "b", "a"], "the holdout's own order, named"
    assert stored.epoch == 5
    assert stored.predictions == pytest.approx(predictions, abs=1e-6)
    assert stored.source == DERIVED, "a back-fill is a re-derivation, and only the file can say so"


def test_write_commits_by_rename_so_a_kill_cannot_truncate_a_record(tmp_path: Path):
    """The file is re-written by a back-fill over a run that already has one, and a kill lands mid-write.

    `read` answers None only for a file that is ABSENT; a truncated archive raises instead, so an
    interrupted write used to take every later comparison of that run down with it. Committing by
    rename means the path either holds the old record or the new one.
    """
    record = HoldoutRecord(tmp_path / "member.pt")
    truth = np.zeros((1, LABELS), dtype=np.float32)
    record.write(["a"], np.zeros((1, LABELS)), truth, epoch=1)
    record.write(["a"], np.ones((1, LABELS)), truth, epoch=2)

    stored = record.read()

    assert stored is not None
    assert stored.epoch == 2, "the second write replaced the first whole"
    assert not list(tmp_path.glob("*.pending.npz")), "nothing is left half-committed beside the checkpoint"


def test_named():
    """Positions belong to one corpus build; the file has to survive the next one, so it stores names."""
    assert HoldoutRecord.named(["a", "b", "c", "d"], np.array([3, 1], dtype=np.int64)) == ["d", "b"]


def test_aligned():
    """Two arms rarely hold out the same set, and the overlap is what a paired interval runs on."""
    mine, theirs, truth = _reading(["a", "b", "c"]).aligned(_reading(["b", "c", "z"], offset=1.0))

    assert mine.shape == (2, LABELS)
    assert truth.shape == (2, LABELS)
    assert mine[:, 0] == pytest.approx([0.1, 0.2]), "b and c, at MY positions"
    assert theirs[:, 0] == pytest.approx([1.0, 1.1]), "b and c, at THEIRS"


def test_aligned_refuses_two_runs_scored_by_different_rulers():
    """The one failure a paired interval cannot detect for itself: the arithmetic succeeds.

    This project has three label tables and has already re-run its ablations after one moved. Pairing
    across that boundary is arithmetic on two different questions and comes out tighter than either arm
    has earned.
    """
    other = np.ones((2, LABELS), dtype=np.float32)

    with pytest.raises(ValueError, match="different label tables"):
        _reading(["a", "b"]).aligned(_reading(["a", "b"], truth=other))


def test_aligned_refuses_runs_with_nothing_in_common():
    """An empty overlap is a mismatched split, not an interval of width zero."""
    with pytest.raises(ValueError, match="no study is held out by both"):
        _reading(["a", "b"]).aligned(_reading(["y", "z"]))
