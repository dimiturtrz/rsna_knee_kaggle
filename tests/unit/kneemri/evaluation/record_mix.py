"""Mixing members from their recorded holdout predictions, over records this test writes itself."""

from pathlib import Path

import numpy as np
import pytest

from kneemri.evaluation.record_mix import MixedRecords, RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.runs.holdout_record import HoldoutRecord

LABELS = 12
RNG = np.random.default_rng(2026)


def _record(path: Path, studies: list[str], *, shift: float = 0.0, truth: np.ndarray | None = None) -> Path:
    """One run's record: predictions that follow the truth by `shift`, so a mix has something to gain."""
    answers = truth if truth is not None else (RNG.random((len(studies), LABELS)) > 0.5).astype(np.float32)
    predictions = answers * shift + RNG.random((len(studies), LABELS))
    HoldoutRecord(path).write(studies, predictions.astype(np.float64), answers.astype(np.float32), epoch=6)
    return path


def test_readings(tmp_path: Path):
    """A record that exists is read; a checkpoint without one is named and skipped, not raised on.

    Held after the first read, since `shared` and `per_member` both ask: an uncached read would name
    the one missing member three times, which reads as three missing members.
    """
    studies = [f"s{index}" for index in range(8)]
    _record(tmp_path / "a.pt", studies)
    mix = RecordMix([tmp_path / "a.pt", tmp_path / "never_trained.pt"])

    found = mix.readings()

    assert [name for name, _ in found] == ["a.pt"], "the missing record is skipped, not fatal"
    assert found[0][1].studies == studies
    assert mix.readings() is found, "read once, not once per caller"


def test_shared(tmp_path: Path):
    """The intersection, in the FIRST member's order — and no intersection is a refusal."""
    order = [f"s{index}" for index in reversed(range(20))]
    truth = (RNG.random((20, LABELS)) > 0.5).astype(np.float32)
    _record(tmp_path / "a.pt", order, truth=truth)
    _record(tmp_path / "b.pt", order[::-1][:19], truth=truth[::-1][:19])

    shared = RecordMix([tmp_path / "a.pt", tmp_path / "b.pt"]).shared()
    assert shared == order[1:], "the first member's order, minus what b never held"

    _record(tmp_path / "elsewhere.pt", [f"e{index}" for index in range(20)], truth=truth)
    with pytest.raises(ValueError, match="different folds or splits"):
        RecordMix([tmp_path / "a.pt", tmp_path / "elsewhere.pt"]).shared()

    with pytest.raises(ValueError, match="no member carries a holdout record"):
        RecordMix([tmp_path / "absent.pt"]).shared()


def test_shared_refuses_a_narrow_member(tmp_path: Path):
    """A smoke run beside real members is named, not silently made the denominator.

    The failure this guards is the one that reports a plausible number: intersecting twenty-three
    909-study members with an 11-study smoke record leaves eleven, and a greedy ordering over eleven
    studies prints exactly like one over 909.
    """
    truth = (RNG.random((40, LABELS)) > 0.5).astype(np.float32)
    studies = [f"s{index}" for index in range(40)]
    _record(tmp_path / "real.pt", studies, truth=truth)
    _record(tmp_path / "smoke.pt", studies[:4], truth=truth[:4])

    with pytest.raises(ValueError, match="smoke.pt"):
        RecordMix([tmp_path / "real.pt", tmp_path / "smoke.pt"]).shared()

    _record(tmp_path / "one_short.pt", studies[:39], truth=truth[:39])
    assert len(RecordMix([tmp_path / "real.pt", tmp_path / "one_short.pt"]).shared()) == 39, (
        "a member missing a handful of studies is still a mix"
    )


def test_per_member(tmp_path: Path):
    """Members are aligned onto the shared studies, and a different label table is refused.

    The refusal is the one failure a mix cannot detect for itself: averaging two rulers succeeds
    arithmetically and reports a number that answers no question.
    """
    studies = [f"s{index}" for index in range(10)]
    truth = (RNG.random((10, LABELS)) > 0.5).astype(np.float32)
    _record(tmp_path / "a.pt", studies, truth=truth)
    _record(tmp_path / "b.pt", list(reversed(studies)), truth=truth[::-1])

    mixed = RecordMix([tmp_path / "a.pt", tmp_path / "b.pt"]).per_member()

    assert mixed.names == ["a.pt", "b.pt"]
    assert all(member.shape == (10, LABELS) for member in mixed.predictions)
    assert np.array_equal(mixed.truth, truth), "aligned to the first member's order"

    _record(tmp_path / "other_ruler.pt", studies, truth=1.0 - truth)
    with pytest.raises(ValueError, match="different label table"):
        RecordMix([tmp_path / "a.pt", tmp_path / "other_ruler.pt"]).per_member()


def test_alone(tmp_path: Path):
    """Every member's own score is reported beside its name, which is what a mix is read against."""
    studies = [f"s{index}" for index in range(40)]
    truth = (RNG.random((40, LABELS)) > 0.5).astype(np.float32)
    _record(tmp_path / "strong.pt", studies, shift=3.0, truth=truth)
    _record(tmp_path / "weak.pt", studies, shift=0.2, truth=truth)

    lines = RecordMix([tmp_path / "strong.pt", tmp_path / "weak.pt"]).per_member().alone(LabelScores())

    assert [line.split(" alone")[0] for line in lines] == ["strong.pt", "weak.pt"]
    assert float(lines[0].split()[-1]) > float(lines[1].split()[-1]), "the member that follows truth scores higher"


def test_mixed_records_holds_its_three_arrays_together(tmp_path: Path):
    """The container refuses to be built from mismatched lists, which a bare tuple could not."""
    studies = [f"s{index}" for index in range(5)]
    _record(tmp_path / "a.pt", studies)
    mixed = RecordMix([tmp_path / "a.pt"]).per_member()

    assert isinstance(mixed, MixedRecords)
    assert len(mixed.names) == len(mixed.predictions)
