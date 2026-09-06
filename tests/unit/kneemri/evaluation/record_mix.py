"""Mixing members from their recorded holdout predictions, over records this test writes itself."""

from pathlib import Path

import numpy as np
import pytest

from kneemri.evaluation.member_curve import MemberCurve
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import MixReading, RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.runs.holdout_record import FOREIGN, HoldoutRecord

LABELS = 12
RNG = np.random.default_rng(2026)


def _record(path: Path, studies: list[str], *, shift: float = 0.0, truth: np.ndarray | None = None) -> Path:
    """One run's record: predictions that follow the truth by `shift`, so a mix has something to gain.

    The CHECKPOINT is written too, empty but present, because a member is weights plus a reading of them
    — a record whose `.pt` never landed is a dead run, and the pool now drops it.
    """
    answers = truth if truth is not None else (RNG.random((len(studies), LABELS)) > 0.5).astype(np.float32)
    predictions = answers * shift + RNG.random((len(studies), LABELS))
    HoldoutRecord(path).write(studies, predictions.astype(np.float64), answers.astype(np.float32), epoch=6)
    path.write_bytes(b"")
    return path


def test_readings(tmp_path: Path):
    """A record that exists is read; a checkpoint without one is named and skipped, not raised on.

    Held after the first read, since `shared` and `per_member` both ask: an uncached read would name
    the one missing member three times, which reads as three missing members.
    """
    studies = [f"s{index}" for index in range(8)]
    _record(tmp_path / "a.pt", studies)
    (tmp_path / "never_trained.pt").write_bytes(b"")
    mix = RecordMix([tmp_path / "a.pt", tmp_path / "never_trained.pt"])

    found = mix.readings()

    assert [name for name, _ in found] == ["a.pt"], "the missing record is skipped, not fatal"
    assert found[0][1].studies == studies
    assert mix.readings() is found, "read once, not once per caller"


def test_readings_refuses_a_path_that_is_not_there(tmp_path: Path):
    """A checkpoint with no record is skipped; a path with no checkpoint EITHER is a wrong root.

    Both once printed the same skip line, so one mistyped name in nine scored an eight-member ensemble
    and reported it as nine.
    """
    _record(tmp_path / "a.pt", [f"s{index}" for index in range(8)])

    with pytest.raises(FileNotFoundError, match="check the root"):
        RecordMix([tmp_path / "a.pt", tmp_path / "elsewhere" / "b.pt"]).readings()


def test_readings_drops_a_record_whose_run_never_finished(tmp_path: Path):
    """A record with no checkpoint is a DEAD run — the array is real, the weights are gone.

    `Fitting` writes the record on every improving epoch and the checkpoint only after it returns, so
    this is the shape a crash leaves behind. Nine of ninety-one records under the processed root are in
    it, and a glob of that directory would seat a tenth of its pool on members nothing can package.
    """
    studies = [f"s{index}" for index in range(8)]
    _record(tmp_path / "alive.pt", studies)
    _record(tmp_path / "died.pt", studies).unlink()
    mix = RecordMix([tmp_path / "alive.pt", tmp_path / "died.pt"])

    assert [name for name, _ in mix.readings()] == ["alive.pt"], "a run that never ended is not a member"
    assert mix.dropped() == ["died.pt"], "and it is NAMED, so the pool size the caller asked for is checkable"


def test_readings_drops_a_foreign_record_unless_asked(tmp_path: Path):
    """Someone else's weights through someone else's decode are refused by default, and named.

    A foreign record is decoded onto OUR studies under OUR label table, which is what makes it mixable
    here and what makes it contaminated. A glob of the processed root that seated the six of them read
    an honest peak of 0.8994 at six seats against the 0.8754 over the members we trained.
    """
    studies = [f"s{index}" for index in range(8)]
    _record(tmp_path / "ours.pt", studies)
    theirs = _record(tmp_path / "theirs.pt", studies)
    reading = HoldoutRecord(theirs).read()
    assert reading is not None
    HoldoutRecord(theirs).write(reading.studies, reading.predictions, reading.truth, reading.epoch, source=FOREIGN)

    mix = RecordMix([tmp_path / "ours.pt", theirs])

    assert [name for name, _ in mix.readings()] == ["ours.pt"], "guilt by default, as foreign_gate states it"

    asked = RecordMix([tmp_path / "ours.pt", theirs], foreign=True)

    assert [name for name, _ in asked.readings()] == ["ours.pt", "theirs.pt"], "a shipped entry is reproducible"


def test_foreign_dropped(tmp_path: Path):
    """What was refused is NAMED, so the pool that ran is checkable against the paths handed in."""
    studies = [f"s{index}" for index in range(8)]
    _record(tmp_path / "ours.pt", studies)
    theirs = _record(tmp_path / "theirs.pt", studies)
    reading = HoldoutRecord(theirs).read()
    assert reading is not None
    HoldoutRecord(theirs).write(reading.studies, reading.predictions, reading.truth, reading.epoch, source=FOREIGN)

    assert RecordMix([tmp_path / "ours.pt"]).foreign_dropped() == [], "a clean pool refuses nothing"
    assert RecordMix([tmp_path / "ours.pt", theirs]).foreign_dropped() == ["theirs.pt"]
    assert RecordMix([tmp_path / "ours.pt", theirs], foreign=True).foreign_dropped() == [], "nothing was refused"


def test_dropped(tmp_path: Path):
    """A healthy pool drops nothing, so the warning means what it says on the day it fires."""
    studies = [f"s{index}" for index in range(8)]
    _record(tmp_path / "a.pt", studies)
    assert RecordMix([tmp_path / "a.pt"]).dropped() == []

    _record(tmp_path / "died.pt", studies).unlink()
    assert RecordMix([tmp_path / "a.pt", tmp_path / "died.pt"]).dropped() == ["died.pt"]


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

    (tmp_path / "untrained.pt").write_bytes(b"")
    with pytest.raises(ValueError, match="no member carries a holdout record"):
        RecordMix([tmp_path / "untrained.pt"]).shared()


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


def test_per_member_holds_its_three_arrays_together(tmp_path: Path):
    """The join hands back one Pool rather than three lists a caller could reorder against each other."""
    studies = [f"s{index}" for index in range(5)]
    _record(tmp_path / "a.pt", studies)
    mixed = RecordMix([tmp_path / "a.pt"]).per_member()

    assert isinstance(mixed, Pool)
    assert len(mixed.names) == len(mixed.predictions)


def test_lines(tmp_path: Path):
    """A named roster wins the dispatch: nobody pins the members that ship and means `search subsets`."""
    studies = [f"s{index}" for index in range(40)]
    truth = (RNG.random((40, LABELS)) > 0.5).astype(np.float32)
    for name in ("shipped.pt", "candidate.pt"):
        _record(tmp_path / name, studies, shift=2.0, truth=truth)
    mixed = RecordMix([tmp_path / "shipped.pt", tmp_path / "candidate.pt"]).per_member()
    curve = MemberCurve(LabelScores())

    priced = MixReading(mixed, [tmp_path / "shipped.pt"]).lines(
        curve, "greedy", paired=False, labels=False, routing=False
    )
    built = MixReading(mixed, []).lines(curve, "greedy", paired=False, labels=False, routing=False)
    opened = MixReading(mixed, [tmp_path / "shipped.pt"]).lines(
        curve, "greedy", paired=False, labels=True, routing=False
    )
    routed = MixReading(mixed, [tmp_path / "shipped.pt"]).lines(
        curve, "greedy", paired=False, labels=True, routing=True
    )

    assert priced[0].startswith("roster of 1:"), "the roster is honoured even with --seats greedy also passed"
    assert built[0].startswith("greedy 1:"), "with no roster the same call builds an ensemble instead"
    assert opened[0].startswith("mix of 1:"), "--labels opens the roster up rather than pricing against it"
    assert routed[0].startswith("top- 1 per label:"), "--routing wins the dispatch over every other reading"
    assert len(routed) == len(mixed.names), "one row per k, over the WHOLE pool rather than the roster"


def test_narrow(tmp_path: Path):
    """The largest agreeing pool survives; a stub split and a foreign ruler are dropped and NAMED.

    Both offenders at once, because the bisect this replaces converged on neither: `per_member` compares
    every truth against the FIRST record's, so dropping whatever gets named walks toward whichever ruler
    sorts first rather than toward the majority.
    """
    studies = [f"s{index}" for index in range(20)]
    truth = (RNG.random((len(studies), LABELS)) > 0.5).astype(np.float32)
    for name in ("a.pt", "b.pt", "c.pt"):
        _record(tmp_path / name, studies, truth=truth)
    _record(tmp_path / "smoke.pt", studies[:2], truth=truth[:2])
    _record(tmp_path / "other_ruler.pt", studies, truth=1.0 - truth)
    mix = RecordMix(sorted(tmp_path.glob("*.pt")))

    dropped = mix.narrow()

    assert sorted(dropped) == ["other_ruler.pt", "smoke.pt"], "both offenders named, neither silently kept"
    assert [name for name, _ in mix.readings()] == ["a.pt", "b.pt", "c.pt"], "the majority ruler is the pool"


def test_narrow_refuses_a_pool_where_nothing_carries_a_record(tmp_path: Path):
    """A pool of whole-corpus members is refused by name, not crashed on inside the widest-split read.

    `--fold -1` writes no `.holdout.npz`, so every reading is skipped and `_on_widest_split` took
    `max()` over an empty list. That reads as a tool defect rather than as the true answer, which is
    that a whole-corpus member has nothing to be narrowed against.
    """
    for name in ("wc_one.pt", "wc_two.pt"):
        (tmp_path / name).write_bytes(b"")
    mix = RecordMix(sorted(tmp_path.glob("*.pt")))

    with pytest.raises(ValueError, match="carries a holdout array"):
        mix.narrow()


def test_narrow_keeps_the_majority_even_when_the_foreign_ruler_sorts_first(tmp_path: Path):
    """Two agreeing members beat one that sorts ahead of them — the failure the hand bisect walked into."""
    studies = [f"s{index}" for index in range(20)]
    truth = (RNG.random((len(studies), LABELS)) > 0.5).astype(np.float32)
    _record(tmp_path / "aaa_alone.pt", studies, truth=1.0 - truth)
    _record(tmp_path / "zzz_one.pt", studies, truth=truth)
    _record(tmp_path / "zzz_two.pt", studies, truth=truth)
    mix = RecordMix(sorted(tmp_path.glob("*.pt")))

    assert mix.narrow() == ["aaa_alone.pt"], "the first-sorting ruler loses to the bigger group"
    assert [name for name, _ in mix.readings()] == ["zzz_one.pt", "zzz_two.pt"]


def test_narrowed(tmp_path: Path):
    """What was dropped stays askable after the fact — the pool that remains IS the denominator."""
    studies = [f"s{index}" for index in range(20)]
    truth = (RNG.random((len(studies), LABELS)) > 0.5).astype(np.float32)
    _record(tmp_path / "a.pt", studies, truth=truth)
    _record(tmp_path / "b.pt", studies, truth=truth)
    _record(tmp_path / "smoke.pt", studies[:2], truth=truth[:2])
    mix = RecordMix(sorted(tmp_path.glob("*.pt")))

    assert mix.narrowed() == [], "nothing is dropped until narrowing is asked for"
    dropped = mix.narrow()

    assert mix.narrowed() == dropped == ["smoke.pt"]


def test_readings_seats_a_foreign_record_that_has_no_checkpoint_of_ours(tmp_path: Path):
    """A vendor member IS its record: there was never a checkpoint of ours to be missing.

    The dead-run rule asked first, so `dinov3f0_solo` — someone else's fold served through their own
    kernel code onto our holdout — was dropped as a crashed run of ours, and `--foreign` could not
    seat it however loudly it was passed. The two refusals answer different questions and only the
    dead-run one is about a checkpoint we would have written.
    """
    studies = [f"s{index}" for index in range(8)]
    _record(tmp_path / "ours.pt", studies)
    theirs = _record(tmp_path / "theirs.pt", studies)
    reading = HoldoutRecord(theirs).read()
    assert reading is not None
    HoldoutRecord(theirs).write(reading.studies, reading.predictions, reading.truth, reading.epoch, source=FOREIGN)
    theirs.unlink()

    seated = RecordMix([tmp_path / "ours.pt", theirs], foreign=True)

    assert [name for name, _ in seated.readings()] == ["ours.pt", "theirs.pt"]
    assert seated.dropped() == [], "a vendor record is not a dead run"
