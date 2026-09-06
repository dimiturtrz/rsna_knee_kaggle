"""A scoreable member is either packaged or declined with a reason, and anything else is a finding."""

from pathlib import Path

import numpy as np
import torch

from kneemri.inference.members import Member
from kneemri.kaggle.roster_check import MemberDeclines, RosterDrift
from kneemri.runs.holdout_record import HoldoutRecord

LABELS = 3
STUDIES = 8
SIZE = 224
SLICES = 12
RNG = np.random.default_rng(2026)

SHIPPED = "shipped_s2026"
"""The one member the fixture roster names, and the only one that should ever read as packaged."""
SMOKE = "run_smoke"
"""Scoreable and deliberately not shipped — the case the declines file exists to record."""
BARE = "bare_s2026"
"""Declined by name alone: memory moved one file over, which is still a finding."""
FORGOTTEN = "forgotten_s2026"
"""Scoreable, in neither file. The defect knee-0dtv was filed for."""


def _member(root: Path, name: str, *, scoreable: bool) -> None:
    """A checkpoint, with the holdout predictions that make it priceable or without them.

    A REAL torch payload rather than an empty file, because `unreadable` asks each packaged member which
    pixels it read and a member is the only thing that can answer. The config carries the fields that
    name a cache and nothing else.
    """
    path = root / f"{name}.pt"
    torch.save({"config": {"img": SIZE, "slices": SLICES, "slots": ["sag"]}}, path)
    if scoreable:
        truth = (RNG.random((STUDIES, LABELS)) > 0.5).astype(np.float32)
        HoldoutRecord(path).write(
            [f"s{index}" for index in range(STUDIES)], RNG.random((STUDIES, LABELS)), truth, epoch=1
        )


def _drift(tmp_path: Path, roster_text: str, declines_text: str) -> RosterDrift:
    """A processed root holding four scoreable members and one that never wrote a record."""
    for name in (SHIPPED, SMOKE, BARE, FORGOTTEN):
        _member(tmp_path, name, scoreable=True)
    _member(tmp_path, "crashed_s2026", scoreable=False)
    roster = tmp_path / "members.roster"
    declines = tmp_path / "members.declines"
    roster.write_text(roster_text, encoding="utf-8")
    declines.write_text(declines_text, encoding="utf-8")
    return RosterDrift(tmp_path, [roster], declines)


def _clean(tmp_path: Path) -> RosterDrift:
    """The state the check exists to hold: every scoreable member accounted for on one side or the other.

    Its packaged member's pixels are written too — a clean roster is one whose members can still be READ,
    and leaving the arrays out would make every other test in this file assert against a broken tree.
    """
    drift = _drift(
        tmp_path,
        f"# what ships\n{SHIPPED}.pt   # the member\n",
        f"{SMOKE}.pt   # a throwaway smoke run\n{BARE}.pt  # scored under an older label table\n"
        f"{FORGOTTEN}.pt  # superseded\n",
    )
    _pixels(tmp_path)
    return drift


def test_checkpoints(tmp_path: Path):
    """Every `.pt` on disk is surveyed, scoreable or not — the denominator the report counts against."""
    drift = _clean(tmp_path)

    assert "crashed_s2026" in drift.checkpoints(), "a checkpoint with no record is still a checkpoint"


def test_scoreable(tmp_path: Path):
    """A checkpoint that wrote no holdout predictions cannot take a seat, so it is not this file's business."""
    drift = _clean(tmp_path)

    assert "crashed_s2026" not in drift.scoreable(), "it left nothing to price"


def test_read(tmp_path: Path):
    """A declines file that does not exist yet reads as no declines, not as a crash."""
    written = tmp_path / "members.declines"
    written.write_text("a.pt # smoke\n", encoding="utf-8")

    assert MemberDeclines.read(written).reasons() == {"a": "smoke"}, "the file on disk"
    assert MemberDeclines.read(tmp_path / "absent").reasons() == {}, "a repo before anything was declined"


def test_patterns(tmp_path: Path):
    """The roster's lines are read through the packager's own reader, so both see one spelling."""
    drift = _drift(tmp_path, f"# what ships\n{SHIPPED}.pt   # the member\n", "")

    assert drift.patterns() == [f"{SHIPPED}.pt"], "the comment and the blank are not patterns"


def test_patterns_spans_every_shelf_and_skips_one_that_is_not_there(tmp_path: Path):
    """Members split across two packages are documented by the same rule, and a missing shelf is silent.

    There are two shelves because the shipped package is ~10 GB and a member added to it can only reach
    the board by re-pushing all of it. If this check read only the first, every member on the second
    would report as undocumented — a permanently red gate, which is how a real refusal stops being read.
    A shelf that does not exist is not a complaint: the second package is optional.
    """
    for name in (SHIPPED, SMOKE, BARE, FORGOTTEN):
        _member(tmp_path, name, scoreable=True)
    first = tmp_path / "members.roster"
    second = tmp_path / "members2.roster"
    first.write_text(f"{SHIPPED}.pt\n", encoding="utf-8")
    second.write_text(f"{BARE}.pt\n", encoding="utf-8")
    declines = tmp_path / "members.declines"
    declines.write_text("", encoding="utf-8")

    spanning = RosterDrift(tmp_path, [first, second], declines)
    assert spanning.packaged() == {SHIPPED, BARE}, "both shelves ship"

    alone = RosterDrift(tmp_path, [first, tmp_path / "absent.roster"], declines)
    assert alone.packaged() == {SHIPPED}, "a shelf that does not exist is skipped, not read as empty prose"


def test_reasons(tmp_path: Path):
    """The declines file keeps the prose, which is the whole point of it existing."""
    declines = MemberDeclines("a.pt  # a smoke run\nb.pt\n\n# a whole-line comment\n")

    assert declines.reasons() == {"a": "a smoke run", "b": ""}, "a whole-line comment names no member"


def test_reasons_folds_a_wrapped_argument_into_the_entry_above_it(tmp_path: Path):
    """A decline is an argument, and an argument wraps. Each wrapped line is prose, never a member."""
    declines = MemberDeclines("a.pt  # declined because the delta is under the floor\nand the sign is wrong.\nb.pt\n")

    assert declines.reasons() == {
        "a": "declined because the delta is under the floor and the sign is wrong.",
        "b": "",
    }, "the continuation joins its entry, and the bare name after it is still a finding"


def test_packaged(tmp_path: Path):
    """A roster line reaches its member whether or not it is written as a glob."""
    drift = _drift(tmp_path, "shipped_*.pt\n", "")

    assert drift.packaged() == {SHIPPED}, "the glob reaches exactly one scoreable member"


def test_undocumented(tmp_path: Path):
    """A scoreable member in neither file is named, and one in either file is not."""
    drift = _drift(tmp_path, f"{SHIPPED}.pt\n", f"{SMOKE}.pt # smoke\n{BARE}.pt # stale labels\n")

    assert drift.undocumented() == [FORGOTTEN], "three are accounted for, one is not"


def test_reasonless(tmp_path: Path):
    """A bare decline records that somebody decided, not what they decided."""
    drift = _drift(tmp_path, f"{SHIPPED}.pt\n", f"{SMOKE}.pt\n{BARE}.pt # stale labels\n{FORGOTTEN}.pt # gone\n")

    assert drift.reasonless() == [SMOKE], "only the one with no prose"


def test_contradictory(tmp_path: Path):
    """Both files claiming one member is a contradiction the roster wins, and it is still reported."""
    drift = _drift(tmp_path, f"{SHIPPED}.pt\n", f"{SHIPPED}.pt # not really\n{SMOKE}.pt # smoke\n")

    assert drift.contradictory() == [SHIPPED], "packaged and declined at once"


def test_phantom(tmp_path: Path):
    """A roster line matching nothing packages an ensemble one member short of the one measured."""
    drift = _drift(tmp_path, f"{SHIPPED}.pt\ntypo_s2026.pt\n", "")

    assert drift.phantom() == ["typo_s2026.pt"], "the typo matches no checkpoint on disk"


def _pixels(root: Path) -> Path:
    """Write the arrays the fixture's packaged member names, so its cache resolves on disk."""
    cache = Member.from_checkpoint(root / f"{SHIPPED}.pt").cache(root)
    cache.images_path.write_bytes(b"")
    cache.mask_path.write_bytes(b"")
    return cache.images_path


def test_unreadable(tmp_path: Path):
    """A packaged member whose pixels are gone is a finding, and losing EITHER array is enough.

    `adjacent224_s2026` kept its mask and lost its images, so a check reading one array would have
    called it healthy while it was unscoreable and unshippable (knee-dhpo9).
    """
    drift = _drift(tmp_path, f"{SHIPPED}.pt\n", f"{SMOKE}.pt # smoke\n{BARE}.pt # stale\n{FORGOTTEN}.pt # gone\n")
    images = _pixels(tmp_path)

    assert drift.unreadable() == [], "both arrays are on disk"

    images.unlink()

    assert drift.unreadable() == [f"{SHIPPED} {images}"], "the mask survived and the member is still gone"


def test_complaints(tmp_path: Path):
    """A roster that describes disk raises nothing."""
    assert _clean(tmp_path).complaints() == [], "every scoreable member is on one side or the other"


def test_report(tmp_path: Path):
    """The survey closes with the counts, whether or not anything drifted."""
    lines = _clean(tmp_path).report()

    assert lines[-1] == "4 scoreable of 5 checkpoints: 1 packaged, 3 declined, 0 undocumented"
