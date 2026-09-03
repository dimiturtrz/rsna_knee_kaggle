import json
from pathlib import Path

import pytest

from kneemri.evaluation.seat_bonus import SeatBonus, SeatEntry, Seating

RECORD = [
    {"note": "a comment line names no roster and must not be read as an entry"},
    {"id": "solo_a", "board": 0.920, "members_named": ["a"], "kit": "k1", "scope": "ensemble"},
    # Two entries at one seat count: `matched()` must report the anchor's BEST there, not its first.
    {"id": "pair_a_worse", "board": 0.931, "members_named": ["a", "c"], "kit": "k1", "scope": "ensemble"},
    {"id": "pair_a", "board": 0.933, "members_named": ["a", "b"], "kit": "k1", "scope": "ensemble"},
    {"id": "four_a", "board": 0.936, "members_named": ["a", "b", "c", "d"], "kit": "k2", "scope": "ensemble"},
    {"id": "six_a", "board": 0.935, "members_named": ["a", "b", "c", "d", "e", "f"], "kit": "k2", "scope": "ensemble"},
    {"id": "solo_b", "board": 0.925, "members_named": ["b"], "kit": "k1", "scope": "ensemble"},
    {"id": "pair_b", "board": 0.930, "members_named": ["b", "a"], "kit": "k1", "scope": "ensemble"},
    # Seated first more than anyone, never scored alone: the member that cannot anchor a curve.
    {"id": "lead_z", "board": 0.931, "members_named": ["z", "a"], "kit": "k1", "scope": "ensemble"},
    {"id": "lead_z2", "board": 0.932, "members_named": ["z", "b"], "kit": "k1", "scope": "ensemble"},
    {"id": "lead_z3", "board": 0.930, "members_named": ["z", "c"], "kit": "k1", "scope": "ensemble"},
    {"id": "averaged", "board": 0.900, "members_named": ["a", "b"], "kit": "k1", "scope": "member_mean"},
    {"id": "blanked", "board": None, "members_named": ["a", "b"], "kit": "k1", "scope": "ensemble"},
    {"id": "nameless", "board": 0.928, "kit": "k1", "scope": "ensemble"},
]


def _record(tmp_path: Path) -> Path:
    path = tmp_path / "board.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in RECORD) + "\n", encoding="utf-8")
    return path


def test_entries(tmp_path: Path):
    """Only scored rows on this scope that name their roster, fewest seats first."""
    entries = SeatBonus(_record(tmp_path)).entries()

    assert [entry.seats for entry in entries] == sorted(entry.seats for entry in entries)
    assert {entry.id for entry in entries} == {
        "solo_a",
        "pair_a_worse",
        "pair_a",
        "four_a",
        "six_a",
        "solo_b",
        "pair_b",
        "lead_z",
        "lead_z2",
        "lead_z3",
    }
    assert "averaged" not in {entry.id for entry in entries}, "a member mean is a different denominator"
    assert "blanked" not in {entry.id for entry in entries}, "an unscored entry prices nothing"
    assert "nameless" not in {entry.id for entry in entries}, "no roster, no seat count"


def test_solos(tmp_path: Path):
    """The members the board scored ALONE — the only ones a curve can have a zero at."""
    solos = SeatBonus(_record(tmp_path)).solos()

    assert set(solos) == {"a", "b"}
    assert solos["a"].id == "solo_a"


def test_anchors(tmp_path: Path):
    """Candidates are members that both lead a roster and were scored alone, most-seated first."""
    assert SeatBonus(_record(tmp_path)).anchors() == ["a", "b"]


def test_anchor(tmp_path: Path):
    """`z` leads three rosters and outranks both candidates, and can anchor nothing: no zero."""
    assert SeatBonus(_record(tmp_path)).anchor() == "a"
    assert SeatBonus(_record(tmp_path), chosen="b").anchor() == "b"


def test_anchor_refuses_a_member_the_board_never_scored_alone(tmp_path: Path):
    """Asking for the most-seated member is how this reading would fail silently at a made-up zero."""
    with pytest.raises(ValueError, match="never scored alone"):
        SeatBonus(_record(tmp_path), chosen="z").anchor()


def test_anchor_refuses_a_record_with_no_solo_at_all(tmp_path: Path):
    """A record of ensembles only has no zero anywhere, and must say so rather than pick one."""
    path = tmp_path / "no_solo.jsonl"
    rows = [{"id": "pair", "board": 0.93, "members_named": ["a", "b"], "kit": "k", "scope": "ensemble"}]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="seated first and scored alone"):
        SeatBonus(path).anchor()


def test_solo(tmp_path: Path):
    """The curve's zero is a SCORED single, never another ensemble."""
    zero = SeatBonus(_record(tmp_path)).solo()

    assert zero.id == "solo_a"
    assert zero.seats == 1


def test_curve(tmp_path: Path):
    """Every entry anchored on that member, with what its extra seats bought over the single."""
    curve = SeatBonus(_record(tmp_path)).curve()

    assert [(seating.entry.seats, round(seating.bonus, 3)) for seating in curve] == [
        (2, 0.011),
        (2, 0.013),
        (4, 0.016),
        (6, 0.015),
    ]
    assert all(isinstance(seating, Seating) for seating in curve)
    assert "pair_b" not in {seating.entry.id for seating in curve}, "another anchor is another curve"


def test_saturation(tmp_path: Path):
    """The most any seat count bought — the bonus does not keep growing with members."""
    assert SeatBonus(_record(tmp_path)).saturation() == pytest.approx(0.016)


def test_knee(tmp_path: Path):
    """The fewest seats within the board's resolution of the best; the rest is unpaid runtime."""
    # Two seats reach 0.013, inside BOARD_RESOLVABLE (0.0040) of the 0.016 four seats reach, so the
    # two extra decodes buy a spread the board cannot order. At the old 0.0015 this read 4 seats.
    assert SeatBonus(_record(tmp_path)).knee().entry.seats == 2


def test_required_single(tmp_path: Path):
    """What a member must score alone for an ensemble around it to reach the target."""
    assert SeatBonus(_record(tmp_path)).required_single(0.950) == pytest.approx(0.934)


def test_alternatives(tmp_path: Path):
    """Each candidate anchor's own saturation, because they disagree and the disagreement is the result."""
    assert SeatBonus(_record(tmp_path)).alternatives() == {
        "a": pytest.approx(0.016),
        "b": pytest.approx(0.005),
    }


def test_matched(tmp_path: Path):
    """Held at one seat count the differing zeros drop out and only the roster is left."""
    matched = SeatBonus(_record(tmp_path)).matched()

    assert set(matched) == {2}, "4 and 6 seats are reached by one anchor only, so they compare nothing"
    assert matched[2]["a"].id == "pair_a", "the anchor's BEST entry at that count, not its first"
    assert matched[2]["b"].id == "pair_b"
    assert matched[2]["a"].board > matched[2]["b"].board


def test_kits(tmp_path: Path):
    """The confound this reading cannot hold fixed, counted rather than left unsaid."""
    assert SeatBonus(_record(tmp_path)).kits() == {"k1", "k2"}


def test_report(tmp_path: Path):
    """The report carries the zero, the saturation, the kit confound and the rival anchors."""
    report = "\n".join(SeatBonus(_record(tmp_path)).report(0.950))

    assert "anchor a, zero solo_a at board 0.920" in report
    assert "saturation +0.016, matched to within 0.004 at 2 seats" in report
    assert "needs a single at 0.934 board" in report
    assert "2 distinct kits" in report
    assert "other anchors: a +0.016, b +0.005" in report
    assert "at 2 seats the zeros drop out: a 0.933 (pair_a)  b 0.930 (pair_b)" in report


def _entry(*members: str) -> SeatEntry:
    return SeatEntry(id="entry", board=0.93, members=members, kit="k")


def test_seats():
    """How many members were averaged, which is the whole x-axis."""
    assert _entry("a", "b", "c").seats == 3


def test_first():
    """The seat-1 member: the identity a bonus is read against, and the roster order is meaningful."""
    assert _entry("a", "b").first == "a"
