"""A roster's cost is its decode count, and resolution is the axis that must never raise it."""

from pathlib import Path

import numpy as np
import pytest

from core.data.corpus.slot_reader import Band
from core.data.corpus.slots import SCHEMES
from kneemri.inference.members import Member, MemberConfig
from kneemri.kaggle.roster_reads import ReadGroup, RosterReads

CANONICAL = "canonical"
SLOTS = [slot.name for slot in SCHEMES[CANONICAL]]
LABELS = 12


def _config(**overrides: object) -> MemberConfig:
    """The fields a decode key is taken from, at their historic values, plus whatever a test varies."""
    config = {
        "img": 336,
        "slots": SLOTS,
        "slices": 12,
        "group": 3,
        "band": list(Band().recorded()),
    }
    return dict(config | overrides)  # type: ignore[return-value]


def _member(identifier: str, holdout: float = 0.85, **overrides: object) -> Member:
    """One member with no weights behind it — every field this module reads is in the config."""
    return Member(
        identifier=identifier,
        file=Path(f"{identifier}.pt"),
        fold=0,
        config=_config(**overrides),
        fingerprint=np.zeros((2, LABELS), np.float32),
        holdout=holdout,
        annot=0.5,
    )


def test_over():
    """THE FINDING, as a test: three resolutions are ONE decode, and a different crop is a second.

    If this ever fails, an entry that reads as cheap is being served expensively, and the seats-per-decode
    argument the roster was built on is void.
    """
    members = [
        _member("small", img=224),
        _member("mid", img=336),
        _member("wide", img=448),
        _member("cropped", crop_mm=80.0),
    ]

    reads = RosterReads.over(members)

    assert len(reads.groups) == 2, "resolution must not split a decode; the crop must"
    assert [len(group.members) for group in reads.groups] == [3, 1], "largest read first"
    assert reads.groups[0].sizes == [224, 336, 448], "one read serves every size it was asked for"


def test_keyed():
    """A checkpoint that describes no reading is placed nowhere rather than taking the report down."""
    unreadable = Member(
        identifier="teacher",
        file=Path("teacher.pt"),
        fold=0,
        config={},  # type: ignore[typeddict-item]
        fingerprint=np.zeros((2, LABELS), np.float32),
        holdout=float("nan"),
        annot=float("nan"),
    )

    assert RosterReads.keyed(unreadable) is None, "no slot list means no decode"
    assert RosterReads.keyed(_member("real")) is not None, "a member with a reading names its decode"


def test_placed():
    """The report speaks only about members that answered a decode, never about everything read."""
    reads = RosterReads.over([_member("a"), _member("b", img=224)])

    assert [member.identifier for member in reads.placed] == ["a", "b"], "both rode the one read"


def test_seats():
    """Seats are members served, which is what `per_decode` divides."""
    assert RosterReads.over([_member("a"), _member("b"), _member("c", slices=20)]).seats == 3


def test_per_decode():
    """Two members sharing a read buy twice the seats of two members that cannot."""
    shared = RosterReads.over([_member("a"), _member("b", img=224)]).per_decode
    split = RosterReads.over([_member("a"), _member("b", slices=20)]).per_decode

    assert shared == pytest.approx(2.0), "one decode, two seats"
    assert split == pytest.approx(1.0), "a different slice depth is a second decode"


def test_roster_reads_lines():
    """The trade is the last line, because it is the number the tool exists to report."""
    lines = RosterReads.over([_member("a"), _member("b", img=224)]).lines()

    assert lines[-1] == "2 seats on 1 decodes = 2.0 seats per decode"
    assert "224, 336" in lines[0].replace("[", "").replace("]", ""), "the free axis is visible in the header"


def test_sizes():
    """A group reports the distinct sizes it serves, sorted, however its members were listed."""
    assert ReadGroup([_member("a", img=448), _member("b", img=224)]).sizes == [224, 448]


def test_description():
    """The header names every field that CAN split a decode, so a stray group explains itself."""
    description = ReadGroup([_member("a")]).description

    for field in (CANONICAL, "12 slices", "band 0.2-0.8", "centre image", "crop 130.0mm", "tie first"):
        assert field in description, f"{field} decides a decode and must be readable in the header"


def test_read_group_lines():
    """Members read best-holdout first, because the best seat is the one a roster takes."""
    rows = ReadGroup([_member("worse", holdout=0.80), _member("better", holdout=0.86)]).lines()

    assert "better" in rows[1] and "worse" in rows[2], "the best seat reads first"


def test_read_members(tmp_path: Path):
    """A named checkpoint that is not on disk is reported, not raised on — a candidate list is edited."""
    assert RosterReads.read_members([tmp_path / "absent.pt"]) == [], "an absent checkpoint contributes no member"


def test_resolved(tmp_path: Path):
    """A roster line carries a comment and may omit the suffix; both resolve to the same checkpoint."""
    lines = ["native224_s2026.pt  # DARK, never scored", "native224_s2026", "", "# a whole-line note"]
    paths = RosterReads.resolved(lines, tmp_path)

    assert paths == [tmp_path / "native224_s2026.pt", tmp_path / "native224_s2026.pt"], "comments strip, suffix fills"
