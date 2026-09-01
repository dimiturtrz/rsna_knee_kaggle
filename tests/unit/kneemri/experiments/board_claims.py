"""One representative per equivalence class of board claim: the correct quote, the misattributed one,
and the four shapes that look wrong and are not (a pre-registered bar, a range, a number belonging to
something further up the line, a number on the member's ruler)."""

import json
from pathlib import Path

from kneemri.experiments.board_claims import BoardClaims, BoardDocument

ENTRIES = {"basedense_v1": 0.931, "votefit_v1": 0.938, "sixstrong_v1": 0.93}
STANDING: dict[str, object] = {"rank": 30, "field": 2814, "taken": "2026-09-01T07:33:16+00:00"}


def check(corpus: list[BoardDocument] | None = None) -> BoardClaims:
    """A check over injected truth, so no test needs the competition record."""
    return BoardClaims(ENTRIES, STANDING, corpus or [])


def document(number: int, line: str) -> BoardDocument:
    """A one-line corpus document."""
    return BoardDocument(label="note.md", lines=((number, line),))


def test_of(tmp_path: Path) -> None:
    path = tmp_path / "note.md"
    path.write_text("first\n| basedense_v1 | 0.931 |\nsecond\n", encoding="utf-8")
    assert BoardDocument.of(path) == BoardDocument(label=str(path), lines=((1, "first"), (3, "second")))


def test_beads(tmp_path: Path) -> None:
    path = tmp_path / "issues.jsonl"
    rows = [
        {"id": "knee-abcd", "title": "votefit_v1 reads 0.938", "notes": "| a | 0.931 |\nand a tail"},
        {"schema": "beads/v1"},
    ]
    path.write_text("\n\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    found = BoardDocument.beads(path)
    # The record line carrying no id is not an issue, and the note's table row is not a sentence.
    assert found == [BoardDocument(label="knee-abcd", lines=((1, "votefit_v1 reads 0.938"), (4, "and a tail")))]
    assert BoardDocument.beads(tmp_path / "absent.jsonl") == []


def test_scored(tmp_path: Path) -> None:
    board = tmp_path / "board.jsonl"
    rows = [
        {"id": "votefit_v1", "board": 0.938},
        {"id": "excluded_v1", "board": 0.9, "scope": "excluded"},
        {"ref": 1234, "board": 0.921},
        {"id": "unscored_v1"},
    ]
    board.write_text("\n\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    # An excluded row still has a score somebody can misquote; an unscored or unnamed row has nothing.
    assert BoardClaims.scored(board) == {"votefit_v1": 0.938, "excluded_v1": 0.9}
    assert BoardClaims.scored(tmp_path / "absent.jsonl") == {}


def test_latest(tmp_path: Path) -> None:
    history = tmp_path / "leaderboard.jsonl"
    rows = [{"rank": 29, "taken": "2026-08-31T05:09:00+00:00"}, {"rank": 30, "taken": "2026-09-01T07:33:16+00:00"}]
    history.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert BoardClaims.latest(history)["rank"] == 30
    assert BoardClaims.latest(tmp_path / "absent.jsonl") == {}


def test_documents(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "note.md").write_text("a line\n", encoding="utf-8")
    beads = tmp_path / "issues.jsonl"
    beads.write_text(json.dumps({"id": "knee-abcd", "title": "a title"}), encoding="utf-8")
    found = BoardClaims.documents((tmp_path / "sub",), beads)
    assert [item.label for item in found] == [str(tmp_path / "sub" / "note.md"), "knee-abcd"]


def test_quoted() -> None:
    assert [text for _, _, text in BoardClaims.quoted("votefit_v1 reads 0.938 and 0.93")] == ["0.938", "0.93"]
    # A bar, a signed delta, a range, and a number below the board's own floor.
    assert BoardClaims.quoted("its branch was >=0.939") == []
    assert BoardClaims.quoted("the swap bought +0.938") == []
    assert BoardClaims.quoted("FLAT is 0.9425-0.9455") == []
    assert BoardClaims.quoted("a blend weight of 0.330") == []


def test_named() -> None:
    assert check().named("`votefit_v1` scored 0.938") == "votefit_v1"
    # Two entries named is a comparison, and both its numbers are somebody's by design.
    assert check().named("`votefit_v1` 0.938 against `basedense_v1` 0.931") is None
    assert check().named("no entry here at all") is None


def test_about() -> None:
    assert check().about("`votefit_v1` scored 0.938", "votefit_v1") == {"0.938"}
    # Far enough away to be a claim about something else, and a number on the member's ruler.
    assert check().about("0.946 is what the target asks of any entry, so votefit_v1 is short", "votefit_v1") == set()
    assert check().about("votefit_v1's member holdout predicts 0.938", "votefit_v1") == set()


def test_misquoted() -> None:
    assert check([document(4, "`votefit_v1` scored 0.938")]).misquoted() == []
    # 0.93 and 0.930 are the same claim written two ways, so neither is a finding.
    assert check([document(4, "`sixstrong_v1` scored 0.930")]).misquoted() == []
    findings = check([document(4, "`votefit_v1` scored 0.931")]).misquoted()
    assert findings == [("note.md", 4, "votefit_v1", {"0.931"}, 0.938)]


def test_stale_ranks() -> None:
    assert check([document(1, "we sit at rank 30 of 2814")]).stale_ranks() == []
    # A line that carries its own read date is a dated read, and a dated read is never stale.
    assert check([document(1, "rank 50 of 2404, read 2026-08-25")]).stale_ranks() == []
    assert check([document(1, "0.919 is rank 84 of 1758")]).stale_ranks() == [("note.md", 1, 84, 1758)]
    assert BoardClaims(ENTRIES, {}, [document(1, "rank 84 of 1758")]).stale_ranks() == []
