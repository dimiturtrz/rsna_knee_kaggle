"""Whether the board already holds a candidate's exact member set, read without spending a slot."""

import json
from pathlib import Path

from kneemri.kaggle.board_check import BoardRow
from kneemri.kaggle.submit_check import SubmitCheck, TriedIdea

PEAK8 = ("a", "b", "c")


def _row(identifier: str, members: tuple[str, ...], *, board: float | None = 0.932) -> BoardRow:
    return BoardRow.parsed(
        {"ref": "55751365", "id": identifier, "board": board, "members_named": list(members), "date": "2026-08-24"}
    )


def _kernel(tmp_path: Path, name: str, body: str) -> Path:
    kernel = tmp_path / name
    kernel.mkdir()
    (kernel / f"{name}.py").write_text(body, encoding="utf-8")
    return kernel


def test_repeats():
    """Set equality, not order and not the kernel's name — renaming is what happens between attempts."""
    shipped = SubmitCheck([_row("peak8_v1", PEAK8), _row("other_v1", ("a", "b", "d"))])
    matched = shipped.repeats({"c", "b", "a"})
    assert [row.identifier for row in matched] == ["peak8_v1"], "a reordered tuple is the same entry"


def test_overlaps():
    """Near misses report and never refuse, ranked by how much of the SHIPPED entry is being re-run."""
    shipped = SubmitCheck([_row("half_v1", ("a", "b", "x", "y")), _row("same_v1", PEAK8)])
    near = shipped.overlaps(set(PEAK8))
    assert [(row.identifier, part) for row, part in near] == [("half_v1", 0.5)], "an exact match is not an overlap"


def test_submit_check_described():
    """The line carries the ref and the score, which is what decides whether to spend the slot anyway."""
    assert "ref 55751365" in SubmitCheck([]).described(_row("peak8_v1", PEAK8))
    assert "unscored" in SubmitCheck([]).described(_row("pending_v1", PEAK8, board=None))


def test_tried_idea_described():
    """A scored and an unscored try both read as one line, the prose cut to what a reader scans."""
    assert TriedIdea("a_v1", 0.9421, "x" * 500).described() == f"a_v1 (board 0.942): {'x' * 160}"
    assert TriedIdea("b_v1", None, "short").described() == "b_v1 (board unscored): short"


def test_seats(tmp_path: Path):
    """A named tuple resolves; a kernel that ranks the package at run time answers None, never empty."""
    named = _kernel(tmp_path, "kneemri-named", 'KernelRun(names=("a", "b")).run()\n')
    ranked = _kernel(tmp_path, "kneemri-ranked", "KernelRun(members=2).run()\n")
    assert SubmitCheck.seats(named) == {"a", "b"}
    assert SubmitCheck.seats(ranked) is None, "unknown seats must not read as a new entry"
    blended = _kernel(
        tmp_path, "kneemri-blended", 'KernelRun(names=("a",)).run()\nRaptorEntry.weights(mounts=mounts)\n'
    )
    assert SubmitCheck.seats(blended) == {"a", "raptor_v4_swa"}, "a seat outside `names` is still a seat"


def test_resolved(tmp_path: Path):
    """A bare name resolves under the kernels root; an existing directory is taken as given."""
    kernel = _kernel(tmp_path, "kneemri-named", "KernelRun().run()\n")
    assert SubmitCheck.resolved("kneemri-named", tmp_path) == kernel
    assert SubmitCheck.resolved(str(kernel), tmp_path / "unused") == kernel
    assert SubmitCheck.resolved("dimiturnt/kneemri-named", tmp_path) == kernel, "the Kaggle id names the same kernel"
    assert not SubmitCheck.resolved("kneemri-absent", tmp_path).is_dir(), "a missing kernel stays missing"


def test_rows_without_names(tmp_path: Path):
    """A row that never named its members is dropped, since an empty set matches nothing honestly."""
    (tmp_path / "board.jsonl").write_text(json.dumps({"ref": "1", "id": "old_v1"}) + "\n", encoding="utf-8")
    shipped = SubmitCheck([BoardRow.parsed({"ref": "1", "id": "old_v1"})])
    assert shipped.rows == [], "silence about seats is not a claim about them"


def _record(tmp_path: Path) -> Path:
    rows = [
        {"note": "commentary names a ruler too, and is no entry"},
        {"id": "cleanseat_v1", "board": 0.942, "what": "Greedy seats on the corrected RULER", "why": "clean pool"},
        {"id": "l75seat_v1", "board": 0.939, "what": "seats chosen on ledger75", "offline_scope": "greedy, ruler T1"},
        {"id": "peak8_v1", "board": None, "what": "the eight strongest members"},
    ]
    record = tmp_path / "board.jsonl"
    record.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return record


def _tried(record: Path, *words: str) -> list[str]:
    return [idea.identifier for idea in TriedIdea.matched(record, list(words))]


def test_matched(tmp_path: Path):
    """Every word must appear, in any prose field, any case; a note line and a partial match are not tries."""
    record = _record(tmp_path)
    assert _tried(record, "ruler", "greedy") == ["cleanseat_v1", "l75seat_v1"], "any field, any case"
    assert _tried(record, "ruler", "strongest") == [], "ALL words, not any"
    assert _tried(record, "commentary") == [], "a note is no entry"


def test_prose():
    """Only strings are prose: a score or a member list never matches a word."""
    row = {"what": "Greedy", "why": "Ruler", "board": 0.9, "members_named": ["ruler"]}
    assert TriedIdea.prose(row) == "greedy ruler"


def test_licence_refusals(tmp_path: Path):
    """Every shelf is read, so a tag on a LATER shelf refuses as surely as one on the first."""
    first, second = tmp_path / "shelf1", tmp_path / "shelf2"
    first.write_text("clean_s2026.pt  # nothing external\n", encoding="utf-8")
    second.write_text("tainted_s2026.pt  # licence: rijeka\n", encoding="utf-8")
    assert SubmitCheck.licence_refusals([first, second], {"clean_s2026"}) == []
    (refused,) = SubmitCheck.licence_refusals([first, second], {"clean_s2026", "tainted_s2026"})
    assert refused.startswith("tainted_s2026 trains on the rijeka table")


def test_warnings(tmp_path: Path):
    """No idea word reads as UNCHECKED, never as clean; with words, each matching try is one line."""
    record = _record(tmp_path)
    (unchecked,) = TriedIdea.warnings(record, [], "kneemri-new")
    assert unchecked.startswith("IDEA NOT CHECKED kneemri-new")
    tried = TriedIdea.warnings(record, ["ruler", "greedy"], "kneemri-new")
    assert [line.split(" ")[1] for line in tried] == ["cleanseat_v1", "l75seat_v1"]
    assert all(line.startswith("TRIED ") for line in tried)
    assert TriedIdea.warnings(record, ["unheard"], "kneemri-new") == []
