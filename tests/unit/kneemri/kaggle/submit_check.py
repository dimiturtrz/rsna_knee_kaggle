"""Whether the board already holds a candidate's exact member set, read without spending a slot."""

import json
import logging
from collections.abc import Mapping
from pathlib import Path

import pytest

from core import cli
from kneemri.kaggle.blend_shape import Fusion
from kneemri.kaggle.board_check import BoardRow
from kneemri.kaggle.kernel_script import ALL_OURS, Blend
from kneemri.kaggle.submit_check import UNRECORDED, SubmitCheck, TriedIdea

PEAK8 = ("a", "b", "c")


SEAT_THIRD = Blend.of({"nartaa384": 1 / 3}, "rank")
SEAT_PROBIT = Blend.of({"nartaa384": 1 / 3}, "probit")
MARKED = 'KernelRun(names=("a",)).run()\nload("nartaa_seat.csv")\n'


def _row(
    identifier: str, members: tuple[str, ...], *, board: float | None = 0.932, blend: Mapping[str, object] | None = None
) -> BoardRow:
    fields = {"ref": "55751365", "id": identifier, "board": board, "members_named": list(members), "date": "2026-08-24"}
    return BoardRow.parsed({**fields, **(blend or {})})


def _kernel(tmp_path: Path, name: str, body: str) -> Path:
    kernel = tmp_path / name
    kernel.mkdir()
    (kernel / f"{name}.py").write_text(body, encoding="utf-8")
    return kernel


def _foreign(identifier: str, fusion: str = "rank") -> BoardRow:
    return _row(identifier, PEAK8, blend={"foreign": {"nartaa384": 0.3333}, "fusion": fusion})


def test_same_members():
    """Set equality, not order and not the kernel's name — renaming is what happens between attempts."""
    shipped = SubmitCheck([_row("peak8_v1", PEAK8), _row("other_v1", ("a", "b", "d"))])
    matched = shipped.same_members({"c", "b", "a"})
    assert [row.identifier for row in matched] == ["peak8_v1"], "a reordered tuple is the same entry"


def test_repeats(tmp_path: Path):
    """Only members AND foreign shares AND fusion all equal make a repeat; any one axis apart is a new entry."""
    shipped = SubmitCheck([_row("plain_v1", PEAK8), _foreign("third_v1"), _foreign("probit_v1", "probit")], tmp_path)
    assert [row.identifier for row in shipped.repeats(set(PEAK8), ALL_OURS)] == ["plain_v1"]
    assert [row.identifier for row in shipped.repeats(set(PEAK8), SEAT_THIRD)] == ["third_v1"]
    assert [row.identifier for row in shipped.repeats(set(PEAK8), SEAT_PROBIT)] == ["probit_v1"]
    assert shipped.repeats(set(PEAK8), Blend.of({"nartaa384": 0.5}, "rank")) == [], "another share is another entry"
    assert shipped.repeats({"a", "b"}, ALL_OURS) == [], "the blend never rescues different members"


def test_variants(tmp_path: Path):
    """Every same-members row whose blend differs gets one line; the matching row gets none."""
    shipped = SubmitCheck([_row("plain_v1", PEAK8), _foreign("third_v1"), _foreign("probit_v1", "probit")], tmp_path)
    lines = shipped.variants(set(PEAK8), SEAT_PROBIT)
    assert len(lines) == len(["plain_v1", "third_v1"])
    assert all(line.startswith("SAME members as") for line in lines)
    assert not any("probit_v1" in line for line in lines), "the repeat itself is not a variant"


def test_varied(tmp_path: Path):
    """The line names exactly the axes that differ, and a blank foreign row says it is unrecorded."""
    shipped = SubmitCheck([], tmp_path)
    assert shipped.varied(_foreign("third_v1"), SEAT_PROBIT).endswith("but fusion probit vs rank")
    assert "foreign {} vs {'nartaa384': 0.3333}" in shipped.varied(_foreign("third_v1"), ALL_OURS)
    _kernel(tmp_path, "kneemri-blank", MARKED)
    assert shipped.varied(_row("blank_v1", PEAK8), SEAT_THIRD).endswith(UNRECORDED)


def test_recorded(tmp_path: Path):
    """Fields win; a blank row is all-ours unless its own kernel runs a foreign seat, which makes it unknown."""
    shipped = SubmitCheck([], tmp_path)
    assert shipped.recorded(_foreign("third_v1")) == SEAT_THIRD
    assert shipped.recorded(_row("fused_v1", PEAK8, blend={"fusion": "probit"})) == Blend((), Fusion.PROBIT)
    assert shipped.recorded(_row("gone_v1", PEAK8)) == ALL_OURS, "no surviving script and no fields: all ours"
    _kernel(tmp_path, "kneemri-ours", 'KernelRun(names=("a",)).run()\n')
    assert shipped.recorded(_row("ours_v1", PEAK8)) == ALL_OURS
    _kernel(tmp_path, "kneemri-blank", MARKED)
    assert shipped.recorded(_row("blank_v1", PEAK8)) is None, "a foreign entry left blank is never a match"


def test_judge(tmp_path: Path):
    """A whole repeat refuses unless `--anyway` says why; members alone shared with another blend ships."""
    shipped = SubmitCheck([_foreign("third_v1")], tmp_path)
    command = cli.Command("a test check", logging.getLogger(__name__))
    entry = (set(PEAK8), SEAT_THIRD)
    with pytest.raises(SystemExit) as refused:
        shipped.judge(command, "kneemri-third", entry, "")
    assert refused.value.code == cli.Exit.FOUND
    assert shipped.judge(command, "kneemri-third", entry, "re-reading the scorer") is None
    assert shipped.judge(command, "kneemri-probit", (set(PEAK8), SEAT_PROBIT), "") is None, "a variant never refuses"


def test_blend(tmp_path: Path):
    """A kernel's blend through its script; no script reads as None, never as all-ours."""
    seated = _kernel(
        tmp_path, "kneemri-seated", MARKED + 'RaptorSeat.combined("ours.csv", working / "nartaa_seat.csv", (1 / 3,))\n'
    )
    assert SubmitCheck.blend(seated) == SEAT_THIRD
    assert SubmitCheck.blend(_kernel(tmp_path, "kneemri-plain", "KernelRun().run()\n")) == ALL_OURS
    empty = tmp_path / "kneemri-empty"
    empty.mkdir()
    assert SubmitCheck.blend(empty) is None


def test_script(tmp_path: Path):
    """The directory's script parsed, or None when it holds no python."""
    kernel = _kernel(tmp_path, "kneemri-named", "KernelRun().run()\n")
    parsed = SubmitCheck.script(kernel)
    assert parsed is not None
    assert parsed.call() is not None
    assert SubmitCheck.script(tmp_path) is None


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


def test_licence(tmp_path: Path):
    """Every shelf is read: untagged passes silently, a foreign tag warns, a KneeCoT tag on a LATER shelf refuses."""
    first, second = tmp_path / "shelf1", tmp_path / "shelf2"
    first.write_text("clean_s2026.pt  # nothing external\nforeign_s2026.pt  # licence: rijeka\n", encoding="utf-8")
    second.write_text("banned_s2026.pt  # licence: kneecot\n", encoding="utf-8")
    shelves = [first, second]
    assert SubmitCheck.licence(shelves, {"clean_s2026"}) == ([], [])
    refused, (noted,) = SubmitCheck.licence(shelves, {"clean_s2026", "foreign_s2026"})
    assert refused == []
    assert noted.startswith("foreign_s2026 trains on the rijeka table: pick-eligible")
    (refused_line,), _ = SubmitCheck.licence(shelves, {"foreign_s2026", "banned_s2026"})
    assert refused_line.startswith("banned_s2026 trains on the kneecot table")


def test_warnings(tmp_path: Path):
    """No idea word reads as UNCHECKED, never as clean; with words, each matching try is one line."""
    record = _record(tmp_path)
    (unchecked,) = TriedIdea.warnings(record, [], "kneemri-new")
    assert unchecked.startswith("IDEA NOT CHECKED kneemri-new")
    tried = TriedIdea.warnings(record, ["ruler", "greedy"], "kneemri-new")
    assert [line.split(" ")[1] for line in tried] == ["cleanseat_v1", "l75seat_v1"]
    assert all(line.startswith("TRIED ") for line in tried)
    assert TriedIdea.warnings(record, ["unheard"], "kneemri-new") == []
