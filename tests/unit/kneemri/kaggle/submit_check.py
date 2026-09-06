"""Whether the board already holds a candidate's exact member set, read without spending a slot."""

import json
from pathlib import Path

from kneemri.kaggle.board_check import BoardRow
from kneemri.kaggle.submit_check import SubmitCheck

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


def test_described():
    """The line carries the ref and the score, which is what decides whether to spend the slot anyway."""
    assert "ref 55751365" in SubmitCheck([]).described(_row("peak8_v1", PEAK8))
    assert "unscored" in SubmitCheck([]).described(_row("pending_v1", PEAK8, board=None))


def test_seats(tmp_path: Path):
    """A named tuple resolves; a kernel that ranks the package at run time answers None, never empty."""
    named = _kernel(tmp_path, "kneemri-named", 'KernelRun(names=("a", "b")).run()\n')
    ranked = _kernel(tmp_path, "kneemri-ranked", "KernelRun(members=2).run()\n")
    assert SubmitCheck.seats(named) == {"a", "b"}
    assert SubmitCheck.seats(ranked) is None, "unknown seats must not read as a new entry"


def test_resolved(tmp_path: Path):
    """A bare name resolves under the kernels root; an existing directory is taken as given."""
    kernel = _kernel(tmp_path, "kneemri-named", "KernelRun().run()\n")
    assert SubmitCheck.resolved("kneemri-named", tmp_path) == kernel
    assert SubmitCheck.resolved(str(kernel), tmp_path / "unused") == kernel


def test_rows_without_names(tmp_path: Path):
    """A row that never named its members is dropped, since an empty set matches nothing honestly."""
    (tmp_path / "board.jsonl").write_text(json.dumps({"ref": "1", "id": "old_v1"}) + "\n", encoding="utf-8")
    shipped = SubmitCheck([BoardRow.parsed({"ref": "1", "id": "old_v1"})])
    assert shipped.rows == [], "silence about seats is not a claim about them"
