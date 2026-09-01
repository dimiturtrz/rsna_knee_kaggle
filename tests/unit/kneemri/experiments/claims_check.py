"""One representative per equivalence class of prose line: correct, mis-attributed, and the three
shapes that look wrong and are not (a second name on the line, an entry id, a non-score marker)."""

from pathlib import Path

from kneemri.experiments.claims_check import ClaimsCheck

RUNS = {"band_dense_s2026": 0.8578, "band_dense_long_s2026": 0.8587, "attend_dense_s2026": 0.8569}
RECORDED = {
    "band_dense_s2026": {"0.8578", "0.8501"},
    "band_dense_long_s2026": {"0.8587"},
    "attend_dense_s2026": {"0.8569"},
}


def check(documents_read: list[Path] | None = None, entries: set[str] | None = None) -> ClaimsCheck:
    """A check over injected truth, so no test needs a data root."""
    return ClaimsCheck(RUNS, RECORDED, documents_read or [], entries or set())


def test_readable(tmp_path: Path) -> None:
    document = tmp_path / "note.md"
    document.write_text("first\n| a | 0.8578 |\nsecond\n", encoding="utf-8")
    assert check().readable(document) == [(1, "first"), (3, "second")]
    assert check().readable(tmp_path / "absent.md") == []


def test_columnar() -> None:
    assert ClaimsCheck.columnar("| band_dense | 0.8578 |")
    assert ClaimsCheck.columnar("    band_dense       0.8578")
    assert ClaimsCheck.columnar("#     band_dense       0.8578")
    assert not ClaimsCheck.columnar("`band_dense` scored 0.8578 on the holdout")


def test_named() -> None:
    assert check().named("`band_dense` scored 0.8578") == "band_dense_s2026"
    assert check().named("band_dense_s2026 scored 0.8578") == "band_dense_s2026"
    # Two runs named is a comparison, and both its numbers are somebody's by design.
    assert check().named("`band_dense` 0.8578 against `attend_dense` 0.8569") is None
    # A board entry's own score is not a claim about the member whose name it contains.
    assert check(entries={"basedense_v1"}).named("`basedense_v1` (0.8690) beat `band_dense`") is None
    assert check().named("no run here at all") is None


def test_quoted() -> None:
    assert check().quoted("`band_dense` 0.8578 and 0.8501") == {"0.8578", "0.8501"}
    # Out of band, a threshold, a delta, and a line whose subject is a pair or a loss.
    assert check().quoted("weight 0.3300 of the blend") == set()
    assert check().quoted("the test was > 0.8731") == set()
    assert check().quoted("its nearest neighbour is 0.9306") == set()
    assert check().quoted("train loss 0.7493") == set()


def test_disputed(tmp_path: Path) -> None:
    good = tmp_path / "right.md"
    good.write_text("Both are pairs against `band_dense` (0.8578).\n", encoding="utf-8")
    bad = tmp_path / "wrong.md"
    bad.write_text("Both are pairs against `band_dense` (0.8587).\n", encoding="utf-8")
    assert check([good]).disputed() == []
    findings = check([bad]).disputed()
    assert len(findings) == 1
    document, line, name, scores, best = findings[0]
    assert (document, line, name, scores, best) == (bad, 1, "band_dense_s2026", {"0.8587"}, "0.8578")


def test_held_by() -> None:
    # The real case: 0.8587 is a number, but it is the long twin's.
    assert check().held_by("0.8587") == ["band_dense_long_s2026"]
    assert check().held_by("0.9999") == []


def test_documents(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "note.md").write_text("x", encoding="utf-8")
    (tmp_path / "sub" / "code.py").write_text("x", encoding="utf-8")
    (tmp_path / "sub" / "data.json").write_text("{}", encoding="utf-8")
    loose = tmp_path / "AGENTS.md"
    loose.write_text("x", encoding="utf-8")
    found = ClaimsCheck.documents((tmp_path / "sub", loose, tmp_path / "absent.md"))
    assert found == [tmp_path / "sub" / "code.py", tmp_path / "sub" / "note.md", loose]


def test_board_entries(tmp_path: Path) -> None:
    board = tmp_path / "board.jsonl"
    board.write_text('{"id": "basedense_v1"}\n\n{"id": "votefit_v1"}\n', encoding="utf-8")
    assert ClaimsCheck.board_entries(board) == {"basedense_v1", "votefit_v1"}
    assert ClaimsCheck.board_entries(tmp_path / "absent.jsonl") == set()


def test_sourced() -> None:
    assert check().sourced({"0.8587"}) == "0.8587 (held by band_dense_long_s2026)"
    assert check().sourced({"0.9999"}) == "0.9999 (held by NO run)"


def test_recorded_scores(tmp_path: Path) -> None:
    (tmp_path / "band_dense_s2026.metrics.jsonl").write_text(
        '{"epoch": 1, "auc": 0.8501}\n{"epoch": 2, "auc": 0.8578}\n', encoding="utf-8"
    )
    scores = ClaimsCheck.recorded_scores(tmp_path, {"band_dense_s2026": 0.8578, "no_metrics_s2026": 0.8123})
    # Every epoch a run wrote, plus the kept value for the 50 runs that left only a checkpoint.
    assert scores == {"band_dense_s2026": {"0.8501", "0.8578"}, "no_metrics_s2026": {"0.8123"}}
