import json
from pathlib import Path

from kneemri.ledger_check import LedgerCheck


def test_best(tmp_path: Path):
    """The best scored epoch, and None for a run that died before scoring one.

    None rather than 0.0, because a run that never scored is not a run that scored badly — quoting a
    zero in a results ledger would be worse than omitting it.
    """
    scored = tmp_path / "a.metrics.jsonl"
    scored.write_text(
        "\n".join(
            [
                json.dumps({"recipe": {"epochs": "3"}}),
                *(json.dumps({"epoch": e, "auc": auc}) for e, auc in enumerate([0.81, 0.8473, 0.84])),
            ]
        ),
        encoding="utf-8",
    )
    empty = tmp_path / "b.metrics.jsonl"
    empty.write_text("", encoding="utf-8")
    unscored = tmp_path / "c.metrics.jsonl"
    unscored.write_text(json.dumps({"epoch": 0, "auc": float("nan")}), encoding="utf-8")

    assert LedgerCheck(tmp_path / "absent.md").best(scored) == 0.8473, "the recipe line is not an epoch"
    assert LedgerCheck(tmp_path / "absent.md").best(empty) is None
    assert LedgerCheck(tmp_path / "absent.md").best(unscored) is None, (
        "a final fit records NaN by design, and 'nan' quoted in a ledger is never found again"
    )


def test_unrecorded(tmp_path: Path):
    """A score the ledger quotes is recorded; one it does not is reported.

    Matched on the NUMBER, not the arm name: the table names arms in prose (`whole_stack`) while the
    files are outputs (`wholestack_s2026`), and a mapping between the two is one more thing to get
    wrong. This is exactly the case that went unnoticed for two days — five arms sat in the ledger as
    "queued" while their metrics files held real scores.
    """
    ledger = tmp_path / "README.md"
    ledger.write_text("| `slice_attention` | attention over slices | 0.8473 | +0.0016 | kept |", encoding="utf-8")

    missing = LedgerCheck(ledger).unrecorded({"sliceattn_s2026": 0.8473, "extended_ws_s2026": 0.8459})

    assert "sliceattn_s2026" not in missing, "a score quoted in the ledger is recorded"
    assert missing == {"extended_ws_s2026": 0.8459}
