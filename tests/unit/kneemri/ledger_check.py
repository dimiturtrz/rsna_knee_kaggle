import json
from pathlib import Path

import torch

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


def test_finished(tmp_path: Path):
    """A run counts as finished on its checkpoint, and the metrics file refines the score.

    Scanning metrics files alone was a pass computed over 15 of 69 runs: 50 checkpoints had no metrics
    file, and 11 of those carried a score written down in no document at all.
    """
    torch.save({"holdout": 0.8143}, tmp_path / "checkpoint_only.pt")
    torch.save({"holdout": 0.81}, tmp_path / "both.pt")
    torch.save({"holdout": float("nan")}, tmp_path / "final_fit.pt")
    torch.save({"holdout": 0.77}, tmp_path / "throughput_bf16.pt")
    (tmp_path / "both.metrics.jsonl").write_text(
        "\n".join(json.dumps({"epoch": e, "auc": auc}) for e, auc in enumerate([0.81, 0.8473])),
        encoding="utf-8",
    )

    finished = LedgerCheck(tmp_path / "absent.md").finished(tmp_path)

    assert finished["checkpoint_only"] == 0.8143, "a run with no metrics file still ran"
    assert finished["both"] == 0.8473, "the metrics file knows the BEST epoch, the checkpoint only its own"
    assert "final_fit" not in finished, "a final fit records NaN by design and never scored"
    assert "throughput_bf16" not in finished, "speed probes have no score worth quoting in a results ledger"


def test_unrecorded(tmp_path: Path):
    """A score the ledger quotes is recorded; one it does not is reported.

    Matched on the NUMBER, not the arm name: the table names arms in prose (`whole_stack`) while the
    files are outputs (`wholestack_s2026`), and a mapping between the two is one more thing to get
    wrong. This is exactly the case that went unnoticed for two days — five arms sat in the ledger as
    "queued" while their metrics files held real scores.
    """
    ledger = tmp_path / "README.md"
    ledger.write_text("| `slice_attention` | attention over slices | 0.8473 | +0.0016 | kept |", encoding="utf-8")
    interpretation = tmp_path / "2026-08-11_what-actually-moved.md"
    interpretation.write_text("| + weight floor at 0.25 | 0.8221 | +0.0019 | kept |", encoding="utf-8")

    missing = LedgerCheck(ledger, interpretation, tmp_path / "absent.md").unrecorded(
        {"sliceattn_s2026": 0.8473, "floor_fold0": 0.8221, "extended_ws_s2026": 0.8459}
    )

    assert "sliceattn_s2026" not in missing, "a score quoted in the ledger is recorded"
    assert "floor_fold0" not in missing, "an interpretation is part of the written record, not a lesser one"
    assert missing == {"extended_ws_s2026": 0.8459}
