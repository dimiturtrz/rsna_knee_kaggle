"""An arm's gate, over a board record this test writes itself."""

import json
from pathlib import Path

from kneemri.experiments.arm_gate import ArmGate

HEADER = """#!/usr/bin/env bash
# final_fit_six.sh — the expensive half of a cost-tiered pair.
# Run this ONLY if `kneemri-finalfit` reads 0.936 or above on the board.
# gate: kneemri-finalfit >= 0.936
"""


def _board(path: Path, rows: list[dict[str, object]]) -> Path:
    """A board record: a note line the reader must tolerate, then one object per scored entry."""
    lines = [json.dumps({"note": "the prose ledger is comp/SUBMISSIONS.md"})]
    lines += [json.dumps(row) for row in rows]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_declared():
    """The gate is the LINE, not the paragraph saying the same thing — prose has no consumer."""
    gate = ArmGate.declared(HEADER)

    assert gate == ArmGate("kneemri-finalfit", 0.936)
    assert ArmGate.declared("# bead: knee-1e3m\n") is None, "an ungated arm is the normal one"


def test_reading(tmp_path: Path):
    """Best over the entry's rows, matched on `id` OR `kernel` — half the record predates `kernel`."""
    board = _board(
        tmp_path / "board.jsonl",
        [
            {"id": "finalfit_v1", "kernel": "kneemri-finalfit", "board": 0.934},
            {"id": "finalfit_v2", "kernel": "kneemri-finalfit", "board": 0.931},
            {"id": "peak8_v1", "board": 0.932},
        ],
    )

    assert ArmGate("kneemri-finalfit", 0.936).reading(board) == 0.934, "a worse re-push does not retire it"
    assert ArmGate("peak8_v1", 0.9).reading(board) == 0.932, "an older row carries no kernel, and still counts"
    assert ArmGate("kneemri-never", 0.9).reading(board) is None


def test_unmet(tmp_path: Path):
    """The sentence the launcher reads, and the three ways an arm is refused its card.

    MEASURED 2026-08-23: `kneemri-finalfit` read 0.934 against this arm's own 0.936 floor and the arm
    ran for five hours anyway. An UNREAD gate refuses for the same reason a low one does — an arm
    gated on a submission nobody has pushed is the arm most likely to be launched on a hunch.
    """
    board = _board(tmp_path / "board.jsonl", [{"id": "finalfit_v1", "kernel": "kneemri-finalfit", "board": 0.934}])

    assert "the board reads 0.934" in ArmGate("kneemri-finalfit", 0.936).unmet(board)
    assert ArmGate("kneemri-finalfit", 0.934).unmet(board) == "", "met at exactly the floor is met"
    assert "never scored" in ArmGate("kneemri-unpushed", 0.9).unmet(board)
    assert "is not on disk" in ArmGate("kneemri-finalfit", 0.9).unmet(tmp_path / "absent.jsonl"), (
        "a check that passes because its evidence is missing is the failure mode, not the pass"
    )
