"""An entry is decomposed over the members that held something out, or not at all — never silently."""

import json
from pathlib import Path

import numpy as np

from core.findings import TARGETS
from kneemri.kaggle.champion_check import Champion
from kneemri.kaggle.champion_uncertainty import ChampionUncertainty
from kneemri.runs.holdout_record import HoldoutRecord

STUDIES = 40


def _champion(*members: str) -> Champion:
    """A pinned entry, whose members are BARE names — the pin stores `band_dense_s2026`, not a filename."""
    return Champion(entry="test", kernel="k", board=0.9, members=members, kit="abc123")


def _member(path: Path, seed: int) -> None:
    """One member's record and the checkpoint it was read from."""
    studies = [f"s{index}" for index in range(STUDIES)]
    logits = np.random.default_rng(seed).normal(0.0, 1.0, (STUDIES, len(TARGETS)))
    truth = np.repeat(np.linspace(0.0, 1.0, STUDIES)[:, None], len(TARGETS), axis=1).astype(np.float32)
    HoldoutRecord(path).write(studies, 1.0 / (1.0 + np.exp(-logits)), truth, epoch=6)
    path.write_bytes(b"")


def _board(tmp_path: Path) -> Path:
    """A board with one fully scoreable entry on it, for the refusal's last line to name."""
    rows = [
        {"note": "the record's own preamble, which is not an entry"},
        {"id": "whole", "kernel": "k", "board": 0.933, "members_named": ["one", "other"], "scope": "ensemble"},
    ]
    path = tmp_path / "board.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_members(tmp_path: Path):
    """The pin's bare names, resolved under the processed root rather than the working directory."""
    members = ChampionUncertainty(tmp_path).members(_champion("one", "other"))

    assert members == [tmp_path / "one.pt", tmp_path / "other.pt"]


def test_held(tmp_path: Path):
    """The split into what can be scored and what never can, by whether a record exists at all."""
    _member(tmp_path / "one.pt", 1)
    (tmp_path / "allfit_ff.pt").write_bytes(b"")

    scoreable, unscoreable = ChampionUncertainty(tmp_path).held([tmp_path / "one.pt", tmp_path / "allfit_ff.pt"])

    assert scoreable == [tmp_path / "one.pt"]
    assert unscoreable == [tmp_path / "allfit_ff.pt"]


def test_undecomposable(tmp_path: Path):
    """A refusal carrying its own denominator, the members that caused it, and a way out."""
    _member(tmp_path / "one.pt", 1)
    _member(tmp_path / "other.pt", 2)
    lines = ChampionUncertainty(tmp_path, _board(tmp_path)).undecomposable([tmp_path / "one.pt"], [tmp_path / "x.pt"])

    assert "1 of 2 members" in lines[0]
    assert "x" in lines[1]
    assert "whole" in lines[2], "the entry that could answer instead is named, not left to the reader"


def test_report(tmp_path: Path):
    """The split over an entry whose members all held something out."""
    for index, name in enumerate(("one", "other")):
        _member(tmp_path / f"{name}.pt", index + 1)

    lines = ChampionUncertainty(tmp_path).report([tmp_path / "one.pt", tmp_path / "other.pt"])

    assert "2 separately trained members" in lines[0]
    assert "POOLED" in lines[-1]


def test_report_names_the_members_it_dropped(tmp_path: Path):
    """A partial entry is decomposed and SAYS SO — the silent subset is the failure being guarded."""
    for index, name in enumerate(("one", "other")):
        _member(tmp_path / f"{name}.pt", index + 1)
    (tmp_path / "allfit_ff.pt").write_bytes(b"")

    lines = ChampionUncertainty(tmp_path).report(
        [tmp_path / "one.pt", tmp_path / "other.pt", tmp_path / "allfit_ff.pt"]
    )

    assert "allfit_ff" in lines[0]
    assert "2 separately trained members" in lines[1]


def test_report_refuses_an_entry_with_nothing_left_to_compare(tmp_path: Path):
    """One scoreable member is not a disagreement, and the report says so instead of printing 0.0%."""
    _member(tmp_path / "one.pt", 1)
    (tmp_path / "allfit_ff.pt").write_bytes(b"")

    lines = ChampionUncertainty(tmp_path, _board(tmp_path)).report([tmp_path / "one.pt", tmp_path / "allfit_ff.pt"])

    assert "NO UNCERTAINTY SPLIT" in lines[0]
