"""The mirror for the seat-to-name repair: every public method called, and asserted."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.patch_map_rekey import PatchMapRekey

STUDY_COLUMN = "StudyInstanceUID"
IDENTIFIERS = ["s0", "s1", "s2", "s3"]
SEATS = [2, 0]
GRID = 4


def _tables(tmp_path: Path) -> tuple[Path, Path]:
    """One derived row per study, each at its own level, so a wrong ORDER reads as wrong labels."""
    labels = tmp_path / "derived.csv"
    with labels.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[STUDY_COLUMN, *TARGETS])
        writer.writeheader()
        for position, study in enumerate(IDENTIFIERS):
            level = f"{0.1 * (position + 1):.1f}"
            writer.writerow({STUDY_COLUMN: study} | dict.fromkeys(TARGETS, level))

    reports = tmp_path / "train.csv"
    with reports.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[STUDY_COLUMN])
        writer.writeheader()
        for study in IDENTIFIERS:
            writer.writerow({STUDY_COLUMN: study})
    return labels, reports


def _index(tmp_path: Path, name: str, identifiers: list[str]) -> Path:
    path = tmp_path / f"index_{name}.json"
    path.write_text(json.dumps(identifiers), encoding="utf-8")
    return path


def _truth() -> np.ndarray:
    return np.asarray([[0.1 * (seat + 1)] * len(TARGETS) for seat in SEATS], dtype=np.float32)


def _maps(tmp_path: Path, studies: np.ndarray, name: str = "seats.npz") -> Path:
    path = tmp_path / name
    np.savez(
        path,
        maps=np.zeros((len(SEATS), len(TARGETS), GRID, GRID), dtype=np.float32),
        studies=studies,
        truth=_truth(),
        tissue=np.zeros((GRID, GRID), dtype=np.float64),
    )
    return path


def _rekey(tmp_path: Path, studies: np.ndarray | None = None) -> PatchMapRekey:
    labels, reports = _tables(tmp_path)
    recorded = np.asarray(SEATS) if studies is None else studies
    return PatchMapRekey(_maps(tmp_path, recorded), labels, reports)


def test_positions(tmp_path: Path) -> None:
    assert _rekey(tmp_path).positions().tolist() == SEATS


def test_positions_refuses_a_sweep_already_keyed_on_names(tmp_path: Path) -> None:
    rekey = _rekey(tmp_path, studies=np.asarray(["s2", "s0"]))
    with pytest.raises(ValueError, match="already keyed on names"):
        rekey.positions()


def test_named(tmp_path: Path) -> None:
    assert _rekey(tmp_path).named(_index(tmp_path, "full", IDENTIFIERS)) == ["s2", "s0"]


def test_named_refuses_an_index_too_short_for_the_seats(tmp_path: Path) -> None:
    rekey = _rekey(tmp_path)
    with pytest.raises(ValueError, match="names seat 2"):
        rekey.named(_index(tmp_path, "short", IDENTIFIERS[:2]))


def test_reproduces(tmp_path: Path) -> None:
    """The right order reproduces the stored truth; a permuted one is caught by the labels it implies."""
    rekey = _rekey(tmp_path)
    assert rekey.reproduces(_index(tmp_path, "full", IDENTIFIERS))
    assert not rekey.reproduces(_index(tmp_path, "shuffled", ["s1", "s0", "s3", "s2"]))


def test_write(tmp_path: Path) -> None:
    rekey = _rekey(tmp_path)
    written = rekey.write(_index(tmp_path, "full", IDENTIFIERS), tmp_path / "named.npz")

    held = np.load(written)
    assert [str(name) for name in held["studies"]] == ["s2", "s0"]
    assert np.array_equal(held["truth"], _truth())
    assert held["maps"].shape == (len(SEATS), len(TARGETS), GRID, GRID)


def test_write_refuses_an_order_that_does_not_reproduce_the_truth(tmp_path: Path) -> None:
    rekey = _rekey(tmp_path)
    with pytest.raises(ValueError, match="refusing to re-key"):
        rekey.write(_index(tmp_path, "shuffled", ["s1", "s0", "s3", "s2"]), tmp_path / "named.npz")
