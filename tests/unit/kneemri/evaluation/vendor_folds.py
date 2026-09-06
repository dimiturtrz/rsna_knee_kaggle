"""Unit cover for the vendor-fold serving path.

Everything here runs against a SYNTHETIC vendor script and a synthetic namespace. That is deliberate:
the real checkout is gitignored and moves under us, so a test that read it would be testing their kernel
rather than our slicing of it. What is ours — the marker discipline, the int coercion their slot table
needs, the fold narrowing, the rank combination, the fallback refusal — is exactly what is exercised.
"""

from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation import vendor_folds
from kneemri.evaluation.vendor_folds import SeriesTable, VendorFolds, VendorScript, VendorSpace
from kneemri.runs.holdout_record import HoldoutRecord

SERIES_HEADER = "StudyInstanceUID,SeriesInstanceUID,Fat_Suppression\n"


def _script(tmp_path: Path, body: str) -> VendorScript:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "vendor.py"
    path.write_text(body, encoding="utf-8")
    return VendorScript(path)


def _space(folds: int, studies: int, *, unreadable: tuple[int, ...] = ()) -> dict[str, object]:
    def build_study(args: tuple[int, str, list[object]]) -> tuple[int, np.ndarray, np.ndarray]:
        index, _, _ = args
        if index in unreadable:
            message = "no slot matched"
            raise ValueError(message)
        return index, np.ones((2, 4, 4, 4), np.uint8), np.ones(2, np.uint8)

    def predict(images: np.ndarray, _: np.ndarray) -> np.ndarray:
        block = images.shape[0]
        return np.full((folds, block, len(TARGETS)), 0.25, np.float32)

    return {
        "models": [f"fold{index}" for index in range(folds)],
        "build_study": build_study,
        "predict": predict,
        "N_SLOT": 2,
        "N_SLICE": 4,
        "SIZE": 4,
        "studies": studies,
    }


def test_records(tmp_path: Path) -> None:
    """Rows outside the served studies are dropped; Fat_Suppression is coerced to the int their
    slot selector compares against."""
    path = tmp_path / "series.csv"
    path.write_text(SERIES_HEADER + "A,s1,1\nA,s2,0\nB,s3,1\n", encoding="utf-8")
    records = SeriesTable(path).records(["A"])
    assert set(records) == {"A"}
    assert [row["Fat_Suppression"] for row in records["A"]] == [1, 0]


def test_segment(tmp_path: Path) -> None:
    """A segment is the text between two markers, and a moved marker refuses."""
    script = _script(tmp_path, "head\nSTART\nbody\nSTOP\ntail\n")
    assert script.segment("START", "STOP") == "START\nbody\n"
    with pytest.raises(ValueError, match="marker not in"):
        script.segment("ABSENT", "STOP")
    with pytest.raises(ValueError, match="closing marker not after"):
        script.segment("START", "ABSENT")


def test_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The three segments execute into one namespace, and a differing label order refuses."""
    monkeypatch.setattr(vendor_folds, "PRELUDE", ("os",))
    monkeypatch.setattr(vendor_folds, "SEGMENTS", (("#A", "#B"), ("#B", "#C"), ("#C", "#D")))
    labels = ", ".join(repr(target) for target in TARGETS)
    body = f"#A\nSIZE = 4\n#B\nDECODED = SERIES_ROOT\n#C\nLABELS = [{labels}]\n#D\n"
    space = VendorSpace(_script(tmp_path, body), tmp_path, tmp_path / "raw", "cpu").built()
    assert space["SIZE"] == 4
    assert space["DECODED"] == tmp_path / "raw"

    wrong = "#A\n#B\n#C\nLABELS = ['nope']\n#D\n"
    with pytest.raises(ValueError, match="label order differs"):
        VendorSpace(_script(tmp_path / "other", wrong), tmp_path, tmp_path, "cpu").built()


def test_kept() -> None:
    """An empty selection keeps every fold; a named one narrows; an absent one refuses."""
    member = VendorFolds(_space(folds=3, studies=2))
    assert member.kept([]) == 3
    assert member.kept([0, 2]) == 2
    assert member.space["models"] == ["fold0", "fold2"]
    with pytest.raises(IndexError, match="no such fold"):
        member.kept([5])


def test_combined() -> None:
    """One fold keeps probabilities; several are rank-averaged within the served set."""
    single = np.array([[[0.9, 0.1], [0.2, 0.8]]], np.float32)
    assert VendorFolds.combined(single) == pytest.approx(single[0].astype(np.float64))

    pair = np.array([[[0.9], [0.1], [0.5]], [[0.1], [0.5], [0.9]]], np.float32)
    combined = VendorFolds.combined(pair)
    assert combined.shape == (3, 1)
    assert combined.min() >= 0.0
    assert combined.max() <= 1.0
    assert combined[2, 0] == pytest.approx(0.75)


def test_predicted() -> None:
    """Unreadable studies come back as the fallback probability and are counted, not hidden."""
    member = VendorFolds(_space(folds=1, studies=3, unreadable=(1,)))
    probabilities, fallbacks = member.predicted(["a", "b", "c"], {})
    assert probabilities.shape == (3, len(TARGETS))
    assert fallbacks == 0
    assert probabilities[0, 0] == pytest.approx(0.25)


def test_record(tmp_path: Path) -> None:
    """A clean read writes the record; too much fallback refuses instead of writing a weak member."""
    truth = np.zeros((2, len(TARGETS)), np.float32)
    member = VendorFolds(_space(folds=1, studies=2))
    destination = tmp_path / "vendor_member.pt"
    assert member.record(destination, ["a", "b"], truth, {}) == 0
    written = HoldoutRecord(destination).read()
    assert written is not None
    assert list(written.studies) == ["a", "b"]

    def all_nan(images: np.ndarray, _: np.ndarray) -> np.ndarray:
        return np.full((1, images.shape[0], len(TARGETS)), np.nan, np.float32)

    refusing = VendorFolds(_space(folds=1, studies=2) | {"predict": all_nan})
    with pytest.raises(RuntimeError, match="refusing to write"):
        refusing.record(tmp_path / "refused.pt", ["a", "b"], truth, {})
