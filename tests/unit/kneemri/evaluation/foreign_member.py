"""Serving someone else's checkpoint through their decode, checked against a stand-in for their script.

The stand-in is the point rather than a shortcut. Their real script needs pydicom, a 292 MB checkpoint
and a DICOM tree, none of which a unit test may touch — and every failure this module exists to prevent
is about the SEAM (label order, fallback accounting, the source stamp), not about their arithmetic.
"""

from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.foreign_member import FALLBACK_SHARE, Decode, ForeignMember, Records
from kneemri.runs.holdout_record import FOREIGN, UNKNOWN_EPOCH, HoldoutRecord

STUDIES = ["study-a", "study-b", "study-c", "study-d"]

STAND_IN = """
LAB = {labels}
ARMS = [{{"file": "x.pt", "arch": "coatnet", "res": 384, "w": 1.0}}]
K_EVAL = 42
NORM = "imagenet"


def _make_reader():
    return ("order", "read", "crop")


def load_model(path, arch, res, device, ngpu=1):
    return ("model", res)


def build_study(study, rows, series_root, reader):
    if study in {failing}:
        raise OSError("unreadable series")
    return ("volume", "mask")


def eval_windows(volume, mask, k, res, norm):
    return "windows"


def infer_probs(model, windows, device):
    return [0.9] * len(LAB)
"""


def stand_in(tmp_path: Path, failing: tuple[str, ...] = (), labels: list[str] | None = None) -> Path:
    """Write a module with their script's shape, naming which studies fail to decode."""
    script = tmp_path / "raptor-stand-in.py"
    script.write_text(STAND_IN.format(labels=repr(labels or list(TARGETS)), failing=repr(set(failing))))
    return script


def member(tmp_path: Path, failing: tuple[str, ...] = (), labels: list[str] | None = None) -> ForeignMember:
    return ForeignMember(
        module=Decode(stand_in(tmp_path, failing, labels)).loaded(),
        weights=tmp_path / "weights.pt",
        series_root=tmp_path / "series",
        device="cpu",
    )


def rows() -> Records:
    """Their `ser_records`, in the shape `csv.DictReader` produces — str keys, str values, no pandas."""
    return {study: [{"SeriesInstanceUID": "s", "Anatomical_Plane": "Sagittal"}] for study in STUDIES}


def test_loaded(tmp_path: Path) -> None:
    """A hyphenated filename is not an import, which is the whole reason this goes through importlib."""
    module = Decode(stand_in(tmp_path)).loaded()
    assert list(module.LAB) == list(TARGETS)
    assert module.K_EVAL == 42


def test_loaded_refuses_a_path_that_is_not_a_module(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, OSError)):
        Decode(tmp_path / "absent.py").loaded()


def test_checked(tmp_path: Path) -> None:
    """Matching label orders pass through unchanged — the module, not a copy."""
    module = Decode(stand_in(tmp_path)).loaded()
    assert Decode.checked(module) is module


def test_checked_refuses_a_different_label_order(tmp_path: Path) -> None:
    """THE ONE MISMATCH NOTHING DOWNSTREAM CAN SEE. Served under a permuted order the member is wrong
    on all twelve findings and still writes a well-formed file that scores near chance — which reads as
    a weak member rather than a broken seam."""
    swapped = [*list(TARGETS)[1:], TARGETS[0]]
    module = Decode(stand_in(tmp_path, labels=swapped)).loaded()
    with pytest.raises(ValueError, match="label order differs"):
        Decode.checked(module)


def test_predicted(tmp_path: Path) -> None:
    probabilities, fallbacks = member(tmp_path).predicted(STUDIES, rows())
    assert probabilities.shape == (len(STUDIES), len(TARGETS))
    assert fallbacks == 0
    assert probabilities == pytest.approx(0.9)


def test_predicted_counts_a_fallback_instead_of_hiding_it(tmp_path: Path) -> None:
    """A failed study answers 0.5 on every finding, which is invisible in every number after this point:
    it is indistinguishable from a genuinely uncertain model. The COUNT is what tells them apart."""
    probabilities, fallbacks = member(tmp_path, failing=("study-b",)).predicted(STUDIES, rows())
    assert fallbacks == 1
    assert probabilities[1] == pytest.approx(0.5)
    assert probabilities[0] == pytest.approx(0.9)


def test_record(tmp_path: Path) -> None:
    """The stamp is the point: a foreign array must not be readable as one of ours."""
    truth = np.zeros((len(STUDIES), len(TARGETS)), np.float32)
    assert member(tmp_path).record(tmp_path / "raptor.pt", STUDIES, truth, rows()) == 0
    reading = HoldoutRecord(tmp_path / "raptor.pt").read()
    assert reading is not None
    assert reading.source == FOREIGN, "neither trained by us nor re-derived from our weights"
    assert reading.epoch == UNKNOWN_EPOCH, "a foreign checkpoint has no epoch of ours to claim"
    assert reading.studies == STUDIES


def test_record_refuses_a_run_that_is_mostly_fallback(tmp_path: Path) -> None:
    """Above the share the number would be about the reader rather than the weights — and a member
    priced on that is the harness bug this project keeps paying for."""
    truth = np.zeros((len(STUDIES), len(TARGETS)), np.float32)
    failing = tuple(STUDIES[: int(len(STUDIES) * FALLBACK_SHARE) + 1])
    with pytest.raises(RuntimeError, match="refusing to write"):
        member(tmp_path, failing=failing).record(tmp_path / "raptor.pt", STUDIES, truth, rows())
    assert not (tmp_path / "raptor.holdout.npz").exists(), "a refusal must not leave a partial record"
