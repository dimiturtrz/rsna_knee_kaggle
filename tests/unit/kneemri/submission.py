import json
from dataclasses import dataclass
from pathlib import Path
from typing import override

import pandas as pd
import torch
from torch import Tensor

from kneemri.ensemble import Ensemble
from kneemri.members import MemberPackage
from kneemri.model import TARGETS, StudyModel
from kneemri.submission import Submission

from .conftest import write_series

SEED = 2026
DIM = 8
PATCHES = 16
SLOT_NAMES = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]


@dataclass
class _Output:
    last_hidden_state: Tensor


class _FakeBackbone(torch.nn.Module):
    """A stand-in encoder whose tokens vary PER DIMENSION with the input.

    An earlier version returned `base + mean(pixels)` — the same scalar added to every feature. The
    head begins with a LayerNorm, which removes exactly that, so the stand-in made the model's output
    independent of its pixels and no test could detect a change in pooling. A useful double has to
    perturb the feature vector's SHAPE, not just its level.
    """

    @override
    def forward(self, *, pixel_values: Tensor) -> _Output:
        n = pixel_values.shape[0]
        rows = pixel_values.mean(dim=(1, 3))[:, :DIM] if pixel_values.shape[2] >= DIM else None
        if rows is None:
            rows = pixel_values.mean(dim=(1, 3)).repeat(1, (DIM // pixel_values.shape[2]) + 1)[:, :DIM]
        base = torch.arange(PATCHES + 1, dtype=torch.float32).reshape(1, -1, 1) * 0.01
        return _Output(last_hidden_state=base + rows.reshape(n, 1, DIM))


def _competition(root: Path) -> Path:
    """A miniature competition mount: two studies, one series each, and the three CSVs."""
    root.mkdir(parents=True, exist_ok=True)
    identifiers = ["study_a", "study_b"]
    series_rows = []
    for identifier in identifiers:
        write_series(root / "test_series" / identifier / "sag", slices=4, size=32)
        series_rows.append({"StudyInstanceUID": identifier, "SeriesInstanceUID": "sag", "Anatomical_Plane": "Sagittal"})
    pd.DataFrame({"StudyInstanceUID": identifiers}).to_csv(root / "test.csv", index=False)
    pd.DataFrame(series_rows).to_csv(root / "test_series.csv", index=False)
    sample = {"StudyInstanceUID": identifiers} | {target: [0.5, 0.5] for target in TARGETS}
    pd.DataFrame(sample).to_csv(root / "sample_submission.csv", index=False)
    return root


def _submission(tmp_path: Path) -> Submission:
    device = torch.device("cpu")
    torch.manual_seed(SEED)
    model = StudyModel(_FakeBackbone(), DIM, len(SLOT_NAMES))
    package_root = tmp_path / "weights"
    package_root.mkdir()
    torch.save(model.state_dict(), package_root / "m.pt")
    config = {
        "img": 32,
        "slices": 3,
        "group": 3,
        "crop_mm": 130.0,
        "band": [0.2, 0.8],
        "slots": SLOT_NAMES,
        "rules": {"order": "normal", "lat": "centre", "slot_fallback": False, "decode_fill": "nearest"},
        "variant": "small",
        "pool": "cls_mean",
    }
    (package_root / "manifest.json").write_text(
        json.dumps(
            {
                "members": [
                    {
                        "id": "m",
                        "file": "m.pt",
                        "fold": 0,
                        "config": config,
                        "fingerprint": model.fingerprint(32, 3, device).tolist(),
                        "holdout": 0.8,
                        "annot": 0.8,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    package = MemberPackage(package_root)
    ensemble = Ensemble(package, {"m": model}, device)
    working = tmp_path / "working"
    working.mkdir()
    return Submission(ensemble, _competition(tmp_path / "comp"), working)


def test_fallback(tmp_path: Path):
    """A scoreable file exists BEFORE any prediction, so a later crash still leaves something."""
    submission = _submission(tmp_path)
    submission.fallback()
    written = pd.read_csv(submission.working / "submission.csv")
    assert list(written.columns) == ["StudyInstanceUID", *TARGETS]
    assert len(written) == 2


def test_decode(tmp_path: Path):
    """A chunk of studies becomes one batched bag and its presence mask."""
    submission = _submission(tmp_path)
    images, mask = submission.decode(["study_a", "study_b"], {"sag": "Sagittal"}, slices=3)
    assert images.shape[0] == 2
    assert mask.shape == (2, len(SLOT_NAMES))
    assert mask[:, 0].tolist() == [1.0, 1.0], "the sagittal fluid slot is filled for both"


def test_predict(tmp_path: Path):
    """Every study gets one probability per target, in [0, 1]."""
    submission = _submission(tmp_path)
    combined = submission.predict(["study_a", "study_b"], {"sag": "Sagittal"}, slices=3)
    assert combined.shape == (2, len(TARGETS))
    assert combined.min() >= 0.0
    assert combined.max() <= 1.0


def test_run(tmp_path: Path):
    """The whole path writes a submission whose rows are the test studies, in order."""
    submission = _submission(tmp_path)
    frame = submission.run(slices=3)
    written = pd.read_csv(submission.working / "submission.csv")
    assert written["StudyInstanceUID"].tolist() == ["study_a", "study_b"]
    assert list(written.columns) == ["StudyInstanceUID", *TARGETS]
    assert len(frame) == 2


def test_chunks(tmp_path: Path):
    """Chunks arrive already on the ensemble's device, so the predictor never has to move them."""
    submission = _submission(tmp_path)
    produced = list(submission.chunks(["study_a", "study_b"], {"sag": "Sagittal"}, slices=3))
    assert len(produced) == 1, "two studies fit in one chunk"
    images, mask = produced[0]
    assert images.device.type == submission.device.type
    assert images.shape[0] == 2
