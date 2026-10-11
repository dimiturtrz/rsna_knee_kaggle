from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from core.findings import STUDY
from kneemri.inference.foreign_folds import ForeignFolds

KERNEL = """\
CROP_MM = 10_000.0

CROP_MM = 130.0


def render(value):
    return value * CROP_MM


def serve():
    return SERIES_ROOT
"""

SERVING = (3, 20)


def _folds(tmp_path: Path) -> ForeignFolds:
    kernel = tmp_path / "their_kernel.py"
    kernel.write_text(KERNEL, encoding="utf-8")
    return ForeignFolds(
        kernel=kernel,
        checkpoints=tmp_path / "ckpt",
        series_root=tmp_path / "series",
        symbols=("CROP_MM", "render", "serve"),
        scope=SERVING,
    )


def test_module(tmp_path: Path) -> None:
    """The lifted module carries the serving program's constants and the injected DICOM root."""
    module = _folds(tmp_path).module()
    assert module.render(2.0) == 260.0
    assert module.serve() == tmp_path / "series", "SERIES_ROOT came from their Kaggle mount; here it is ours"


def test_records(tmp_path: Path) -> None:
    """Their `by`: one entry per requested study, its series rows as dicts, everything else dropped."""
    series = pd.DataFrame(
        {
            STUDY: ["a", "a", "b"],
            "SeriesInstanceUID": ["s1", "s2", "s3"],
            "Anatomical_Plane": ["Sagittal", "Coronal", "Axial"],
        }
    )
    by = ForeignFolds.records(series, ["a"])
    assert set(by) == {"a"}, "a study we did not ask for is not decoded"
    assert [row["SeriesInstanceUID"] for row in by["a"]] == ["s1", "s2"]


def test_read_studies(tmp_path: Path) -> None:
    """A holdout npz names the 909; a CSV names the corpus, deduplicated and in file order."""
    npz = tmp_path / "holdout.npz"
    np.savez(npz, studies=np.array(["b", "a"]))
    assert ForeignFolds.read_studies(npz) == ["b", "a"]
    csv = tmp_path / "train.csv"
    pd.DataFrame({STUDY: ["a", "a", "c"]}).to_csv(csv, index=False)
    assert ForeignFolds.read_studies(csv) == ["a", "c"], "a study listed twice is served once"


class _Net(torch.nn.Module):
    """Their `Net`'s shape: the encoder under `enc.`, one head of its own."""

    def __init__(self, encoder: torch.nn.Module, cond: str, n_meta: int, pool: str, **kwargs: object) -> None:
        super().__init__()
        self.enc = encoder
        self.head = torch.nn.Linear(2, 1)
        self.built = (cond, n_meta, pool, kwargs)


def _their_module() -> SimpleNamespace:
    return SimpleNamespace(
        timm=SimpleNamespace(create_model=lambda backbone, **kwargs: torch.nn.Linear(2, 2)),
        ViTSlotToken=lambda encoder, types: encoder,
        N_SLOT_TYPES=6,
        Net=_Net,
        DEV="cpu",
    )


def _checkpoint(directory: Path, name: str, state: dict[str, torch.Tensor], fold: int) -> None:
    cfg = {"backbone": "vit_small", "img": 64, "cond": "token", "pool": "mean"}
    torch.save({"cfg": cfg, "state_dict": state, "fold": fold}, directory / name)


def test_nets_builds_every_fold_and_binds_them_as_their_models(tmp_path: Path) -> None:
    """The head must load; the encoder may be left to timm's own init, which their asserts allowed."""
    folds = _folds(tmp_path)
    folds.checkpoints.mkdir()
    head = _Net(torch.nn.Linear(2, 2), "token", 0, "mean").head.state_dict()
    for fold in (1, 0):
        _checkpoint(folds.checkpoints, f"vit_f{fold}.pt", {f"head.{key}": value for key, value in head.items()}, fold)
    module = _their_module()
    built = folds.nets(module)  # pyrefly: ignore[bad-argument-type]
    assert len(built) == 2
    assert module.models is built
    assert module.CFG["backbone"] == "vit_small"
    assert all(not net.training for net in built), "served in eval mode"


def test_nets_refuses_a_checkpoint_the_lifted_classes_no_longer_fit(tmp_path: Path) -> None:
    folds = _folds(tmp_path)
    folds.checkpoints.mkdir()
    _checkpoint(folds.checkpoints, "vit_f0.pt", {"renamed.weight": torch.zeros(1, 2)}, 0)
    with pytest.raises(ValueError, match="drifted apart"):
        folds.nets(_their_module())  # pyrefly: ignore[bad-argument-type]


LABELS = ("acl", "mcl", "pcl")
"""Their label count, small enough to read by eye."""


def test_serve_keeps_every_fold_separate_and_every_study_in_its_place(tmp_path: Path) -> None:
    """Chunks reassemble in study order, and each fold's column block is its own, not a mean."""

    def build_study(job: tuple[int, str, list[dict[str, object]]]) -> tuple[int, np.ndarray, np.ndarray]:
        index, study, rows = job
        array = np.full((1, 1, 2, 2), int(study[1:]), np.uint8)
        return index, array, np.full(1, len(rows), np.uint8)

    def predict(images: np.ndarray, masks: np.ndarray) -> np.ndarray:
        per_study = images[:, 0, 0, 0, 0].astype(np.float32) + masks[:, 0] / 10
        return np.stack([np.repeat(per_study[:, None] * scale, len(LABELS), axis=1) for scale in (1.0, 2.0)])

    module = SimpleNamespace(
        models=[None, None], LABELS=LABELS, CHUNK=2, N_SLOT=1, N_SLICE=1, SIZE=2,
        build_study=build_study, predict=predict,
    )  # fmt: skip
    studies = ["s3", "s1", "s7"]
    out = _folds(tmp_path).serve(module, studies, {"s1": [{}, {}]})  # pyrefly: ignore[bad-argument-type]
    assert out.shape == (2, 3, len(LABELS))
    assert out[0, :, 0].tolist() == pytest.approx([3.0, 1.2, 7.0]), "a study with no rows still serves, on zeros"
    assert out[1, :, 0].tolist() == pytest.approx([6.0, 2.4, 14.0])
