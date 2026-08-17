"""A kernel's declared mounts, and whether they cover the package it is about to run."""

import json
from pathlib import Path

from kneemri.kaggle.kernel_sources import KernelSources


def _metadata(tmp_path: Path, **fields: list[str]) -> Path:
    written = tmp_path / "kernel-metadata.json"
    written.write_text(json.dumps({"id": "dimiturnt/test", **fields}), encoding="utf-8")
    return written


def test_covers(tmp_path: Path):
    """A model source matches exactly; a dataset source covers the directories inside it.

    The segment boundary is the point: `dimiturnt/dinov3-hf` must not be read as covering a
    differently-named `dimiturnt/dinov3-hf-v2`, which is a real mount that would not be attached.
    """
    sources = KernelSources(
        _metadata(
            tmp_path,
            model_sources=["metaresearch/dinov2/PyTorch/small/1"],
            dataset_sources=["dimiturnt/dinov3-hf"],
        )
    )
    assert sources.covers("metaresearch/dinov2/PyTorch/small/1")
    assert sources.covers("dimiturnt/dinov3-hf/vits16"), "an encoder names a directory inside the dataset"
    assert not sources.covers("metaresearch/dinov2/PyTorch/base/1")
    assert not sources.covers("dimiturnt/dinov3-hf-v2/vits16"), "a prefix that is not a path segment is not a mount"


def test_undeclared(tmp_path: Path):
    """Only the missing direction is an error — declaring more than a package needs costs mount time.

    This is the drift that adding a DINOv3 member to a DINOv2-only entry creates: every member verifies,
    every fingerprint agrees, and the kernel still cannot build one of the backbones.
    """
    sources = KernelSources(_metadata(tmp_path, model_sources=["metaresearch/dinov2/PyTorch/small/1"]))
    assert sources.undeclared(["metaresearch/dinov2/PyTorch/small/1"]) == []
    assert sources.undeclared(["dimiturnt/dinov3-hf/vits16"]) == ["dimiturnt/dinov3-hf/vits16"]

    bare = KernelSources(_metadata(tmp_path))
    assert bare.undeclared(["metaresearch/dinov2/PyTorch/small/1"]) == ["metaresearch/dinov2/PyTorch/small/1"], (
        "a metadata file naming no sources attaches nothing"
    )
