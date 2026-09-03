"""ArmRead resolves an arm's declared control and refuses an arm that declares none."""

import json
from pathlib import Path

import pytest

from kneemri.experiments.arm_header import ArmHeader
from kneemri.experiments.arm_read import ArmRead

CONTROLLED = """#!/usr/bin/env bash
# controlled.sh — an arm that names its denominator.
#
# bead: knee-test
# control  band_dense = 0.8578
run_train controlled $RECIPE --out "$PROCESSED/controlled_s2026.pt"
"""

SILENT = """#!/usr/bin/env bash
# silent.sh — an arm from before the convention.
#
# bead: knee-test
run_train silent $RECIPE
"""


@pytest.fixture(name="read")
def _read(tmp_path: Path) -> ArmRead:
    (tmp_path / "controlled.sh").write_text(CONTROLLED, encoding="utf-8")
    (tmp_path / "silent.sh").write_text(SILENT, encoding="utf-8")
    return ArmRead(ArmHeader(tmp_path))


def test_control(read: ArmRead) -> None:
    assert read.control("controlled") == "band_dense"
    assert read.control("silent") is None


def test_produced(read: ArmRead) -> None:
    assert read.produced("controlled") == "controlled_s2026"
    assert read.produced("silent") == "silent"


def test_uncontrolled(read: ArmRead) -> None:
    assert read.uncontrolled(["controlled", "silent"]) == ["silent"]


def test_report(read: ArmRead, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no denominator"):
        read.report("silent", tmp_path, None)


def _run(models: Path, name: str, auc: float) -> None:
    per_label: dict[str, float] = {}
    row = {"epoch": 8, "loss": 0.1, "auc": auc, "seconds": 400, "per_label": per_label}
    (models / f"{name}.metrics.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_sibling(read: ArmRead, tmp_path: Path) -> None:
    """A control NAME with no run of its own is picked out of its siblings by the score it declared.

    THE SPLITS ARE THE POINT. `lr4e-5_ws` has three runs on disk and one of them is a `--fold -1` fit
    that has seen every holdout study, so choosing by suffix convention would silently read a fold-0
    arm against a control that cheated. Only the declared number says which run the header quoted.
    """
    models = tmp_path / "models"
    models.mkdir()
    _run(models, "lr4e-5_ws_s2026", 0.8525)
    _run(models, "lr4e-5_ws_cf", 0.8712)

    assert read.sibling("lr4e-5_ws", 0.8525, models).stem == "lr4e-5_ws_s2026"
    with pytest.raises(SystemExit, match="siblings: lr4e-5_ws_cf=0.8712 lr4e-5_ws_s2026=0.8525"):
        read.sibling("lr4e-5_ws", 0.8600, models)


def test_arms(read: ArmRead) -> None:
    assert read.arms() == ["controlled", "silent"]
