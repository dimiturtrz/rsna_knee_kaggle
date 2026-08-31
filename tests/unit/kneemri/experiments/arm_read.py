"""ArmRead resolves an arm's declared control and refuses an arm that declares none."""

from pathlib import Path

import pytest

from kneemri.experiments.arm_header import ArmHeader
from kneemri.experiments.arm_read import ArmRead

CONTROLLED = """#!/usr/bin/env bash
# controlled.sh — an arm that names its denominator.
#
# bead: knee-test
# control  band_dense = 0.8578
run_train controlled $RECIPE
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


def test_uncontrolled(read: ArmRead) -> None:
    assert read.uncontrolled(["controlled", "silent"]) == ["silent"]


def test_report(read: ArmRead, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no denominator"):
        read.report("silent", tmp_path, None)


def test_arms(read: ArmRead) -> None:
    assert read.arms() == ["controlled", "silent"]
