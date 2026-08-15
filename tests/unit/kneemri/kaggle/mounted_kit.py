"""What Kaggle holds, without credentials or network — the kaggle call is faked and its argv asserted."""

import subprocess
from pathlib import Path

import pytest

from kneemri.kaggle.mounted_kit import STAMP, MountedKit


def test_recorded(monkeypatch: pytest.MonkeyPatch):
    """Fetches ONLY the stamp — the weights are gigabytes and answer nothing — and parses it."""
    seen: list[list[str]] = []

    def fake(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        (Path(command[command.index("-p") + 1]) / "kit.json").write_text('{"commit": "4faac3d", "dirty": false}')
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("kneemri.kaggle.mounted_kit.subprocess.run", fake)

    assert MountedKit("owner/kit").recorded() == {"commit": "4faac3d", "dirty": False}
    assert seen[0][1:4] == ["datasets", "download", "-f"]
    assert STAMP in seen[0], "the 46-byte stamp, not the dataset"
    assert "owner/kit" in seen[0]


def test_recorded_without_a_stamp(monkeypatch: pytest.MonkeyPatch):
    """A dataset predating the stamp reports None, which `KitCheck.behind` refuses rather than assumes."""
    monkeypatch.setattr(
        "kneemri.kaggle.mounted_kit.subprocess.run",
        lambda command, **_: subprocess.CompletedProcess(command, 0, "", ""),
    )

    assert MountedKit("owner/kit").recorded() is None
