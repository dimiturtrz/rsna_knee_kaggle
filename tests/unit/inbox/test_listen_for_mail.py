"""The refusal path: a lane switched off does not arm, and its mail stays where it is.

Only the switch is exercised here. The delivery loop wants a live filesystem race and a background
process, which is an integration concern, not a unit one.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "inbox"))

import agents  # pyrefly: ignore[missing-import]  (the sys.path line above is what resolves it)
import listen_for_mail  # pyrefly: ignore[missing-import]

HEADER = "box,role,active,note,updated_utc"


def _lane(tmp_path: Path, active: str, monkeypatch: pytest.MonkeyPatch) -> Path:
    registry = tmp_path / "agents.csv"
    registry.write_text(f"{HEADER}\nmain,execution,{active},asleep,2026-09-07T21:40Z\n", encoding="utf-8")
    monkeypatch.setattr(agents, "REGISTRY", str(registry))
    monkeypatch.setattr(agents, "SCRIPT_DIR", str(tmp_path))
    box = tmp_path / "main"
    box.mkdir()
    (box / "20260907T2200Z-x-y.md").write_text("letter", encoding="utf-8")
    return box


def test_refuse_if_inactive_exits_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Exit 2, not a silent return: 'not arming' must never read as 'armed and quiet'."""
    box = _lane(tmp_path, "0", monkeypatch)

    with pytest.raises(SystemExit) as refusal:
        listen_for_mail._refuse_if_inactive("main")

    assert refusal.value.code == 2
    assert (box / "20260907T2200Z-x-y.md").exists(), "off means silenced, never lost"
    assert not (box / "read").exists()


def test_refuse_if_inactive_passes_an_active_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _lane(tmp_path, "1", monkeypatch)

    listen_for_mail._refuse_if_inactive("main")


def test_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The arm writes its own runtime line, and a delivery accumulates onto it."""
    _lane(tmp_path, "1", monkeypatch)

    listen_for_mail._record("main", delivered=0)
    armed = agents.Status.read("main")
    listen_for_mail._record("main", delivered=2)
    after = agents.Status.read("main")

    assert armed is not None and armed.delivered == 0
    assert after is not None and after.delivered == 2
    assert after.armed_utc == armed.armed_utc, "a delivery is not a new arm"
    assert after.last_delivered_utc.endswith("Z")
