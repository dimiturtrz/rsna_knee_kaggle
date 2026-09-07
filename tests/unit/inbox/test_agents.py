"""The lane switch: what `active` means, and the two ways it can lie.

Every test here writes its own registry into a tmp path — the module's real one names the live
mailboxes, and a test that toggled it would silence a lane for real.
"""

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "inbox"))

from agents import (  # pyrefly: ignore[missing-import]  (the sys.path line above is what resolves it)
    ACTIVE,
    FIELDS,
    STALE_SECONDS,
    Agent,
    Registry,
    Status,
    _joined,
    _split,
    unread,
    utc_now,
)

HEADER = _joined(list(FIELDS))
ROWS = (
    'main,"execution — card, kernels",1,,2026-09-07T21:40Z',
    "researcher,read-only audit,0,corpus offline,2026-09-07T21:40Z",
)


def _registry(tmp_path: Path) -> Path:
    path = tmp_path / "agents.csv"
    path.write_text("\n".join((HEADER, *ROWS)) + "\n", encoding="utf-8")
    return path


def test_parsed():
    """A quoted role holds a comma without becoming two fields."""
    agent = Agent.parsed(ROWS[0])

    assert agent.box == "main"
    assert agent.role == "execution — card, kernels"
    assert agent.active


def test_parsed_refuses_a_short_row():
    """A corrupt registry is a refusal — an implied default here silences or arms a lane by accident."""
    with pytest.raises(ValueError, match="holds 2 fields"):
        Agent.parsed("main,1")


def test_row():
    """A row round-trips, so a toggle rewrites the file without reshaping the rows it did not touch."""
    assert Agent.parsed(Agent.parsed(ROWS[1]).row()).note == "corpus offline"


def test_switched():
    """The switch carries its reason and a fresh stamp; everything else about the lane is untouched."""
    moved = Agent.parsed(ROWS[0]).switched(active=False, reason="card work only")

    assert not moved.active
    assert moved.note == "card work only"
    assert moved.updated_utc.endswith("Z")
    assert moved.role == "execution — card, kernels"


def test_read(tmp_path: Path):
    """Both lanes, with the header checked — a positional read of a reshaped file is the failure."""
    registry = Registry.read(str(_registry(tmp_path)))

    assert [agent.box for agent in registry.agents] == ["main", "researcher"]


def test_read_refuses_a_foreign_header(tmp_path: Path):
    path = tmp_path / "agents.csv"
    path.write_text("box,active\nmain,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="the columns are"):
        Registry.read(str(path))


def test_agent(tmp_path: Path):
    """An unknown box refuses rather than reading as an inactive lane, which would silence mail."""
    registry = Registry.read(str(_registry(tmp_path)))

    assert registry.agent("researcher").note == "corpus offline"
    with pytest.raises(ValueError, match="no lane named"):
        registry.agent("mastermind")


def test_active(tmp_path: Path):
    registry = Registry.read(str(_registry(tmp_path)))

    assert registry.active("main")
    assert not registry.active("researcher")


def test_toggled(tmp_path: Path):
    """One switch moves; every other row is left exactly as written."""
    registry = Registry.read(str(_registry(tmp_path))).toggled("main", active=False, reason="asleep")

    assert not registry.active("main")
    assert registry.agent("researcher").row() == ROWS[1]


def test_written(tmp_path: Path):
    """`.part` then replace, and the file round-trips through `read` — an off flips back on."""
    path = _registry(tmp_path)
    Registry.read(str(path)).toggled("researcher", active=True, reason="").written(str(path))

    assert not (tmp_path / "agents.csv.part").exists()
    assert Registry.read(str(path)).active("researcher")
    assert Registry.read(str(path)).agent("researcher").note == ""


def test_path():
    assert Status.path("main").endswith(os.path.join("main", ".status"))


def test_read_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A box that never armed reads None, which is what `--list` flags rather than a zeroed record."""
    monkeypatch.setattr("agents.SCRIPT_DIR", str(tmp_path))
    (tmp_path / "main").mkdir()

    assert Status.read("main") is None
    (tmp_path / "main" / ".status").write_text(json.dumps({"pid": 7, "armed_utc": "x", "delivered": 2}))
    held = Status.read("main")
    assert held is not None
    assert (held.pid, held.delivered) == (7, 2)


def test_written_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("agents.SCRIPT_DIR", str(tmp_path))
    (tmp_path / "main").mkdir()
    Status(pid=11, armed_utc="a", last_delivered_utc="b", delivered=3).written("main")

    held = Status.read("main")
    assert held is not None
    assert held.armed_utc == "a"


def test_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An arm older than a day is not evidence of a live listener — a listener dies with its session."""
    monkeypatch.setattr("agents.SCRIPT_DIR", str(tmp_path))
    (tmp_path / "main").mkdir()
    Status(pid=1, armed_utc="a", last_delivered_utc="", delivered=0).written("main")
    fresh = os.path.getmtime(Status.path("main"))

    assert not Status.stale("main", fresh)
    assert Status.stale("main", fresh + STALE_SECONDS + 1)
    assert Status.stale("researcher", fresh), "a box with no status is infinitely old, never fresh"


def test_unread(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """What an OFF lane is holding. A README is documentation, not a letter."""
    monkeypatch.setattr("agents.SCRIPT_DIR", str(tmp_path))
    (tmp_path / "main").mkdir()
    (tmp_path / "main" / "README.md").write_text("docs")
    (tmp_path / "main" / "20260907T2200Z-x-y.md").write_text("letter")

    assert unread("main") == 1
    assert unread("mastermind") == 0


def test_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The join, and both lies: active with no listener, and off with mail waiting."""
    monkeypatch.setattr("agents.SCRIPT_DIR", str(tmp_path))
    for box in ("main", "researcher"):
        (tmp_path / box).mkdir()
    (tmp_path / "researcher" / "20260907T2200Z-x-y.md").write_text("letter")
    registry = Registry.read(str(_registry(tmp_path)))

    lines = registry.table(os.path.getmtime(tmp_path))

    assert "STALE — flagged active, no live listener" in lines[0]
    assert "OFF, 1 letters waiting" in lines[1]


def test_split():
    assert _split('a,"b,c",d') == ["a", "b,c", "d"]


def test_joined():
    """Only a cell that would reshape the row is quoted, so a plain row stays diff-clean."""
    assert _joined(["a", "b,c"]) == 'a,"b,c"'


def test_utc_now():
    assert utc_now().endswith("Z")
    assert len(utc_now()) == len("2026-09-07T21:40Z")


def test_active_constant():
    assert ACTIVE == "1"
