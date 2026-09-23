import re
from pathlib import Path

import pytest

from kneemri.runs.run_identity import GIT_ENV, UNKNOWN, RunIdentity
from kneemri.runs.run_metrics import RunMetrics

SHA = "0123456789abcdef0123456789abcdef01234567"


def test_begun(tmp_path: Path):
    """A fresh run mints a time-ordered id, and names every weights file its recipe started from."""
    parent = RunMetrics(tmp_path / "parent.pt")
    parent.begin({}, run={"id": "parent-id"})
    recipe = {"init": str(tmp_path / "parent.pt"), "out": str(tmp_path / "child.pt"), "lr": "1e-4"}

    identity = RunIdentity.begun(recipe, tmp_path / "child.pt", resuming=False)

    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{8}", identity.id)
    assert identity.parents == {"init": {"path": str(tmp_path / "parent.pt"), "run": "parent-id"}}, (
        "its own --out is not a parent, and a plain flag is not weights"
    )


def test_begun_keeps_the_id_when_resuming(tmp_path: Path):
    """A resume continues ONE run; a new id would split its trajectory into two truncated ones."""
    RunMetrics(tmp_path / "run.pt").begin({}, run={"id": "first"})

    assert RunIdentity.begun({}, tmp_path / "run.pt", resuming=True).id == "first"
    assert RunIdentity.begun({}, tmp_path / "run.pt", resuming=False).id != "first"


def test_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The arm's exported sha wins; otherwise HEAD resolves through a loose ref or packed-refs."""
    monkeypatch.delenv(GIT_ENV, raising=False)
    assert RunIdentity.commit(tmp_path) == UNKNOWN, "no checkout, no claim"

    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "packed-refs").write_text(f"# pack-refs\n{SHA} refs/heads/main\n", encoding="utf-8")
    assert RunIdentity.commit(tmp_path) == SHA

    (git / "refs" / "heads").mkdir(parents=True)
    (git / "refs" / "heads" / "main").write_text("f" * 40 + "\n", encoding="utf-8")
    assert RunIdentity.commit(tmp_path) == "f" * 40, "a loose ref is newer than the packed one"

    monkeypatch.setenv(GIT_ENV, f"{SHA}+dirty")
    assert RunIdentity.commit(tmp_path) == f"{SHA}+dirty"


def test_of(tmp_path: Path):
    """A weights file's run id, read from its record; None for the ledger's older, id-less half."""
    RunMetrics(tmp_path / "run.pt").begin({}, run={"id": "abc"})

    assert RunIdentity.of(tmp_path / "run.pt") == "abc"
    assert RunIdentity.of(tmp_path / "never.pt") is None


def test_stamp():
    """Plain data only, so a `weights_only` load of the checkpoint reads it back."""
    stamp = RunIdentity(id="x", git=SHA, parents={"init": {"path": "p.pt", "run": None}}).stamp()

    assert stamp == {"id": "x", "git": SHA, "parents": {"init": {"path": "p.pt", "run": None}}}
