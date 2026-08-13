"""The kit-versus-tree check, on a real git repository built in a temp directory."""

import json
import shutil
import subprocess
from pathlib import Path

from kneemri.kit_check import KitCheck

GIT = shutil.which("git") or "git"


def _repo(tmp_path: Path) -> Path:
    """A repository with one commit touching a shipped package, and one touching only docs."""
    repo = tmp_path / "repo"
    (repo / "kneemri").mkdir(parents=True)
    run = lambda *command: subprocess.run([GIT, "-C", str(repo), *command], check=True, capture_output=True)  # noqa: E731, S603
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (repo / "kneemri" / "model.py").write_text("first", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "first")
    return repo


def _kit(tmp_path: Path, commit: str | None) -> Path:
    kit = tmp_path / "kit" / "kneemri_src"
    kit.mkdir(parents=True)
    if commit is not None:
        (kit / "kit.json").write_text(json.dumps({"commit": commit, "dirty": False}), encoding="utf-8")
    return kit.parent


def test_asked(tmp_path: Path):
    """A read-only git question, answered from the given repository."""
    repo = _repo(tmp_path)
    assert KitCheck(repo).asked("rev-parse", "--short", "HEAD")


def test_stamped(tmp_path: Path):
    """The recorded commit, and None for a kit built before the stamp existed."""
    repo = _repo(tmp_path)

    assert KitCheck(repo).stamped(_kit(tmp_path / "a", "abc1234")) == "abc1234"
    assert KitCheck(repo).stamped(_kit(tmp_path / "b", None)) is None


def test_drift(tmp_path: Path):
    """Commits the kit lacks IN THE SHIPPED PACKAGES, and silence when it is current.

    A docs-only commit must not fail the check: what matters is not that the kit differs from HEAD but
    that it differs in code a kernel runs. Failing on every commit would train the reader to ignore it.
    """
    repo = _repo(tmp_path)
    check = KitCheck(repo)
    current = check.asked("rev-parse", "--short", "HEAD")

    assert check.drift(_kit(tmp_path / "current", current)) == []
    assert "no commit stamp" in check.drift(_kit(tmp_path / "unstamped", None))[0]

    run = lambda *command: subprocess.run([GIT, "-C", str(repo), *command], check=True, capture_output=True)  # noqa: E731, S603
    (repo / "README.md").write_text("docs", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "docs only")

    assert check.drift(_kit(tmp_path / "after_docs", current)) == [], "a docs commit changes no kernel"

    (repo / "kneemri" / "model.py").write_text("second", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "the fix the kit would be missing")

    behind = check.drift(_kit(tmp_path / "after_code", current))
    assert len(behind) == 1
    assert "the fix the kit would be missing" in behind[0]
