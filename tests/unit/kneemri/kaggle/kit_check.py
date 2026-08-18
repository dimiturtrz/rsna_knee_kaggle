"""The kit-versus-tree check, on a real git repository built in a temp directory."""

import json
import shutil
import subprocess
from pathlib import Path

from kneemri.kaggle.kit_check import KitCheck

GIT = shutil.which("git") or "git"


def _repo(tmp_path: Path) -> Path:
    """A repository the check can resolve a kernel's import closure inside.

    The entrypoint and the one module it imports are what a commit has to touch to matter; `analysis.py`
    is imported by nothing and stands for every measurement module that used to refuse a kit.
    """
    repo = tmp_path / "repo"
    (repo / "kneemri" / "kaggle").mkdir(parents=True)
    run = lambda *command: subprocess.run([GIT, "-C", str(repo), *command], check=True, capture_output=True)  # noqa: E731, S603
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (repo / "kneemri" / "kaggle" / "kernel_run.py").write_text("from kneemri.model import Model\n", encoding="utf-8")
    (repo / "kneemri" / "model.py").write_text("first", encoding="utf-8")
    (repo / "kneemri" / "analysis.py").write_text("first", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "first")
    return repo


def _kit(tmp_path: Path, commit: str | None, *, dirty: bool = False) -> Path:
    kit = tmp_path / "kit" / "kneemri_src"
    kit.mkdir(parents=True)
    if commit is not None:
        (kit / "kit.json").write_text(json.dumps({"commit": commit, "dirty": dirty}), encoding="utf-8")
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
    """Commits the kit lacks IN THE FILES A KERNEL LOADS, and silence when it is current.

    Neither a docs commit nor a commit to a module outside the import closure may fail the check: what
    matters is not that the kit differs from HEAD but that it differs in code a kernel runs. Counting
    over the whole package refused three purely additive measurement commits on 2026-08-17, and a check
    that is always overridden is one that will be overridden the time it is right.
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

    (repo / "kneemri" / "analysis.py").write_text("measured", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "a measurement no kernel imports")

    assert check.drift(_kit(tmp_path / "after_analysis", current)) == [], "an unimported module ships inert"

    (repo / "kneemri" / "model.py").write_text("second", encoding="utf-8")
    run("add", "-A")
    run("commit", "-qm", "the fix the kit would be missing")

    behind = check.drift(_kit(tmp_path / "after_code", current))
    assert len(behind) == 1
    assert "the fix the kit would be missing" in behind[0]


def test_recorded(tmp_path: Path):
    """The whole stamp, so a dirty build can be refused as well as a stale one."""
    repo = _repo(tmp_path)

    assert KitCheck(repo).recorded(_kit(tmp_path / "a", "abc1234", dirty=True)) == {"commit": "abc1234", "dirty": True}
    assert KitCheck(repo).recorded(_kit(tmp_path / "b", None)) is None


def test_behind(tmp_path: Path):
    """The judgement takes a STAMP rather than a directory, so a mounted kit is judged by the same rule."""
    repo = _repo(tmp_path)
    check = KitCheck(repo)
    current = check.asked("rev-parse", "--short", "HEAD")

    assert check.behind({"commit": current, "dirty": False}) == []
    assert "no commit stamp" in check.behind(None)[0]


def test_drift_refuses_a_kit_built_from_a_dirty_tree(tmp_path: Path):
    """A dirty kit names a commit it is not, which is the unnameable version wearing a hash.

    One was uploaded and run on 2026-08-14: current against HEAD by its stamp, and carrying uncommitted
    code that no commit describes.
    """
    repo = _repo(tmp_path)
    check = KitCheck(repo)
    current = check.asked("rev-parse", "--short", "HEAD")

    assert check.drift(_kit(tmp_path / "clean", current)) == []
    dirty = check.drift(_kit(tmp_path / "dirty", current, dirty=True))
    assert len(dirty) == 1
    assert "DIRTY" in dirty[0]


def test_a_path_inside_the_repo_answers_the_same_as_its_root(tmp_path: Path):
    """The pathspec is relative to where git RUNS, so an anchor one level down matches NOTHING.

    Not hypothetical: `main` anchored at the package directory after the restructure moved this module
    into `kneemri/kaggle/`, so `-- kneemri core` meant `kneemri/kneemri` and `kneemri/core`, matched no
    commit, and called the mounted kit current while it was 53 shipped-package commits behind.
    """
    repo = _repo(tmp_path)
    behind_at = KitCheck(repo).asked("rev-parse", "--short", "HEAD")
    (repo / "kneemri" / "model.py").write_text("second", encoding="utf-8")
    subprocess.run([GIT, "-C", str(repo), "add", "-A"], check=True, capture_output=True)  # noqa: S603
    subprocess.run([GIT, "-C", str(repo), "commit", "-qm", "second"], check=True, capture_output=True)  # noqa: S603
    kit = _kit(tmp_path / "behind", behind_at)

    assert KitCheck(repo).drift(kit) == KitCheck(repo / "kneemri").drift(kit) != []
