"""The script seat: anchors land exactly once, the mount runs unpatched, and the table — not the exit — decides."""

import subprocess
from pathlib import Path

import pytest

from kneemri.kaggle.script_seat import SUBMISSION, Anchor, ScriptSeat

FAKE = """import os
import sys
from pathlib import Path

WRITE = True
EXIT = 0
if WRITE:
    Path("submission.csv").write_text(os.environ.get("SEAT_VALUE", "plain"), encoding="utf-8")
sys.exit(EXIT)
"""
"""A scoring script in miniature: writes its table into the working directory, then exits with a status."""

FAILS_AFTER_WRITING = Anchor("EXIT = 0", "EXIT = 3")
WRITES_NOTHING = Anchor("WRITE = True", "WRITE = False")


def _fake(tmp_path: Path) -> tuple[Path, Path, Path]:
    mount, working, staging = (tmp_path / name for name in ("mount", "working", "staging"))
    for directory in (mount, working, staging):
        directory.mkdir()
    script = mount / "scorer.py"
    script.write_text(FAKE, encoding="utf-8")
    return script, working, staging


def test_applied() -> None:
    """One site is rewritten; none or two are refused; a declared two-site anchor rewrites both."""
    text = "a = 1\nb = 2\nb = 2\n"
    assert Anchor("a = 1", "a = 9").applied(text, "src") == "a = 9\nb = 2\nb = 2\n"
    assert Anchor("b = 2", "b = 7", sites=2).applied(text, "src") == "a = 1\nb = 7\nb = 7\n"
    with pytest.raises(ValueError, match=r"src spells 'c = 3' 0 times, not 1"):
        Anchor("c = 3", "c = 4").applied(text, "src")
    with pytest.raises(ValueError, match="2 times, not 1"):
        Anchor("b = 2", "b = 7").applied(text, "src")


def test_staged(tmp_path: Path) -> None:
    """No anchors runs the mount itself; anchors write a patched copy to staging and leave the mount alone."""
    script, _, staging = _fake(tmp_path)
    assert ScriptSeat(script=script, staging=staging).staged() == script

    staged = ScriptSeat(script=script, anchors=(FAILS_AFTER_WRITING,), staging=staging).staged()
    assert staged.parent == staging
    assert "EXIT = 3" in staged.read_text(encoding="utf-8")
    assert script.read_text(encoding="utf-8") == FAKE

    broken = ScriptSeat(script=script, anchors=(Anchor("EXIT = 0", "EXIT = ("),), staging=staging)
    with pytest.raises(SyntaxError):
        broken.staged()
    with pytest.raises(ValueError, match="0 times"):
        ScriptSeat(script=script, anchors=(Anchor("absent", "x"),), staging=staging).staged()


def test_run(tmp_path: Path) -> None:
    """The table is moved aside, the environment reaches the child, and only a present table is a success."""
    script, working, staging = _fake(tmp_path)

    aside = ScriptSeat(script=script, kept_as="nartaa_seat.csv", staging=staging).run(working)
    assert aside == working / "nartaa_seat.csv"
    assert aside.read_text(encoding="utf-8") == "plain"
    assert not (working / SUBMISSION).exists()

    tolerant = ScriptSeat(
        script=script,
        anchors=(FAILS_AFTER_WRITING,),
        environment={"SEAT_VALUE": "from-env"},
        staging=staging,
        exit_tolerant=True,
    )
    kept = tolerant.run(working)
    assert kept == working / SUBMISSION
    assert kept.read_text(encoding="utf-8") == "from-env"

    strict = ScriptSeat(script=script, anchors=(FAILS_AFTER_WRITING,), staging=staging)
    with pytest.raises(subprocess.CalledProcessError):
        strict.run(working)


def test_run_refuses_a_stale_table(tmp_path: Path) -> None:
    """A table an earlier step left at the output path is deleted first, so an empty run cannot borrow it."""
    script, working, staging = _fake(tmp_path)
    (working / SUBMISSION).write_text("stale", encoding="utf-8")
    silent = ScriptSeat(script=script, anchors=(WRITES_NOTHING,), staging=staging, exit_tolerant=True)
    with pytest.raises(RuntimeError, match="left no submission.csv"):
        silent.run(working)
