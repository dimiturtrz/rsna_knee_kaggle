import pytest

from kneemri.process_table import Probe, ProcessTable

POSIX = "User 107665 107663 0 09:12 ?  00:00:01 bash experiments/after.sh ARM DONE aug_long"
WINDOWS = '55356 45256 "D:\\repo\\.venv\\Scripts\\python.exe" -u -m kneemri.training.train --epochs 12'


def test_probe_read():
    """A probe whose executable is not on THIS interpreter's PATH answers empty rather than raising.

    MEASURED 2026-08-20: `shutil.which("ps")` returns None from the Python process this check runs in
    — the MSYS `ps` lives in git-bash's `/usr/bin`, off the Windows PATH — while `ps -ef` works fine in
    the shell that launches it. That gap is what made the check silently pass on the only machine it
    runs on, so the absent case is a test rather than an assumption.
    """
    assert Probe("no-such-executable-anywhere", ()).read() == ""


def test_process_table_read():
    """An empty table reads as 'nothing is running' and would strand every queued arm, so it is checked.

    Asserted on ROW COUNT, because that is the only thing TRUE OF BOTH PROBES. `ps -ef` prints a
    headed table of the MSYS process table; the PowerShell probe prints bare command lines with no
    header at all, of every Windows process. Nothing else is common to the two.

    It used to assert that `python` appeared, and that was wrong twice over. MSYS `ps` enumerates the
    MSYS process table ONLY, and pytest runs as a native Windows process — so under the first probe the
    assertion held exactly as long as something unrelated had launched a python through bash, and failed
    the moment a gate session was started directly. A test measuring its own launcher, not the code.

    SKIPS RATHER THAN FAILS WHEN NO PROBE SPAWNS, and that is not a silent pass. On 2026-08-20 a
    pre-push hook running beside a training arm, a CPU witness over nineteen members and a full gate
    session failed fourteen tests with `exit status 3221225794` — Windows `STATUS_DLL_INIT_FAILED`,
    the OS refusing to start any new process. The same tree passed serially minutes later. That is a
    fact about the machine's load, not about this code, and a check that cannot spawn `ps` has not
    measured that `ps` is broken. The condition this test guards is enforced where it matters instead:
    both `queue_check.main` and `card_check.main` EXIT NON-ZERO on an unreadable table, so a genuinely
    broken probe fails the check every time it is run for real, while a momentarily overloaded machine
    no longer blocks a push.
    """
    found = ProcessTable.read().text
    if not found:
        pytest.skip("no probe spawned on this machine right now — see test_process_table_read_no_probe_is_empty")
    populated = [line for line in found.splitlines() if line.strip()]
    assert len(populated) > 1, "a machine running this test is running more than one process"


def test_process_table_read_no_probe_is_empty():
    """No probe means no answer — and the caller must be told, not handed a table with nothing in it."""
    assert ProcessTable.read((Probe("no-such-executable-anywhere", ()),)).text == ""


def test_process_table_parentage():
    """The first two whole numbers, which is the one field order both probes agree on."""
    assert ProcessTable.parentage(POSIX) == ("107665", "107663")
    assert ProcessTable.parentage(WINDOWS) == ("55356", "45256")
    assert ProcessTable.parentage("bash with no ids at all") == ("", ""), "an unparseable line keeps the process"


def test_process_table_command():
    """Everything after the parent id — the argv a report shows, without the ids already in its own column."""
    argv = ProcessTable.command(WINDOWS)
    assert argv.startswith('"D:\\repo\\.venv\\Scripts\\python.exe" -u -m kneemri.training.train')
    assert "experiments/after.sh" in ProcessTable.command(POSIX)
    assert ProcessTable.command("no ids here") == "no ids here", "a line this cannot parse is still worth showing"


def test_process_table_commands():
    """Each pid mapped to its argv, which is what both crossings look a holder up in."""
    found = ProcessTable("\n".join([POSIX, WINDOWS])).commands()
    assert set(found) == {"107665", "55356"}
    assert found["55356"].startswith('"'), "the two ids are the row's own columns, not part of its argv"
