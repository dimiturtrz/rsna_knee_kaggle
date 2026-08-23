from pathlib import Path
from typing import override

import pytest

from kneemri.experiments.card_check import REPO, CardHolders, DeviceReport, Holder, main
from kneemri.process_table import Probe, ProcessTable

REPORT = DeviceReport("""+-----------------------------------------------------------------------------------------+
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            5872    C+G   ...Windows\\Explorer.EXE                  N/A   |
|    0   N/A  N/A           26432      G   ...Mozilla Firefox\\firefox.exe           N/A   |
|    0   N/A  N/A           55356      C   ...rs\\User\\miniconda3\\python.exe         N/A   |
+-----------------------------------------------------------------------------------------+
""")

OURS = f'55356 45256 "{REPO}\\.venv-arm\\Scripts\\python.exe" -u -m kneemri.training.train --epochs 12'
THEIRS = '40001 45256 "D:\\personal_projects\\biohub_kaggle\\.venv\\Scripts\\python.exe" -m biohub.train'
TABLE = ProcessTable(f"{OURS}\n{THEIRS}\n")


def _report(*pids: str) -> DeviceReport:
    rows = [f"|    0   N/A  N/A  {pid:>14}      C   ...python.exe   N/A   |" for pid in pids]
    return DeviceReport("\n".join(rows))


def test_holder_normalised():
    """Three tools spell one path three ways, and a check that matches literally stops seeing our jobs."""
    assert Holder.normalised("D:\\Repo\\Thing") == Holder.normalised("d:/repo/thing")


def test_device_report_compute_pids():
    """Only the `C` rows, which is the whole reason the human table is parsed instead of a CSV query.

    MEASURED 2026-08-23: a first cut tested `startswith("C")`, which is true of `C+G` too, and the
    check reported Explorer and two copies of Firefox as holders of the card.
    """
    assert REPORT.compute_pids() == ["55356"], "C+G is a browser drawing, not a job computing"


def test_device_report_compute_pids_ignores_a_report_with_no_process_table():
    """A driver that answers without a process section must read as an idle card, not raise."""
    assert DeviceReport("+---+\n| no processes are running |\n+---+").compute_pids() == []


def test_holder_ours():
    """Ours is decided by the repo root in the argv — never by the driver's own process name.

    MEASURED 2026-08-23: `nvidia-smi` named our training process `...User\\miniconda3\\python.exe`
    while the process table had the same pid running from this checkout's `.venv-arm`. Deciding on the
    driver's name would have filed our own job under the other project's.
    """
    assert Holder("55356", OURS).ours()
    assert not Holder("40001", THEIRS).ours(), "the other project shares this card by arrangement"
    assert not Holder("999", "").ours(), "a holder nothing can name is not claimed as ours"


def test_holder_ours_accepts_a_repo_spelled_the_other_way():
    assert Holder("55356", OURS.replace("\\", "/").upper()).ours()


def test_holder_line():
    assert Holder("55356", OURS).line().startswith("  OURS  pid   55356")
    assert "(not in the process table)" in Holder("999", "").line()


def test_card_holders_holders():
    """Ours sorted first, so the row that decides the verdict is the one at the top."""
    holders = CardHolders(_report("40001", "55356"), TABLE).holders()
    assert [holder.pid for holder in holders] == ["55356", "40001"]


def test_card_holders_ours():
    """Only the holders out of this checkout — the ones this check is entitled to count."""
    mine = CardHolders(_report("40001", "55356"), TABLE).ours()
    assert [holder.pid for holder in mine] == ["55356"], "the other project's job is not ours to count"


def test_card_holders_named():
    """The score that CHOOSES a table — the two pid namespaces are why it is a score and not a branch."""
    assert CardHolders(REPORT, TABLE).named() == 1
    assert CardHolders(REPORT, ProcessTable("")).named() == 0, "an MSYS table names no Windows pid"


def test_card_holders_faults():
    """Two of OURS is the fault. A busy card is the normal state here and blocking on it would be noise."""
    assert CardHolders(_report("55356", "40001"), TABLE).faults() == [], "one of ours beside theirs is fine"
    doubled = ProcessTable(f"{OURS}\n{OURS.replace('55356', '55999')}\n")
    assert CardHolders(_report("55356", "55999"), doubled).faults()[0].startswith("OVERSUBSCRIBED:")


def test_card_holders_faults_is_the_2026_08_23_incident():
    """Two `member_curve` scans on one card: the second `nohup` landed on a first that had not failed.

    The cost was the READING, not the card — contention stretched one member from three minutes to
    eleven, one step from being written down as a fact about the member.
    """
    scan = f"{REPO}\\.venv\\Scripts\\python.exe -m kneemri.evaluation.member_curve"
    both = ProcessTable(f"50108 1 {scan}\n53036 1 {scan}\n")
    assert CardHolders(_report("50108", "53036"), both).faults(), "two identical scans is exactly the shape"


def test_card_holders_lines():
    assert CardHolders(REPORT, TABLE).lines()[-1].startswith("OK:")
    assert "device at 30315 MiB" in CardHolders(REPORT, TABLE).lines("30315 MiB, 32607 MiB")[0]


def test_card_holders_lines_reports_a_shared_card_without_failing_it():
    """The lanes rule: another project's job is named and counted, and is not ours to kill."""
    shown = CardHolders(_report("55356", "40001"), TABLE).lines()
    assert any("not ours — shared by arrangement" in line for line in shown)
    assert shown[-1].startswith("OK:")


def test_card_holders_lines_empty():
    assert CardHolders(_report(), TABLE).lines()[-1] == "OK: the card is free"


def test_card_holders_crossed():
    """The table that names the most of the driver's pids wins, which needs no test of the platform."""
    blind = Probe("no-such-executable-anywhere", ())
    assert CardHolders.crossed(REPORT, (blind,)).named() == 0, "no probe leaves the holders unnamed, not unreported"
    assert CardHolders.crossed(REPORT, (blind,)).holders()[0].pid == "55356"


def test_main_exits_when_the_driver_cannot_be_read(monkeypatch: pytest.MonkeyPatch):
    """An unreadable driver must not leave the status at 0 — that reads as a pass in a log tail."""
    monkeypatch.setattr("kneemri.experiments.card_check.DEVICE", Probe("no-such-executable-anywhere", ()))
    monkeypatch.setattr("sys.argv", ["card_check"])
    with pytest.raises(SystemExit) as exit_code:
        main()
    assert exit_code.value.code == 1


def test_main_exits_on_a_fault(monkeypatch: pytest.MonkeyPatch):
    """An oversubscribed card must not leave the status at 0 either — same trap, and it costs a reading."""
    doubled = f"{OURS}\n{OURS.replace('55356', '55999')}\n"
    monkeypatch.setattr("kneemri.experiments.card_check.DEVICE", _Fixed(_report("55356", "55999").text))
    monkeypatch.setattr("kneemri.experiments.card_check.TOTAL", _Fixed(""))
    monkeypatch.setattr("kneemri.experiments.card_check.PROBES", (_Fixed(doubled),))
    monkeypatch.setattr("sys.argv", ["card_check"])
    with pytest.raises(SystemExit) as exit_code:
        main()
    assert exit_code.value.code == 2


class _Fixed(Probe):
    """A probe that answers a fixed string, so the exit-code tests need no card and spawn no process."""

    answer: str

    def __init__(self, answer: str) -> None:
        super().__init__("fixed", ())
        object.__setattr__(self, "answer", answer)

    @override
    def read(self) -> str:
        return self.answer


def test_repo_is_this_checkout():
    """Derived from the module's own location, because a written-down path is machine-specific."""
    assert (REPO / "kneemri" / "experiments" / "card_check.py").exists()
    assert isinstance(REPO, Path)
