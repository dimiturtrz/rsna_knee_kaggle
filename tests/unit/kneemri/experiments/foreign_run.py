"""The one environment difference bridged for a foreign script, and the script actually running."""

import time
from pathlib import Path

import pandas as pd
import pytest

from kneemri.experiments.foreign_run import DEFAULT_TAG, ForeignArtefacts, ForeignRun

TAG = "cellz"


def _run(script: Path) -> ForeignRun:
    """A run over a throwaway script with no arguments."""
    return ForeignRun(script=script, argv=())


def _writes(directory: Path, tag: str) -> str:
    """Source for a script that writes the one artefact his script's success is recognised by."""
    return f"open(r'{directory}' + '/raptor_ft_{tag}_swa.pt', 'w').write('x')\n"


def test_csv_instead(tmp_path: Path) -> None:
    """A parquet path reads the csv beside it; a csv reads itself; a missing csv is refused."""
    frame = pd.DataFrame({"StudyInstanceUID": ["a", "b"], "ACL": [0.1, 0.2]})
    frame.to_csv(tmp_path / "labels_llm_soft.csv", index=False)
    run = _run(tmp_path / "script.py")

    assert len(run.csv_instead(tmp_path / "labels_llm_soft.parquet")) == 2
    assert len(run.csv_instead(tmp_path / "labels_llm_soft.csv")) == 2

    with pytest.raises(FileNotFoundError, match="stands beside it"):
        run.csv_instead(tmp_path / "absent.parquet")


def test_ran(tmp_path: Path) -> None:
    """The script runs as `__main__`, sees its argv, and reads a parquet through the shim."""
    pd.DataFrame({"ACL": [0.3]}).to_csv(tmp_path / "labels.csv", index=False)
    script = tmp_path / "foreign.py"
    script.write_text(
        "import sys\n"
        "import pandas as pd\n"
        "assert __name__ == '__main__', __name__\n"
        f"frame = pd.read_parquet(r'{tmp_path / 'labels.parquet'}')\n"
        f"(open(r'{tmp_path / 'out.txt'}', 'w')).write(f'{{sys.argv[1]}} {{len(frame)}}')\n" + _writes(tmp_path, TAG),
        encoding="utf-8",
    )
    ForeignRun(script=script, argv=("--ckpt", "timm", "--tag", TAG)).ran()
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "--ckpt 1"


def test_tagged(tmp_path: Path) -> None:
    """The tag is read out of HIS argv, and a run that omits it inherits the default he would use."""
    script = tmp_path / "foreign.py"
    assert ForeignRun(script=script, argv=("--ckpt", "timm", "--tag", TAG)).tagged() == TAG
    assert ForeignRun(script=script, argv=("--ckpt", "timm")).tagged() == DEFAULT_TAG
    trailing = ForeignRun(script=script, argv=("--ckpt", "timm", "--tag"))
    assert trailing.tagged() == DEFAULT_TAG, "a trailing flag has no value to read"


def test_found(tmp_path: Path) -> None:
    """Only this tag's artefacts, and only those written at or after the instant asked about."""
    (tmp_path / f"raptor_ft_{TAG}_swa.pt").write_text("x", encoding="utf-8")
    (tmp_path / "raptor_ft_other.pt").write_text("x", encoding="utf-8")
    (tmp_path / f"notes_{TAG}.txt").write_text("x", encoding="utf-8")
    artefacts = ForeignArtefacts(tmp_path, TAG)

    assert [path.name for path in artefacts.found()] == [f"raptor_ft_{TAG}_swa.pt"]
    assert artefacts.found(since=time.time() + 60.0) == [], "nothing was written after the future"


def test_artefacts(tmp_path: Path) -> None:
    """Outputs land beside the script under the tag his argv named — never a second declaration."""
    run = ForeignRun(script=tmp_path / "foreign.py", argv=("--tag", TAG))
    assert run.artefacts() == ForeignArtefacts(tmp_path, TAG)


def test_ran_refuses_a_spent_tag(tmp_path: Path) -> None:
    """A tag already carrying artefacts is refused, and `--anyway` is what spends it deliberately.

    The refusal must come BEFORE the script runs: his script overwrites on open, so a guard that fired
    afterwards would be reporting a loss rather than preventing one.
    """
    script = tmp_path / "foreign.py"
    script.write_text(_writes(tmp_path, TAG), encoding="utf-8")
    (tmp_path / f"raptor_gold_{TAG}.npz").write_text("x", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already has"):
        ForeignRun(script=script, argv=("--tag", TAG)).ran()

    ForeignRun(script=script, argv=("--tag", TAG), anyway="re-running under the fixed labels").ran()
    assert (tmp_path / f"raptor_ft_{TAG}_swa.pt").exists()


def test_ran_refuses_a_silent_death(tmp_path: Path) -> None:
    """A script that returns having written nothing is a death, not a run, and must not exit zero."""
    script = tmp_path / "foreign.py"
    script.write_text("print('caught its own failure and returned')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="without writing any artefact"):
        ForeignRun(script=script, argv=("--tag", TAG)).ran()
