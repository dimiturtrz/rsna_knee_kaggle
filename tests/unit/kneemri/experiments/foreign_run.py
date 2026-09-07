"""The one environment difference bridged for a foreign script, and the script actually running."""

from pathlib import Path

import pandas as pd
import pytest

from kneemri.experiments.foreign_run import ForeignRun


def _run(script: Path) -> ForeignRun:
    """A run over a throwaway script with no arguments."""
    return ForeignRun(script=script, argv=())


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
        f"(open(r'{tmp_path / 'out.txt'}', 'w')).write(f'{{sys.argv[1]}} {{len(frame)}}')\n",
        encoding="utf-8",
    )
    ForeignRun(script=script, argv=("--ckpt", "timm")).ran()
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "--ckpt 1"
