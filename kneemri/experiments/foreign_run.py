"""Run a foreign training script verbatim under this repo's interpreter.

Invocation: `python -m kneemri.experiments.foreign_run --script <path> -- <its own args>`.

WHY A RUNNER AND NOT A COPY. `external/raptor/knee-mri-training-the-twelve-finding-model.py` is the
published script the dread member was trained by, and the point of running it is that it is HIS recipe,
not ours. Editing it to fit our environment would make the number a reading about our edit. So the file
is executed unchanged and exactly one environment difference is bridged here, where it is visible.

THE ONE SHIM. His labels arrive as a parquet and `pd.read_parquet` has no engine in this venv - neither
`pyarrow` nor `fastparquet` is a dependency of this project, and adding one means `uv` rewriting the
environment, which is the exit-127 failure that has already killed two arms mid-epoch. The same table
sits beside the parquet as a csv (his own export), so `pd.read_parquet` is pointed at it for the life of
the run. The shim is announced in the log: a silent one would make a labels swap invisible in exactly
the cell that prices labels.

WHAT THIS DOES NOT DO. It does not choose his arguments. `--ckpt timm` (his SSL checkpoint is
unpublished, so a run here prices ImageNet init and must be pre-registered as that), the corpus location
and the fold hold-out are the caller's, spelled in the arm.
"""

import logging
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core import cli

logger = logging.getLogger(__name__)

PARQUET = ".parquet"
"""The suffix his labels are spelled with, and the only one the csv beside it stands in for."""

CSV = ".csv"


@dataclass(frozen=True)
class ForeignRun:
    """One execution of a foreign script: which file, and the argv it is handed."""

    script: Path
    argv: tuple[str, ...]

    def csv_instead(self, path: str | Path) -> pd.DataFrame:
        """His parquet read, served from the csv of the same name when no parquet engine exists.

        Refuses rather than guesses when the csv is absent: a labels table silently missing is a run
        that trains on nothing and reports it as an explanation, which is what his own `_NO_LABELS`
        path does.
        """
        wanted = Path(path)
        if wanted.suffix != PARQUET:
            return pd.read_csv(wanted)
        beside = wanted.with_suffix(CSV)
        if not beside.exists():
            message = f"{wanted} needs a parquet engine and no {beside.name} stands beside it"
            raise FileNotFoundError(message)
        logger.info("reading %s in place of %s - no parquet engine in this environment", beside, wanted)
        return pd.read_csv(beside)

    def ran(self) -> None:
        """Execute the script as `__main__` with the shim in place and the caller's argv."""
        was = pd.read_parquet
        pd.read_parquet = self.csv_instead  # type: ignore[assignment]
        sys.argv = [str(self.script), *self.argv]
        logger.info("running %s with %d argument(s)", self.script, len(self.argv))
        try:
            runpy.run_path(str(self.script), run_name="__main__")
        finally:
            pd.read_parquet = was


def main() -> None:
    """Run a foreign script unchanged, bridging the one environment difference it cannot see."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--script", type=Path, required=True, help="the foreign script to run")
    arguments, rest = command.parser.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ForeignRun(script=arguments.script, argv=tuple(rest)).ran()


if __name__ == "__main__":
    main()
