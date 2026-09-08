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

THE GUARD ON BOTH ENDS, AND WHY IT IS HERE AND NOT IN THE ARM. His script names its outputs from `--tag`
alone and overwrites them silently, and it writes no `ARM DONE` line, so the artefacts beside it are the
only evidence a foreign arm ran at all. A tag already carrying artefacts is refused (`--anyway '<why>'`
spends it deliberately), and a run that returns having written none RAISES: `runpy.run_path` returns
normally when the child dies inside a caught exception, which recorded two deaths as `returned 0` with an
idle card behind them while `queue_check` read OK.

WHAT THIS DOES NOT DO. It does not choose his arguments. `--ckpt timm` (his SSL checkpoint is
unpublished, so a run here prices ImageNet init and must be pre-registered as that), the corpus location
and the fold hold-out are the caller's, spelled in the arm.
"""

import logging
import runpy
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core import cli

logger = logging.getLogger(__name__)

PARQUET = ".parquet"
"""The suffix his labels are spelled with, and the only one the csv beside it stands in for."""

CSV = ".csv"

TAG = "--tag"
DEFAULT_TAG = "dev"
"""His own default (`external/.../knee-mri-training-the-twelve-finding-model.py:343`) — an arm that forgets
the flag writes over the LAST arm that forgot it, which is the collision this guard exists to catch."""

ARTEFACTS = ("raptor_ft_{tag}*.pt", "raptor_gold_{tag}*.npz", "raptor_oof_{tag}*.npz")


@dataclass(frozen=True)
class ForeignArtefacts:
    """The files his script writes for one `--tag`, beside the script itself (his `HERE`).

    A foreign arm has no `--out` and no `ARM DONE` line, so these files are the ONLY evidence it ran. That
    makes them both halves of the guard: their presence BEFORE a run means the tag is already spent, and
    their absence after one means the run died — which `runpy` reports as a clean return.
    """

    directory: Path
    tag: str

    def found(self, since: float = 0.0) -> list[Path]:
        """Every artefact of this tag last written at or after `since`, oldest name first."""
        matched = [path for pattern in ARTEFACTS for path in self.directory.glob(pattern.format(tag=self.tag))]
        return sorted(path for path in matched if path.stat().st_mtime >= since)


@dataclass(frozen=True)
class ForeignRun:
    """One execution of a foreign script: which file, the argv it is handed, and whether it may overwrite."""

    script: Path
    argv: tuple[str, ...]
    anyway: str = ""

    def tagged(self) -> str:
        """The `--tag` HIS argv carries, or the default he would fall back to — never declared twice here."""
        named = TAG in self.argv and self.argv.index(TAG) + 1 < len(self.argv)
        return self.argv[self.argv.index(TAG) + 1] if named else DEFAULT_TAG

    def artefacts(self) -> ForeignArtefacts:
        """Where this run's outputs land and under which tag."""
        return ForeignArtefacts(self.script.parent, self.tagged())

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
        """Execute the script as `__main__` with the shim in place, guarded on both ends by its artefacts.

        BEFORE: a tag whose artefacts already exist is refused, because his script overwrites them without
        a word and the overwritten run is one this project has usually already priced. `--anyway '<why>'`
        deliberately spends it. AFTER: a run that wrote nothing raises, because `runpy.run_path` returns
        normally when the child dies inside a caught exception and the chain then records `returned 0`
        over an idle card — twice in one evening, both read as OK by `queue_check`.
        """
        artefacts = self.artefacts()
        standing = artefacts.found()
        if standing and not self.anyway:
            message = (
                f"tag {artefacts.tag!r} already has {len(standing)} artefact(s) beside the script "
                f"({standing[0].name}, ...) - pass --anyway '<why>' to overwrite a priced run"
            )
            raise FileExistsError(message)
        started = time.time()
        was = pd.read_parquet
        pd.read_parquet = self.csv_instead  # type: ignore[assignment]
        sys.argv = [str(self.script), *self.argv]
        logger.info("running %s with %d argument(s)", self.script, len(self.argv))
        try:
            runpy.run_path(str(self.script), run_name="__main__")
        finally:
            pd.read_parquet = was
        written = artefacts.found(started)
        if not written:
            message = f"{self.script.name} returned without writing any artefact for tag {artefacts.tag!r}"
            raise RuntimeError(message)
        logger.info("tag %s wrote %d artefact(s)", artefacts.tag, len(written))


def main() -> None:
    """Run a foreign script unchanged, bridging the one environment difference it cannot see."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--script", type=Path, required=True, help="the foreign script to run")
    command.parser.add_argument("--anyway", default="", help="why this run may overwrite an existing tag")
    arguments, rest = command.parser.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ForeignRun(script=arguments.script, argv=tuple(rest), anyway=arguments.anyway).ran()


if __name__ == "__main__":
    main()
