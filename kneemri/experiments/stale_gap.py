"""What `--patience` would have to be, derived from every run's own curve instead of chosen.

`python -m kneemri.experiments.stale_gap`

`kneemri.training.patience` argues the lever and sizes the prize: the twelve-epoch OneCycle peaks at
epoch 6 and spends five more producing checkpoints that are thrown away, roughly 42% of an arm's
wall-clock. It is off by default and no arm passes the flag, so that 42% is still being paid.

WHAT IS MISSING IS THE NUMBER, and it cannot be picked. A stopper that fires too early costs exactly
the arms that converge late — the module names three (`intensity392` at 9, DINOv3-small at 11 of 15,
DINOv3 at 224 px at 8) and a blanket "stop at 7" would have truncated all of them. So the safe value is
a property of the corpus of curves already on disk: the longest run of NON-improving epochs that was
nonetheless followed by an improvement. Patience must exceed that, or it stops a run that was about to
get better. This module measures it rather than asserting it.

IT ALSO PRICES THE OTHER SIDE. `wasted` is the epochs each run spent after its own best, which is what
the flag would recover. A value derived from the first column is only worth setting if the second one
is large, and both are read off the same files in the same pass.

THE FLAG IS NOT FOR EVERY ARM, and the reason belongs next to the number. An arm whose question is the
SCHEDULE — `band_dense_long` asking whether 24 epochs beat 12, `aug_long` asking whether a regulariser
moves the peak late — is measuring the shape of the tail, and a stopper deletes the measurement.
Patience pays on arms that ask what a knob scores at a fixed schedule; on shape arms it is a blindfold.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from core import cli
from kneemri.runs.artefact import Artefact

logger = logging.getLogger(__name__)

METRICS_SUFFIX = ".metrics.jsonl"


@dataclass(frozen=True)
class StaleRun:
    """One run's curve, read for the two numbers a patience value is a trade between."""

    name: str
    epochs: int
    best_epoch: int
    survived: int
    """The longest non-improving streak this run recovered from — patience must EXCEED it."""
    wasted: int
    """Epochs run after the best one, whose checkpoints were discarded — what the flag recovers."""

    @classmethod
    def of(cls, path: Path) -> "StaleRun | None":
        """One metrics file as a curve, or None when it holds no scored epoch.

        A run with no holdout (`--fold -1`) and a recipe-only file both land here, and neither is an
        error: the sweep is over a processed directory that holds probes and final fits beside arms.
        """
        scores = cls._scores(path)
        if not scores:
            return None
        best = max(range(len(scores)), key=lambda epoch: scores[epoch])
        return cls(
            name=Artefact.stem(path),
            epochs=len(scores),
            best_epoch=best,
            survived=cls._survived(scores),
            wasted=len(scores) - 1 - best,
        )

    @staticmethod
    def _scores(path: Path) -> list[float]:
        """The holdout mean of each epoch line, in order, skipping the recipe header."""
        scored = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if "auc" in record:
                scored.append(float(record["auc"]))
        return scored

    @staticmethod
    def _survived(scores: list[float]) -> int:
        """The longest stretch of non-improving epochs that a LATER improvement recovered from.

        Counted against the running best rather than against the previous epoch: a stopper watches the
        best, so a curve that sags and climbs back to below its own peak never resets the counter.
        """
        longest = 0
        stale = 0
        best = scores[0]
        for score in scores[1:]:
            if score > best:
                best = score
                longest = max(longest, stale)
                stale = 0
            else:
                stale += 1
        return longest


class StaleGaps:
    """Every metrics file in one processed directory, read as curves."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def runs(self) -> list[StaleRun]:
        """Each scored run, worst-wasting first, so the prize is at the top of the table."""
        found = [StaleRun.of(path) for path in sorted(self.directory.glob(f"*{METRICS_SUFFIX}"))]
        return sorted([run for run in found if run is not None], key=lambda run: -run.wasted)

    def saving(self, patience: int) -> int:
        """Epochs this corpus would NOT have run at that patience — the prize, not the waste.

        `wasted` is the ceiling and nobody can have it: a stopper still has to sit through `patience`
        non-improving epochs before it fires, so every run pays that much of its own tail back. Quoting
        the waste as the saving overstates the flag by exactly the value being set, which is the whole
        of the prize on a twelve-epoch arm.
        """
        return sum(max(0, run.wasted - patience) for run in self.runs())

    def lines(self) -> list[str]:
        """The table, then the two numbers the flag is a trade between, then what it actually buys."""
        runs = self.runs()
        if not runs:
            return [f"no metrics files under {self.directory}"]
        survived = max(run.survived for run in runs)
        recoverer = next(run for run in runs if run.survived == survived)
        wasted = sum(run.wasted for run in runs)
        rows = [f"{len(runs)} runs under {self.directory}", "  run  epochs  best  survived  wasted"]
        rows += [f"  {run.name}  {run.epochs}  {run.best_epoch}  {run.survived}  {run.wasted}" for run in runs]
        safe = survived + 1
        return [
            *rows,
            f"longest recovered stall: {survived} epochs ({recoverer.name}), so --patience must exceed it",
            f"epochs run past a best: {wasted} across {len(runs)} runs, {wasted / len(runs):.1f} an arm",
            f"at --patience {safe}, the safe value, that corpus saves {self.saving(safe)} epochs "
            f"({self.saving(safe) / wasted:.0%} of the waste)",
        ]


def main() -> None:
    """Derive a patience value from the curves already on disk."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--directory", type=Path, required=True, help="a processed directory of runs")
    arguments = command.parsed()
    for line in StaleGaps(arguments.directory).lines():
        logger.info(line)


if __name__ == "__main__":
    main()
