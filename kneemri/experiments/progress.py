"""What the card is doing RIGHT NOW: `python -m kneemri.experiments.progress`.

Every other reader here answers a question about a run that has FINISHED. `run_comparison` needs a
checkpoint and a holdout record, `ledger_check` needs the arm written down, `stale_gap` reads whole
curves. None of them can be asked the question a queue actually raises between arms — is the card
producing, how far along, and when does the next slot open — and so that question kept being answered
by a throwaway line of python over the metrics files, five separate times in one session, each one
re-deriving the same three numbers with a different bug in it. This is that line, committed.

IT IS DELIBERATELY NOT A RESULT READER. It quotes the best epoch because a queue decision sometimes
turns on whether a run has already peaked, and it quotes nothing else — no deltas, no control, no seat
value. An arm is read against its control by `run_comparison`, and a progress line that started
reporting deltas would become a second, thinner answer to a question that already has a good one.

PRODUCING IS DERIVED FROM THE RUN'S OWN CLOCK, not from the process table. `ps` answers whether a
python is alive, which is a different fact: a trainer that has wedged on a stalled read is alive and
producing nothing, and that state has been on this machine three times (`run_metrics.stalls` names the
epochs, 20.8 hours between them). An epoch row carries `started`, so the file itself can say whether
the next epoch is overdue — and it says so about a run this shell never launched, which `kill -0`
cannot do here at all (see `lib.sh:wait_for_pid`).
"""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from core.paths import DataRoot
from kneemri.runs.run_metrics import METRICS_RECORD, RunMetrics

logger = logging.getLogger(__name__)

OVERDUE_FACTOR = 2.0
"""How many median epochs may pass with nothing recorded before a run is called stopped.

Below the 4x `run_metrics.STALL_FACTOR` on purpose, and the two measure different things: a stall is
read AFTERWARDS off a completed row and wants to be certain, while this is read while the row is still
missing and wants to be early. The widest honest epoch-to-epoch spread on disk is 1.3x the median
(404-530 s), so 2x cannot be reached by a slow epoch, and a run that has genuinely stopped is called so
within one epoch of its cost rather than four.
"""

SECONDS_PER_HOUR = 3600.0


class ArmProgress:
    """One run's metrics file, read as PROGRESS rather than as a result."""

    def __init__(self, metrics: RunMetrics) -> None:
        # A SNAPSHOT, not a live view. The file is being appended to by the run it describes, so
        # re-reading it per question lets one reading disagree with the next: `producing` can see the
        # epoch that `line` then reports as missing. Reading it once fixes the moment being reported.
        self.name = metrics.name()
        self.rows = metrics.rows()
        self.seconds = metrics.epoch_seconds()
        self.best = metrics.best()
        recipe = metrics.recipe()
        self.planned = int(recipe["epochs"]) if recipe and "epochs" in recipe else 0
        """Epochs the recorded recipe asked for, or 0 when the file predates recipe lines — zero rather
        than a guess, because a denominator invented from the rows present reads 100% at every epoch."""

    def producing(self, now: float) -> bool:
        """Whether the next epoch is still due, judged from the last one's own start stamp."""
        if not self.rows or "started" not in self.rows[-1]:
            return False
        return now < self.rows[-1]["started"] + self.seconds * OVERDUE_FACTOR

    def remaining(self, now: float) -> float:
        """Seconds of card time left before this run's slot opens, 0.0 once it is not producing."""
        if not self.producing(now):
            return 0.0
        return max(self.planned - len(self.rows), 0) * self.seconds

    def line(self, now: float) -> str:
        """One run on one line: state, position, cost, and the peak so far."""
        peak = f"best ep {self.best['epoch']} {self.best['auc']:.4f}" if self.best else "no scored epoch"
        state = "RUNNING" if self.producing(now) else "stopped"
        left = self.remaining(now)
        eta = f", {left / SECONDS_PER_HOUR:.1f} h left" if left else ""
        return (
            f"  {self.name:<24} {state:<8} {len(self.rows)}/{self.planned or '?'} ep  "
            f"{self.seconds:.0f} s/ep  {peak}{eta}"
        )


def main() -> None:
    """Print every recorded run, the producing ones first."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    processed = DataRoot.from_config(arguments.paths).processed()
    now = datetime.now(UTC).timestamp()
    arms = [ArmProgress(RunMetrics(found)) for found in sorted(processed.glob(f"*{METRICS_RECORD}"))]
    producing = [arm for arm in arms if arm.producing(now)]
    logger.info("%d runs recorded, %d producing", len(arms), len(producing))
    for arm in producing:
        logger.info(arm.line(now))
    if not producing:
        logger.warning("THE CARD IS IDLE — no run has recorded an epoch inside %.1f of its own median", OVERDUE_FACTOR)


if __name__ == "__main__":
    main()
