"""Every run on disk as one row: `python -m kneemri.experiments.runs_table`.

The ledger is prose written by hand, so what ran and what was written down drift apart and the only way
to see the drift was a sweep of greps. This table is GENERATED from what each run left — its record, its
checkpoint, its predictions — and committed, so the next drift is a diff rather than an investigation.

A CELL NOBODY RECORDED STAYS EMPTY. A run older than run identities has no id, a run without a metrics
file has no epochs, and neither is filled from a filename, a default or a neighbour: an empty cell is the
record saying it does not know, and a guessed one is a number the next reader cannot tell from a fact.
"""

import csv
import logging
from pathlib import Path

from core import cli
from kneemri.experiments.ledger_check import INTERPRETATIONS, LEDGER, LedgerCheck
from kneemri.runs.artefact import Artefact
from kneemri.runs.run_identity import RunIdentity
from kneemri.runs.run_record import RunRecord

logger = logging.getLogger(__name__)

TABLE = Path("experiments/runs.tsv")
COLUMNS = ("name", "run_id", "git", "epochs", "best_epoch", "best_auc", "checkpoint", "holdout", "ledgered")
UNKNOWN = ""
YES, NO = "yes", "no"
ANSWER = {True: YES, False: NO}


class RunsTable:
    """The runs under one models directory, each read from its own files, against the written record."""

    def __init__(self, models: Path, record: LedgerCheck) -> None:
        self.models = models
        self.record = record

    def names(self) -> list[str]:
        """Every run that left a checkpoint or a metrics record, rotated records excluded."""
        found = [*self.models.glob(f"*{Artefact.CHECKPOINT}"), *self.models.glob(f"*{Artefact.METRICS}")]
        return sorted({Artefact.stem(path) for path in found})

    def row(self, name: str) -> dict[str, str]:
        """One run's cells, each from the file that states it, empty where none does."""
        record = RunRecord(self.models / name)
        checkpoint = record.checkpoint()
        run = record.run() or {}
        identity = run.get("id") or (RunIdentity.of(checkpoint) if checkpoint.exists() else None)
        best, rows = record.best(), record.rows()
        return {
            "name": name,
            "run_id": str(identity or UNKNOWN),
            "git": str(run.get("git", UNKNOWN)),
            "epochs": str(len(rows)) if rows else UNKNOWN,
            "best_epoch": str(best["epoch"]) if best else UNKNOWN,
            "best_auc": f"{best['auc']:.4f}" if best else UNKNOWN,
            "checkpoint": ANSWER[checkpoint.exists()],
            "holdout": ANSWER[Artefact.HOLDOUT.beside(checkpoint).exists()],
            "ledgered": ANSWER[bool(self.record.named(name))],
        }

    def write(self, destination: Path) -> int:
        """Write every run's row as TSV and return how many there were."""
        rows = [self.row(name) for name in self.names()]
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)


def main() -> None:
    """Regenerate the runs table from what is on disk."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--out", type=Path, default=TABLE)
    arguments = command.parsed()
    record = LedgerCheck(LEDGER, *sorted(INTERPRETATIONS.rglob("*.md")))
    written = RunsTable(command.rooted(arguments).models(), record).write(arguments.out)
    logger.info("%d runs -> %s", written, arguments.out)


if __name__ == "__main__":
    main()
