"""One machine-readable line per epoch, so runs can be compared without grepping prose."""

import json
import logging
import math
from pathlib import Path
from typing import NotRequired, TypedDict

import torch

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024**3


class EpochRow(TypedDict):
    """One epoch as it is recorded. A TypedDict rather than `Any` because this IS the schema.

    `vram_peak_gb` is filled in by the recorder rather than the caller — the peak is only readable
    from the allocator, and only before it is reset.
    """

    epoch: int
    loss: float
    auc: float
    seconds: int
    per_label: dict[str, float]
    vram_peak_gb: NotRequired[float]


class RunMetrics:
    """A run's per-epoch record, appended as JSONL beside its checkpoint.

    Comparing arms used to mean grepping `best holdout` out of a dozen log files scattered across two
    directories, by hand, and transcribing the numbers into a markdown ledger. That is how a result
    gets misattributed, and it already happened here: a "new best" that turned out to be a seed change
    rather than the depth change it was credited to.

    JSONL rather than a tracking server: nothing to run, it diffs, it greps, and pandas reads it in one
    line. The per-epoch CURVE is the part that matters and the part a final score throws away — whether
    a run was still climbing when it stopped, or peaked early and gave it back, is what decides the
    next arm's epoch budget. Both of those have already changed a decision in this project.

    Peak VRAM goes in because it decides what else can share the card, and it cannot be recovered
    afterwards.
    """

    def __init__(self, destination: Path) -> None:
        # IDEMPOTENT, because both spellings are natural at a command line: the TRAINER passes its
        # checkpoint path and an analysis reaches for the metrics file it can see on disk. Appending
        # blindly turned the second into `.metrics.metrics.jsonl`, which then reported "recorded no
        # scored epoch" — an error that describes a real state and hides a typo.
        self.path = (
            destination if destination.name.endswith(".metrics.jsonl") else destination.with_suffix(".metrics.jsonl")
        )

    def record(self, row: EpochRow) -> None:
        """Append one epoch's row, attach the VRAM peak, and report the per-label scores.

        It logs as well as records so the caller has ONE statement per epoch rather than two that can
        drift apart — the log line and the stored row should never be able to disagree.
        """
        if torch.cuda.is_available():
            row["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / BYTES_PER_GB, 2)
            torch.cuda.reset_peak_memory_stats()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        logger.info("    %s", {name: round(value, 3) for name, value in row["per_label"].items()})

    def rows(self) -> list[EpochRow]:
        """Everything recorded so far, for whoever is comparing runs."""
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def best(self) -> EpochRow | None:
        """The recorded epoch with the highest mean AUC, or None if the run recorded nothing.

        The BEST rather than the last, because that is the epoch whose weights the checkpoint holds —
        train.py keeps the best state, so a comparison against the final epoch would report a number no
        saved member ever had.
        """
        # An unscored run (--fold -1) records NaN, and NaN loses every comparison - an unguarded
        # max() would return whichever row happened to come first and call it the best.
        rows = [row for row in self.rows() if not math.isnan(row["auc"])]
        return max(rows, key=lambda row: row["auc"]) if rows else None
