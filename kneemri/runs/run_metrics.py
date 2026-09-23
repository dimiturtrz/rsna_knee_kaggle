"""The run record's WRITER: one machine-readable line per epoch, so runs compare without grepping prose.

Everything that only reads a record goes through `RunRecord`; this adds the three things a TRAINING run
does to it — start it (rotating an earlier run aside), describe it, and append to it.
"""

import json
import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, Int
from numpy.typing import NDArray

from kneemri.runs.artefact import Artefact
from kneemri.runs.run_record import RECIPE, RUN, EpochRow, RunRecord, StepRow

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024**3

ROTATING = (Artefact.CHECKPOINT, Artefact.HOLDOUT, Artefact.STEPS, Artefact.DRAWS, Artefact.EPOCHS)
"""Every file one run writes under its name, so a re-run rotates the whole run aside, never part of it."""


@dataclass(frozen=True)
class HeldOut:
    """One epoch's holdout predictions, the graded truth they are read against, and whose studies they are."""

    predicted: Float[NDArray[np.float64], "h o"]
    truth: Float[NDArray[np.float32], "h o"]
    studies: list[str]


class RunMetrics(RunRecord):
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

    def describe(self, recipe: dict[str, str]) -> None:
        """Record WHICH RUN these epochs belong to, once, before the first of them.

        A metrics file used to hold numbers and nothing else, so the only statement of the recipe that
        produced them was the arm's shell script and the file's own name. `RunComparison` then subtracts
        one run's per-label scores from another's with nothing able to say the two differ only in the
        axis under test — and an unmatched control is not a control. This is the same failure this
        project keeps paying for, in the analysis layer: the knowledge existed, and it did not reach
        the place that needed it.
        """
        self._append({RECIPE: recipe})

    def begin(self, recipe: dict[str, str], *, run: dict[str, object] | None = None, resuming: bool = False) -> None:
        """Start a run's record, moving any PREVIOUS run's epochs AND ARTEFACTS aside first.

        A re-run to the same `--out` used to append, which left one file describing two runs and one
        checkpoint beside it, and no way to say which run wrote the checkpoint. `rows()` guesses THE
        LAST group, and that guess is wrong exactly when the re-run was KILLED: `slices16_ws_s2026`
        holds a finished run peaking at 0.8452 and a killed one peaking at 0.8411, its checkpoint is
        the finished run's, and every reader quoted 0.8411 for a member that scores 0.8452 — a gap big
        enough to reorder the ledger and small enough to read as a normal number.

        Rotating rather than truncating, because a killed run's epochs are still the only account of
        what it did; and here rather than in `describe`, which `metrics_recipe` uses to BACK-FILL a
        recipe onto a finished run's file and must go on appending.

        THE WEIGHTS AND THE PREDICTIONS ROTATE WITH THE EPOCHS, because rotating only the epochs is the
        asymmetry that cost `depth_regions_span` its run-1 predictions on 2026-08-19: the arm re-entered
        `run_train` mid-flight, trained a second time under its own name, and the ledger's row went on
        quoting a best whose `.holdout.npz` had been overwritten under it — a number with no file behind
        it, and nothing on disk saying so. Saving one run's epochs while dropping the weights they
        describe preserves the account and destroys the evidence.

        Under `_run<n>`, the spelling a human already reached for by hand that night, so the rotated
        artefacts read as an ordinary run to every tool: `HoldoutRecord` resolves the pair, the ledger
        check counts it in its denominator, and a comparison can pair it like any other member.

        A RESUMING run carries the rotated predictions back — see `_carry_back`. The run's IDENTITY, when
        given, is written as its own line beside the recipe (`RunRecord.run`).
        """
        if any("epoch" in entry for entry in self._entries()):
            rotation = self._rotation()
            for kind in ROTATING:
                if (live := kind.beside(self.path)).exists():
                    live.replace(aside := kind.aside(self.path, rotation))
                    logger.info("    earlier run under this name kept as %s", aside.name)
            if resuming:
                self._carry_back(rotation)
            self.path.replace(Artefact.METRICS.aside(self.path, rotation))
        self.describe(recipe)
        if run is not None:
            self._append({RUN: run})

    def _carry_back(self, run: int) -> None:
        """Put the rotated predictions back under the live name, for a run that is RESUMING.

        `--resume` restores the KEPT epoch, so the record just rotated aside is the reading OF THE
        CHECKPOINT THIS RUN STARTS FROM — and a predictions file is only rewritten when a new best
        fires. Resume onto a best no later epoch beats and the live name stays empty for a run that
        succeeded: `labelcut_v3_ema_s2026` finished e23 with its 0.9039 reading sitting under
        `_run1`, its declared `# reading:` path absent, and `reading_check` refusing the arm. Copied
        rather than moved, because the rotated run's own record must stay whole beside its epochs.
        """
        live, aside = Artefact.HOLDOUT.beside(self.path), Artefact.HOLDOUT.aside(self.path, run)
        if aside.exists():
            shutil.copyfile(aside, live)
            logger.info("    resuming: %s carried back as this run's reading until an epoch beats it", aside.name)

    def _rotation(self) -> int:
        """The next free rotation number, so no earlier run's record is ever overwritten.

        Decided by the EPOCHS alone: they are the one sibling every run writes, so a run killed before
        its first checkpoint still consumes a number and every file of one run keeps one index.
        """
        run = 1
        while Artefact.METRICS.aside(self.path, run).exists():
            run += 1
        return run

    def record(self, row: EpochRow) -> None:
        """Append one epoch's row, attach the VRAM peak, and report the per-label scores.

        It logs as well as records so the caller has ONE statement per epoch rather than two that can
        drift apart — the log line and the stored row should never be able to disagree.
        """
        if torch.cuda.device_count():
            row["vram_peak_gb"] = round(torch.cuda.max_memory_reserved() / BYTES_PER_GB, 2)
            row["vram_alloc_gb"] = round(torch.cuda.max_memory_allocated() / BYTES_PER_GB, 2)
            torch.cuda.reset_peak_memory_stats()
        self._append(row)
        logger.info("    %s", {name: round(value, 3) for name, value in row["per_label"].items()})
        if "vram_peak_gb" in row:
            # SAID WHILE THE RUN IS GOING, not only left in the JSONL. An arm that does not fit crawls
            # instead of failing, and the wording matters: `run_train` shows the terminal only lines
            # matching `epoch |best |rror|emory`, so "memory" is what carries this past that filter to
            # the person who can still kill the arm in its first minutes.
            logger.info(
                "    peak memory %.2f GB reserved, %.2f GB allocated",
                row["vram_peak_gb"],
                row["vram_alloc_gb"],
            )

    def closed(
        self, row: EpochRow, steps: list[StepRow], drawn: Int[NDArray[np.int64], " n"], held: HeldOut | None
    ) -> None:
        """Write one epoch: its steps, what it drew, what it predicted on the holdout, then its row.

        The ROW LAST, so a run stopped by patience has a complete record up to the epoch it stopped on,
        and every file a row points at already exists when the row lands.
        """
        self.steps_taken(steps)
        self.drawn(row["epoch"], drawn)
        if held is not None:
            self.predicted(row["epoch"], held)
        self.record(row)

    def steps_taken(self, rows: list[StepRow]) -> None:
        """Append one epoch's optimiser steps to the step stream beside the record."""
        with Artefact.STEPS.beside(self.path).open("a", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row) + "\n" for row in rows)

    def drawn(self, epoch: int, positions: Int[NDArray[np.int64], " n"]) -> None:
        """Keep the positions the sampler drew this epoch, under `e<epoch>` in the draws file."""
        self._stacked(Artefact.DRAWS, epoch, positions)

    def predicted(self, epoch: int, held: HeldOut) -> None:
        """Keep this epoch's holdout predictions beside the truth and study names they are read against.

        EVERY epoch, not only the kept one: the `.holdout.npz` holds the best epoch's predictions, so how
        a label's ranking moved from epoch to epoch — the curve behind every "peaked early" — had no file.
        """
        self._stacked(Artefact.EPOCHS, epoch, held.predicted, truth=held.truth, studies=np.asarray(held.studies))

    def _stacked(self, kind: Artefact, epoch: int, array: NDArray[np.generic], **constant: NDArray[np.generic]) -> None:
        """Add one epoch's array to a per-run npz, rewriting it whole through a partial file.

        Rewritten rather than appended because npz cannot append, and through `.part` so a run killed
        mid-write keeps every earlier epoch rather than a truncated archive.
        """
        path = kind.beside(self.path)
        held: dict[str, NDArray[np.generic]] = dict(np.load(path)) if path.exists() else {}
        partial = path.with_name(path.name + ".part")
        with partial.open("wb") as handle:
            np.savez(handle, allow_pickle=False, **(held | constant | {f"e{epoch}": array}))
        partial.replace(path)

    def _append(self, line: Mapping[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
