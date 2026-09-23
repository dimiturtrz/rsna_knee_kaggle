"""A run's record, READ: what it was, what each epoch scored, and what each step did.

The read side of `RunMetrics`, split out so every analysis and every check depends on a reader rather
than on the thing that writes the file. It changes nothing a reader sees: files written before any of
the newer keys existed still read, because every key added since the first is `NotRequired`.
"""

import json
from pathlib import Path
from statistics import median
from typing import NotRequired, TypedDict, cast

from kneemri.runs.artefact import Artefact

METRICS_SUFFIX = ".metrics"
"""What a run's score record is called, between the run's name and the record's own extension."""

METRICS_RECORD = Artefact.METRICS.value
"""The record's full trailing name, as it appears on disk."""

STALL_FACTOR = 4.0
"""How many times the median an epoch must cost before it is a stall rather than a slow epoch.

Derived from both sides of the gap it has to separate. The three stalls on disk are 161x, 116x and 69x
their run's median; the widest NORMAL spread in the same files is 404-530 s (1.3x) and 224-257 s (1.15x),
and a cold first epoch is the worst case above that. 4x sits an order of magnitude below the smallest
real stall and three times above the largest honest epoch, so neither side can reach it.
"""
MINIMUM_FOR_A_MEDIAN = 3
"""Rows below which a median cannot separate a stall from the run's own spread — see `stalls`."""

# An AUC is a probability, so anything outside this is a recorder defect rather than a bad score.
AUC_FLOOR = 0.0
AUC_CEILING = 1.0

RECIPE = "recipe"
"""The key a recipe line carries. An epoch row has no such key and a recipe line has no epoch, so the
two are told apart by what they hold rather than by where they sit in the file."""

RUN = "run"
"""The key the run's IDENTITY line carries — id, commit, parents — kept apart from the recipe line so
two runs' recipes still compare equal when only their identities differ."""


class EpochRow(TypedDict):
    """One epoch as it is recorded. A TypedDict rather than `Any` because this IS the schema.

    `vram_peak_gb` and `vram_alloc_gb` are filled in by the recorder rather than the caller — the peak
    is only readable from the allocator, and only before it is reset.

    RESERVED IS THE ONE THAT ANSWERS "DOES IT FIT". `--vram-gb` sets
    `set_per_process_memory_fraction`, which bounds what the allocator RESERVES from the device, and
    Windows answers an over-commit by paging over PCIe rather than raising — so the number that decides
    whether an arm crawls has to be the one the cap is compared against. This recorded
    `max_memory_allocated` instead, which is the model's own demand and excludes every byte the
    allocator holds but has not handed out. The 448 px OOM was written by exactly that gap: 1.17 GB sat
    reserved-but-unallocated inside the cap, invisible to the recorded peak. Rows written before this
    carry allocated under the `vram_peak_gb` name and therefore UNDERSTATE what their run held; the 420
    px probe measured 23.33 GB allocated against a 26 GB cap, and how much of the remaining 2.67 was
    real headroom was unanswerable from the ledger.

    BOTH, because their DIFFERENCE is fragmentation and that is a separate diagnosis from capacity: a
    run whose reserved sits far above its allocated is not short of memory, it is short of contiguous
    memory, and the two have different fixes.

    `started` is UNIX seconds at the top of the epoch, and it is what turns a stall from something
    inferred into something identifiable. `seconds` alone says an epoch took eight hours; it cannot say
    WHEN, so it cannot be lined up against the suspend/resume the stall is suspected to be. Both fields
    are `NotRequired` because files written before either existed are still read by everything here.

    `ece` is the epoch's expected calibration error: fifteen equal-width bins over [0, 1] of the mean
    predicted probability against the truth BINARISED AT 0.5 — the cut `LabelScores.per_label` scores
    the twelve AUCs against — pooled over every (study, label) cell of the holdout with NaN cells
    excluded. `loss_q25`/`loss_q50`/`loss_q75`/`loss_q90`/`loss_q95` are the epoch's per-study loss
    quantiles — the distribution the scalar `loss` averages away. All six are `NotRequired` under the
    same discipline: files written before them must still read.

    `holdout_loss` is the mean binary cross-entropy over every scored holdout cell, the loss-space twin of
    `auc`. `fit_auc` is the mean AUC of the TRAINING predictions as they were made — under augmentation, on
    weights still moving — so it is a fit-side trend to read against `auc`, never a number to compare with
    it. `grad_norm_*` are the pre-clip gradient norms over the epoch's optimiser steps, `nonfinite_steps`
    how many were not finite, `lr` the rate per parameter group at the epoch's last step, and `drawn_*`
    what the sampler drew: rows, distinct studies, and positives per finding. `fit_seconds` is the part of
    `seconds` spent fitting rather than scoring the holdout. All `NotRequired` for the
    same reason as the rest.
    """

    epoch: int
    loss: float
    auc: float
    seconds: int
    per_label: dict[str, float]
    vram_peak_gb: NotRequired[float]
    vram_alloc_gb: NotRequired[float]
    started: NotRequired[int]
    fit_seconds: NotRequired[int]
    ece: NotRequired[float]
    loss_q25: NotRequired[float]
    loss_q50: NotRequired[float]
    loss_q75: NotRequired[float]
    loss_q90: NotRequired[float]
    loss_q95: NotRequired[float]
    holdout_loss: NotRequired[float]
    fit_auc: NotRequired[float]
    grad_norm_mean: NotRequired[float]
    grad_norm_max: NotRequired[float]
    nonfinite_steps: NotRequired[int]
    lr: NotRequired[list[float]]
    drawn_rows: NotRequired[int]
    drawn_studies: NotRequired[int]
    drawn_positives: NotRequired[dict[str, int]]


class StepRow(TypedDict):
    """One optimiser step: the group's mean loss, the gradient norm BEFORE clipping, and each rate."""

    epoch: int
    step: int
    loss: float
    grad_norm: float
    lr: list[float]


class RunRecord:
    """One run's record, read from the JSONL beside its checkpoint."""

    def __init__(self, destination: Path) -> None:
        # ANY SIBLING NAMES THE RUN: the trainer holds the checkpoint, an analysis holds the metrics file
        # it can see on disk, and both must land on the same record rather than `.metrics.metrics.jsonl`.
        self.path = Artefact.METRICS.beside(destination)

    def exists(self) -> bool:
        """Whether anything has been recorded here yet.

        Asked of the recorder rather than of its path, so a caller distinguishing "wrong file" from
        "run died before its first scored epoch" does not have to know where the file lives.
        """
        return self.path.exists()

    def checkpoint(self) -> Path:
        """The member these scores describe, beside them under the run's own name.

        Here rather than at the caller because `.metrics.jsonl` is TWO suffixes: the obvious
        `with_suffix` yields `run.metrics.pt`, a path nothing is written to and everything reads as a
        run with no stamp. This class already resolves that spelling in the other direction, so it is
        the one place that cannot get the pair wrong.
        """
        return Artefact.CHECKPOINT.beside(self.path)

    def name(self) -> str:
        """The run's own name, with none of the trailing spellings its files carry."""
        return Artefact.stem(self.path)

    def _entries(self) -> list[dict[str, object]]:
        """Every recorded line, epochs, recipe and identity alike."""
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def recipe(self) -> dict[str, str] | None:
        """The recorded recipe, or None for a file written before runs recorded one."""
        described = [cast("dict[str, str]", entry[RECIPE]) for entry in self._entries() if RECIPE in entry]
        return described[-1] if described else None

    def run(self) -> dict[str, object] | None:
        """The LAST run's identity line, or None for a file written before runs had an identity."""
        named = [cast("dict[str, object]", entry[RUN]) for entry in self._entries() if RUN in entry]
        return named[-1] if named else None

    def rows(self) -> list[EpochRow]:
        """The LAST run's epoch rows — the run this file's own `recipe()` describes.

        Filtered by the presence of an epoch rather than by position: a recipe line is not an epoch,
        and a reader that assumed the first line was one would rank the recipe as a score.

        THE LAST RUN, because a re-run to the same `--out` appends and the checkpoint beside it holds
        only the newest one. `pair_memory_s2026` is that file: twelve epochs peaking at 0.8353, then a
        second run of four peaking at 0.8155, and reading both at once quotes 0.8353 for a checkpoint
        that no longer exists. Its kept predictions — `trained`, so exact by construction — score
        0.8155, which is the oracle: a trained record and its own recorded best cannot disagree.

        THE LAST NON-EMPTY GROUP rather than whatever follows the last recipe, because a recipe does
        not always precede the epochs it describes. Six files on disk had theirs BACK-FILLED, appended
        after a run that had already finished, and a rule reading "the rows after the last recipe"
        reports every one of them as having recorded nothing at all.

        `recipe()` has always answered for the last run alone. Rows answering for all of them meant one
        file described one run and scored another, which is the same failure as an unmatched control
        wearing a matched one's name.

        AND A RESTARTING EPOCH IS A BOUNDARY TOO, because the recipe line is not always where the
        boundary is. A back-filled recipe sits at the END of the file, so splitting on recipes alone
        puts every run before it into one group: `lora_r8_v3small_224_s2026` is four epochs of an
        aborted run (peaking 0.6867) followed by the real twelve (peaking 0.8217) and then its recipe,
        which reads as one sixteen-epoch trajectory climbing +0.1350 — two and a half times the largest
        real gain in the corpus, and `trajectory_check` decides on a maximum. The boundary was in the
        data the whole time: a run's `epoch` counts up, so a value that does not increase is a new run.
        Three of 142 files on disk need it, and no SHIPPED number moves — the roster quotes
        `lora_r8_v3small_224` at 0.8215 and its clean last-run peak is 0.8217, the same epoch either
        way. What moves is every TRAJECTORY read across such a file. See knee-v5lb.

        An identity line is NOT a boundary: it is written beside the recipe it belongs to, and splitting
        on both would hand the recipe's group an empty sibling.
        """
        groups: list[list[EpochRow]] = [[]]
        for entry in self._entries():
            if RECIPE in entry:
                groups.append([])
            elif "epoch" in entry:
                row = cast("EpochRow", entry)
                if groups[-1] and row["epoch"] <= groups[-1][-1]["epoch"]:
                    groups.append([])
                groups[-1].append(row)
        return next((group for group in reversed(groups) if group), [])

    def steps(self) -> list[StepRow]:
        """Every optimiser step this run recorded, or none for a run older than the step stream."""
        stream = Artefact.STEPS.beside(self.path)
        if not stream.exists():
            return []
        return [cast("StepRow", json.loads(line)) for line in stream.read_text(encoding="utf-8").splitlines() if line]

    def epoch_seconds(self) -> float:
        """The MEDIAN epoch's wall-clock, or 0.0 for a run that recorded none.

        The median rather than the mean, because `seconds` is wall-clock and wall-clock counts hours the
        card spent doing nothing. Two runs on disk each carry ONE ~8-hour epoch — `slices16_ws_s2026`
        30,684 s among epochs of 404-530, `pair_memory_s2026` 27,266 s among epochs of 224-257 — and a
        mean over those reports 2970 s and 2270 s. Read that way, `slices16_ws` looks 8.8x its control's
        cost and `pair_memory` 5.7x, and BOTH readings are false: their real per-epoch costs are ~1.3x
        and ~0.7x. The ledger's "33% slower" for `slices16_ws` was right and the arithmetic that
        contradicted it was the broken instrument. One stalled epoch is enough to invert a throughput
        verdict, so the robust statistic is the one this exposes.
        """
        recorded = sorted(row["seconds"] for row in self.rows())
        return median(recorded) if recorded else 0.0

    def stalls(self) -> list[EpochRow]:
        """The epochs that cost `STALL_FACTOR`x the median one — the run's lost hours, named.

        `epoch_seconds` made a stall HARMLESS to a throughput comparison by reporting the median. That
        left the stall itself invisible: three runs on disk carry one multi-hour epoch each, 20.8 hours
        between them, and the only way to see them was to read all 24 metrics files by hand. This is
        that reading, as an answer the file gives about itself.

        A FACTOR over the median, not a fixed ceiling, because a normal epoch costs 104 s in one recipe
        and 443 s in another and no single wall-clock separates them. Nothing is reported for a run of
        fewer than three epochs: with two rows the median sits between them, so the slower one is always
        above it, and a first epoch that pays a warm-up would report itself as a stall on every run.
        """
        rows = self.rows()
        if len(rows) < MINIMUM_FOR_A_MEDIAN:
            return []
        ceiling = self.epoch_seconds() * STALL_FACTOR
        return [row for row in rows if row["seconds"] > ceiling]

    def best(self) -> EpochRow | None:
        """The recorded epoch with the highest mean AUC, or None if the run recorded nothing.

        The BEST rather than the last, because that is the epoch whose weights the checkpoint holds —
        train.py keeps the best state, so a comparison against the final epoch would report a number no
        saved member ever had.
        """
        # NOTHING held back means nothing measured, whatever the rows say. Until 2026-08-16 the epoch
        # INDEX was written into this column for an unscored run, so `final_ws_e12` and `final_fit_e12`
        # sit on disk with auc = 0.0, 1.0, 2.0 ... 11.0 and `max()` quotes a twelve-epoch final fit as
        # scoring 11.0 — beating every real member in the ledger. The recorder is fixed, but the files
        # it already wrote are not, and asking the run's OWN recipe costs one lookup and is true for
        # both: a run that held nothing back cannot have a best epoch, so it has no score to report.
        if (recipe := self.recipe()) is not None and int(recipe.get("fold", 0)) < 0:
            return None
        # AND THEN BY VALUE, because the guard above is a PROVENANCE lookup: it asks the recipe what the
        # run was for, so a file corrupted the same way whose recipe does not happen to record fold < 0
        # walks straight past it and `max()` quotes an epoch index as a score again. An AUC is a
        # probability; 11.0 is not one, and refusing it costs one comparison and needs nothing from the
        # recipe. NaN, which an unscored run records and which loses every comparison, fails this test
        # too - an unguarded max() would otherwise return whichever row came first and call it the best.
        rows = [row for row in self.rows() if AUC_FLOOR <= row["auc"] <= AUC_CEILING]
        return max(rows, key=lambda row: row["auc"]) if rows else None
