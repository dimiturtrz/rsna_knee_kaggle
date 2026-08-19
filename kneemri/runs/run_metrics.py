"""One machine-readable line per epoch, so runs can be compared without grepping prose."""

import json
import logging
import math
from pathlib import Path
from statistics import median
from typing import NotRequired, TypedDict, cast

import torch

logger = logging.getLogger(__name__)

BYTES_PER_GB = 1024**3
STALL_FACTOR = 4.0
"""How many times the median an epoch must cost before it is a stall rather than a slow epoch.

Derived from both sides of the gap it has to separate. The three stalls on disk are 161x, 116x and 69x
their run's median; the widest NORMAL spread in the same files is 404-530 s (1.3x) and 224-257 s (1.15x),
and a cold first epoch is the worst case above that. 4x sits an order of magnitude below the smallest
real stall and three times above the largest honest epoch, so neither side can reach it.
"""
MINIMUM_FOR_A_MEDIAN = 3
"""Rows below which a median cannot separate a stall from the run's own spread — see `stalls`."""
RECIPE = "recipe"
"""The key a recipe line carries. An epoch row has no such key and a recipe line has no epoch, so the
two are told apart by what they hold rather than by where they sit in the file."""


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
    """

    epoch: int
    loss: float
    auc: float
    seconds: int
    per_label: dict[str, float]
    vram_peak_gb: NotRequired[float]
    vram_alloc_gb: NotRequired[float]
    started: NotRequired[int]


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
        name = self.path.name
        return self.path.with_name(name.removesuffix(".metrics.jsonl") + ".pt")

    def describe(self, recipe: dict[str, str]) -> None:
        """Record WHICH RUN these epochs belong to, once, before the first of them.

        A metrics file used to hold numbers and nothing else, so the only statement of the recipe that
        produced them was the arm's shell script and the file's own name. `RunComparison` then subtracts
        one run's per-label scores from another's with nothing able to say the two differ only in the
        axis under test — and an unmatched control is not a control. This is the same failure this
        project keeps paying for, in the analysis layer: the knowledge existed, and it did not reach
        the place that needed it.
        """
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({RECIPE: recipe}) + "\n")

    def begin(self, recipe: dict[str, str]) -> None:
        """Start a run's record, moving any PREVIOUS run's epochs aside first.

        A re-run to the same `--out` used to append, which left one file describing two runs and one
        checkpoint beside it, and no way to say which run wrote the checkpoint. `rows()` guesses THE
        LAST group, and that guess is wrong exactly when the re-run was KILLED: `slices16_ws_s2026`
        holds a finished run peaking at 0.8452 and a killed one peaking at 0.8411, its checkpoint is
        the finished run's, and every reader quoted 0.8411 for a member that scores 0.8452 — a gap big
        enough to reorder the ledger and small enough to read as a normal number.

        Rotating rather than truncating, because a killed run's epochs are still the only account of
        what it did; and here rather than in `describe`, which `metrics_recipe` uses to BACK-FILL a
        recipe onto a finished run's file and must go on appending.
        """
        if any("epoch" in entry for entry in self._entries()):
            self.path.replace(self._rotation())
        self.describe(recipe)

    def _rotation(self) -> Path:
        """The next free `<run>.metrics.<n>.jsonl`, so no earlier run's epochs are ever overwritten."""
        name = self.path.name
        stem = name.removesuffix(".metrics.jsonl")
        run = 1
        while (aside := self.path.with_name(f"{stem}.metrics.{run}.jsonl")).exists():
            run += 1
        return aside

    def recipe(self) -> dict[str, str] | None:
        """The recorded recipe, or None for a file written before runs recorded one."""
        described = [cast(dict[str, str], entry[RECIPE]) for entry in self._entries() if RECIPE in entry]
        return described[-1] if described else None

    def record(self, row: EpochRow) -> None:
        """Append one epoch's row, attach the VRAM peak, and report the per-label scores.

        It logs as well as records so the caller has ONE statement per epoch rather than two that can
        drift apart — the log line and the stored row should never be able to disagree.
        """
        if torch.cuda.device_count():
            row["vram_peak_gb"] = round(torch.cuda.max_memory_reserved() / BYTES_PER_GB, 2)
            row["vram_alloc_gb"] = round(torch.cuda.max_memory_allocated() / BYTES_PER_GB, 2)
            torch.cuda.reset_peak_memory_stats()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
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

    def _entries(self) -> list[dict[str, object]]:
        """Every recorded line, epochs and recipe alike."""
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

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
        """
        groups: list[list[EpochRow]] = [[]]
        for entry in self._entries():
            if RECIPE in entry:
                groups.append([])
            elif "epoch" in entry:
                groups[-1].append(cast(EpochRow, entry))
        return next((group for group in reversed(groups) if group), [])

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
        # An unscored run (--fold -1) records NaN, and NaN loses every comparison - an unguarded
        # max() would return whichever row happened to come first and call it the best.
        rows = [row for row in self.rows() if not math.isnan(row["auc"])]
        return max(rows, key=lambda row: row["auc"]) if rows else None
