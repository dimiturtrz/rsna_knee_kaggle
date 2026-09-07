"""Mattia's published all-public program, run VERBATIM as a second producer beside our own roster.

WHY VERBATIM AND NOT LIFTED. `foreignseat_v1` lifted 45% of ONE of his three stages and read -0.013.
The program is three stages that each rewrite the same file: a 24-member rank mean over hashed DINO
folds, a `_micro` fold blend at alpha 0.45, and a RadImageNet dual-head pass that `os.replace`s the
result. Lifting a slice of the middle stage reproduces none of it, and the board has already priced
that mistake. What is board-verified is the WHOLE program at 0.939 (kernel 133147282, 2026-09-06), so
the whole program is what runs — as a subprocess, from his own source file, with one constant changed.

THE ONE CONSTANT. `TIME_BUDGET` appears twice in his file (the folds program and the rad program) and
is what makes him surrender members gracefully instead of being killed at Kaggle's nine-hour ceiling.
He wrote 8 h for a session he owns alone; here he shares it with our six, so it is rewritten to what
is actually left when our side has finished. Both occurrences are replaced and the count is asserted —
a single-site patch would leave the rad stage aiming at a ceiling that is not there.

ORDER, AND WHY OURS RUNS FIRST. His stages 2 and 3 read and rewrite `/kaggle/working/submission.csv`
by absolute path, so he cannot be confined to a scratch directory the way `RaptorEntry` is. Ours
therefore runs first and its table is moved aside; he then owns that path outright, and the blend is
written back over it at the end. It also means his budget is MEASURED rather than guessed: our runtime
on the hidden split has never been observed (the visible log is three studies), so a budget split
decided in advance would be a number nobody has.

WHAT THIS CANNOT MEASURE. Nothing. His rad stage has no holdout path, so there is no offline oracle
for this entry at all and none is pretended: the pre-registration is made from the board record, and
the only assertions available are that the program ran to completion, banked what it said it banked
and left a table with no nulls. `run_check` reads those from the log.
"""

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.findings import STUDY, TARGETS

logger = logging.getLogger(__name__)

BUDGET_SOURCE = "TIME_BUDGET = 8.0 * 3600"
"""The line his file spells the ceiling on, twice, verbatim."""

BUDGET_SITES = 2
"""How many times it must appear. Fewer means his source moved and the patch is priced on a guess."""

SUBMISSION = "submission.csv"

PROGRAM = "all_public.py"


@dataclass(frozen=True)
class BendEntry:
    """One run of his program: where his source is, where it writes, and what it may spend."""

    source: Path
    working: Path
    seconds: float

    def prepared(self, destination: Path) -> Path:
        """His source with `TIME_BUDGET` rewritten to our remaining seconds, written beside the run.

        A copy rather than an edit in place: the mounted file is read-only on Kaggle, and a patched
        copy is the artefact a later reader can diff against the published original to see that one
        constant is the whole difference.
        """
        text = self.source.read_text(encoding="utf-8")
        found = text.count(BUDGET_SOURCE)
        if found != BUDGET_SITES:
            message = f"{self.source} spells {BUDGET_SOURCE!r} {found} times, not {BUDGET_SITES}"
            raise ValueError(message)
        patched = text.replace(BUDGET_SOURCE, f"TIME_BUDGET = {self.seconds:.1f}")
        destination.write_text(patched, encoding="utf-8")
        logger.info("his program patched to a %.2f h budget at %s", self.seconds / 3600.0, destination)
        return destination

    def read(self) -> pd.DataFrame:
        """The table he left, refused rather than blended when it is not one.

        The refusals are the three failures his own log would still call a success: no file at all, a
        study missing from the join, and a null — any of which would make the blend silently wrong on
        the rows it touched rather than loudly absent.
        """
        written = self.working / SUBMISSION
        if not written.exists():
            message = f"his program wrote no {SUBMISSION}"
            raise FileNotFoundError(message)
        frame = pd.read_csv(written)
        missing = [column for column in (STUDY, *TARGETS) if column not in frame.columns]
        if missing:
            message = f"his table is missing {missing}"
            raise ValueError(message)
        nulls = int(frame[list(TARGETS)].isna().sum().sum())
        if nulls:
            message = f"his table holds {nulls} nulls over {len(frame)} studies"
            raise ValueError(message)
        logger.info("his table reads %d studies", len(frame))
        return frame

    def run(self) -> pd.DataFrame:  # devtools-ignore: test-mirror
        """Patch, spawn, wait, read. Unmirrored: it spawns a 3,105-line program over the Kaggle mount.

        The child inherits this process's environment and runs with `working` as its directory, which
        is what his relative `submission.csv` writes and his absolute `/kaggle/working` paths agree
        with on Kaggle. Its output is not captured — a nine-hour child whose log arrives only at the
        end is a child nobody can watch — so his log lines land in the kernel log beside ours, which
        is where `run_check` looks for them.
        """
        program = self.prepared(self.working / PROGRAM)
        started = time.time()
        done = subprocess.run([sys.executable, str(program)], cwd=str(self.working), check=False)  # noqa: S603
        logger.info("his program exited %d after %.0fs", done.returncode, time.time() - started)
        if done.returncode != 0:
            message = f"his program exited {done.returncode}"
            raise RuntimeError(message)
        return self.read()
