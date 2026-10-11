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
from kneemri.kaggle.script_seat import Anchor

logger = logging.getLogger(__name__)

BUDGET_SOURCE = "TIME_BUDGET = 8.0 * 3600"
"""The line his file spells the ceiling on, twice, verbatim."""

BUDGET_SITES = 2
"""How many times it must appear. Fewer means his source moved and the patch is priced on a guess."""

SUBMISSION = "submission.csv"

PROGRAM = "all_public.py"

FUTURE = "from __future__ import annotations"
"""The one statement Python allows only at the top of a file — and his file holds three of them."""

WEIGHTS_SOURCE = "if isinstance(man.get('members'), list) and man['members']:"
"""His `find_weights` test for whether a `manifest.json` it walked into is HIS weights package."""

WEIGHTS_PATCH = (
    "if isinstance(man.get('members'), list) and man['members'] "
    "and all('pixel_group' in member for member in man['members']):"
)
"""The same test, restricted to a manifest whose members carry the key his loader then indexes.

WHY THIS IS A FIX AND NOT A CHANGE. `find_weights` walks `/kaggle/input` and accepts the FIRST
`manifest.json` holding a non-empty `members` list. On HIS kernel that is his fold package. On OURS
`dimiturnt/kneemri-members` sorts first, so his stage one loaded 126 of OUR members and died three
lines later on `m['pixel_group']` — a key our manifests have never had. He then wrote a fallback table
and his stages two to four ran on top of it, which is a 0.939 program missing its largest stage and
saying so nowhere a score can see. The predicate is his own contract made explicit: he indexes
`pixel_group` on every member, so a package without it was never a candidate. It restores exactly what
his search finds when our dataset is not mounted beside his."""

WEIGHTS_SITES = 1
"""How many times that test is spelled. His five other `os.walk` prunes are by filename and unaffected."""

BASE_SOURCE = "def find_weights(name='manifest.json'):\n    import json\n    base = Path('/kaggle/input')"
"""Where his WEIGHTS search starts, anchored to the function it starts in.

The assignment alone appears six times in his file - his competition data, his cache and his model
mounts all begin the same walk. Rewriting all six would point every one of them at a weights package.
The anchor is the function header, which appears once."""

HIS_WEIGHTS = "/kaggle/input/datasets/mattiaangeli/knee-mri-fold-weights"
"""His own 24-member fold package - the one the board scored 0.939 on, named rather than searched for.

WHY NAMING IT IS THE FIX AND THE PREDICATE WAS NOT ENOUGH. Version 3 patched his search to skip a
manifest his loader cannot index, which removed OUR package from the candidates. It then took
`pilkwang/rsna-knee-weights` - 20 members, a manifest his loader CAN index, and not his. His program ran
whole, printed twenty healthy `submission.csv = weighted rank mean of N member(s)` lines and exited 0 on
somebody else's members. Sixteen sources are mounted here and his search is a walk in mount order, so
every predicate is a guess about which package sorts first. The only statement that is actually true is
which package is HIS, and it is a constant, so it is written down. The walk stays as the fallback: if
his dataset ever moves, the search is what it was."""

BASE_PATCH = (
    "def find_weights(name='manifest.json'):\n"
    "    import json\n"
    f"    base = Path({HIS_WEIGHTS!r}) if Path({HIS_WEIGHTS!r}).is_dir() else Path('/kaggle/input')"
)
"""His package when it is mounted, and his own search when it is not."""

BASE_SITES = 1


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

        The patched text is `compile`d before it is written. A syntax hazard in a flattened notebook is
        otherwise discovered by a subprocess that exits 1 with its message on a stream nobody keeps —
        which is how version 1 spent a kernel run to learn about a duplicated `__future__` import.
        """
        text = self._replaced(self.source.read_text(encoding="utf-8"), WEIGHTS_SOURCE, WEIGHTS_PATCH)
        text = self._replaced(text, BASE_SOURCE, BASE_PATCH, BASE_SITES)
        budget = f"TIME_BUDGET = {self.seconds:.1f}"
        patched = self.flattened(self._replaced(text, BUDGET_SOURCE, budget, BUDGET_SITES))
        compile(patched, str(destination), "exec")
        destination.write_text(patched, encoding="utf-8")
        logger.info("his program patched to a %.2f h budget at %s", self.seconds / 3600.0, destination)
        return destination

    def _replaced(self, text: str, spelling: str, replacement: str, sites: int = WEIGHTS_SITES) -> str:
        """`text` with every `spelling` replaced, refusing unless it appears exactly `sites` times.

        The count is the whole point. His file is published, not pinned, and a patch that silently lands
        on none of its sites - or on one of two - is a run that costs a slot to discover, which is what
        the single-site `TIME_BUDGET` case would have been: his rad stage would have gone on aiming at a
        nine-hour ceiling that is not there.
        """
        return Anchor(spelling, replacement, sites).applied(text, str(self.source))

    def flattened(self, text: str) -> str:
        """His three concatenated notebook CELLS made into one legal script.

        `from __future__ import annotations` is legal only before any other statement, and his published
        file is three cells pasted end to end — the statement sits at lines 1, 336 and 2210. Each cell
        was legal where it was executed; run as ONE script by a subprocess it is a `SyntaxError` at 336,
        which is exactly how version 1 of the kernel died at 0 s with the patcher blameless. Every
        occurrence after the first is dropped. Nothing else about his source is touched, and the
        statement is a no-op on the Python this runs under, so the program is unchanged by the deletion.
        """
        kept = 0
        lines: list[str] = []
        for line in text.splitlines(keepends=True):
            if line.startswith(FUTURE):
                kept += 1
                if kept > 1:
                    continue
            lines.append(line)
        logger.info("dropped %d trailing %r lines from his cell boundaries", max(kept - 1, 0), FUTURE)
        return "".join(lines)

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
        flat = [target for target in TARGETS if frame[target].nunique() == 1]
        if len(flat) == len(TARGETS) and len(frame) > 1:
            message = f"his table is CONSTANT on all {len(TARGETS)} targets - this is his fallback, not his program"
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
