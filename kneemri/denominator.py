"""What a check MEASURED, and the refusal owed when that is nothing.

Every check here ends the same way: a count of the things it looked at, and an exit code. The two are
independent, and that is the defect — a check whose denominator collapses to zero prints the sentence a
passing check prints ("0 arms checked, 0 refused") and exits 0, so a broken tool is indistinguishable
from a clean tree by exactly the signal a caller reads.

It has happened four times in a week, in three different tools, from three unrelated causes:
`arm_check` twice (once from WSL's bash, once from a PowerShell PATH with no coreutils — every arm
timed out and printed no invocation), `kit_check` for 53 shipped-package commits (a pathspec anchored
one level down matched no file, so `git log` named no commit and every stale kit read as current), and
the encoder drift gate (a compiled checkpoint whose every key carries `_orig_mod.` matched no stock
tensor and reported a drift of 0.000). None of the three was a wrong answer about the thing being
checked; all three were an answer about nothing, wearing a pass.

So the count is not a log line, it is the check's own precondition: a denominator of zero is refused
HERE rather than reported by each caller, because the three that had the guard wrote it three ways and
the four that did not were the ones nobody had watched fail yet.
"""

import logging
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Denominator:
    """How many things a check looked at, and what it would have been looking at."""

    measured: int
    what: str
    """Completes the sentence "REFUSED: <what>", so it names the empty set and its likeliest cause —
    the reader of that line is holding a tool that has just told them everything is fine."""

    def refuse_empty(self) -> None:
        """Exit non-zero when nothing was measured, naming what was missing rather than what passed."""
        if self.measured:
            return
        logger.error("REFUSED: %s. A check that measured nothing has not passed.", self.what)
        sys.exit(1)
