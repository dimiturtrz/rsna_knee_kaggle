"""The wall-clock a Kaggle batch session gives a run, and whether the run is on course to overspend it."""

import time

SECONDS_PER_HOUR = 3600

KERNEL_SECONDS = 9 * SECONDS_PER_HOUR
"""Kaggle's wall-clock ceiling for a batch GPU session, which a rerun is killed at without notice."""

DEADLINE_MARGIN = 0.85
"""Fraction of the ceiling a run aims to finish inside.

Not the ceiling itself: the projection is made from chunks already done, and what remains — the last
chunk, the rank-combine, writing the CSV — is not in it. Fifteen percent of nine hours is eighty
minutes of slack for a tail the measurement cannot see."""


class Budget:
    """The wall-clock a run has left, and whether it is on course to spend more than that.

    THIS IS WHAT THREE UNSCORED ENTRIES WERE MISSING. A member set is chosen when the entry is built,
    from a three-study visible run that finishes in a hundred seconds; the hidden test set is thousands
    of studies, and the same set can project past nine hours there. A run that hits the ceiling is
    killed with no submission at all — so twenty-five members that do not finish score nothing, while
    twenty that do score most of what twenty-five would have.

    The projection needs one number the ensemble cannot compute and one the decode cannot: how far
    through the studies this group is (the decode knows) and how many groups there are (the ensemble
    knows). So the two of them report progress here and this object owns the arithmetic.
    """

    def __init__(self, seconds: float = KERNEL_SECONDS, margin: float = DEADLINE_MARGIN) -> None:
        self.seconds = seconds * margin
        self.started = time.time()
        self.groups = 1
        self.index = 0
        self.fraction = 0.0

    def begin(self, groups: int) -> None:
        """How many decodes of the whole split this entry will make, which is its unit of total work."""
        self.groups = max(groups, 1)

    def group(self, index: int) -> None:
        """Entering the index-th read group; everything before it counts as finished work."""
        self.index = index
        self.fraction = 0.0

    def chunk(self, done: int, total: int) -> None:
        """A chunk of the current group has landed: `done` of `total` for this decode."""
        self.fraction = done / max(total, 1)

    def projected(self) -> float:
        """Seconds the whole run is on course to take, or infinity before anything has finished.

        Every group is counted as equal work, which is what makes this an estimate rather than a
        measurement: a group of one member costs less than a group of twenty. It errs toward
        over-projecting on the cheap groups, and over-projecting is the safe direction — a member
        dropped that need not have been costs a little accuracy, and a ceiling hit costs the entry.
        """
        elapsed = time.time() - self.started
        share = (self.index + self.fraction) / self.groups
        return elapsed / share if share > 0 else float("inf")

    def measured(self) -> bool:
        """Whether any work has finished yet — before that there is nothing to project from."""
        return self.index + self.fraction > 0

    def overrunning(self) -> bool:
        """Whether the run is on course to be killed before it writes anything."""
        return self.measured() and self.projected() > self.seconds
