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

    ONE PROGRESS MODEL: chunks of studies done, out of chunks the split holds. It used to be two — read
    groups, reported by the ensemble, times chunks inside the current group, reported by the decode —
    because the traversal was group-outer and a group WAS a fraction of the run. Chunk-outer advances
    every group together over one pass, so "group k of three" no longer names any part of the work, and
    the party that knows how far along the run is is the only one that ever knew how long the corpus is:
    the decode. That also drops the estimate this class used to carry — every group counted as equal
    work whatever its member count — because there is nothing left to weight.
    """

    def __init__(self, seconds: float = KERNEL_SECONDS, margin: float = DEADLINE_MARGIN) -> None:
        self.seconds = seconds * margin
        self.started = time.time()
        self.fraction = 0.0

    def chunk(self, done: int, total: int) -> None:
        """`done` of the split's `total` chunks have been decoded AND forwarded."""
        self.fraction = done / max(total, 1)

    def projected(self) -> float:
        """Seconds the whole run is on course to take, or infinity before anything has finished."""
        elapsed = time.time() - self.started
        return elapsed / self.fraction if self.fraction > 0 else float("inf")

    def measured(self) -> bool:
        """Whether any work has finished yet — before that there is nothing to project from."""
        return self.fraction > 0

    def overrunning(self) -> bool:
        """Whether the run is on course to be killed before it writes anything."""
        return self.measured() and self.projected() > self.seconds

    def report(self) -> str:
        """The projection as a chunk log reads it, or nothing at all while there is nothing to project.

        Formatted HERE because the numbers are here: a caller that renders them has to know that an
        unmeasured budget projects infinity and must print no line rather than `infh of 7.6h`, which is
        knowledge about this object's arithmetic living in a decode loop.
        """
        if not self.measured():
            return ""
        return f"projecting {self.projected() / SECONDS_PER_HOUR:.1f}h of {self.seconds / SECONDS_PER_HOUR:.1f}h, "
