"""How many slots are left today and when they are due. `python -m kneemri.kaggle.slot_pace`.

THE COUNT MUST COME FROM KAGGLE, NOT FROM `comp/board.jsonl`. The record is written when a submission
SCORES, so a pending entry is in no row and a freshly-scored one is in no row yet. At 11:42Z on
2026-09-06 the record held one of the day's two submissions — 56052777 had been fired at 10:21Z and was
still `PENDING` — and the arithmetic done off the record read `0 spent, 5 left`, spacing the marks
2.36 h apart when the truth was 3.93 h. That does not merely mis-space the day: it authorises shipping
slots that do not exist, and the five that exist are destroyed at 00:00 UTC either way. Asking Kaggle
for the submission list spends NO slot, which is what makes the honest count free.

THE SAME ARITHMETIC, RETYPED, IS THE OTHER HALF OF THE DEFECT. The pacing rule lived in prose and was
re-derived by hand at every wake; the read above used `submitted` as the row's date key, which is not
one of `board.jsonl`'s keys at all (it is `when`), so the filter matched nothing and reported an empty
day with no error. A rule that must be retyped is a rule that gets retyped wrong — the same argument
`reader_end_chain.sh` makes, and `recal_dense` paid for.

THERE IS NO SKIP. The mark is always `(23:30Z - now) / slots_left`, recomputed at every check, so a
mark that passes unspent does not disappear — it COMPRESSES every later one. Deferring buys nothing
and shortens the runway, which is why this prints the compressed interval rather than a fixed
timetable.
"""

import datetime as dt
import logging
import sys
from dataclasses import dataclass

from core import cli
from kneemri.kaggle.board_check import LiveBoard

logger = logging.getLogger(__name__)

DAILY_SLOTS = 5
"""Kaggle's daily allowance for this competition. Unspent slots are destroyed at 00:00 UTC, never
banked, which is why the denominator is the day and not the week."""

LAST_MARK = dt.time(23, 30, tzinfo=dt.UTC)
"""The owner's directive (2026-08-31): the day's last submission goes at ~23:30 UTC, leaving half an
hour against a reset nobody can move."""


@dataclass(frozen=True)
class SlotPace:
    """Today's submissions against the clock, and the marks the ones left are due at."""

    submitted: tuple[dt.datetime, ...]
    now: dt.datetime

    def spent(self) -> int:
        """Submissions fired on the CURRENT UTC day, scored or not — the allowance counts firings."""
        return len([stamp for stamp in self.submitted if stamp.date() == self.now.date()])

    def left(self) -> int:
        """Slots still available today, never below zero even if the allowance was somehow exceeded."""
        return max(DAILY_SLOTS - self.spent(), 0)

    def interval(self) -> dt.timedelta:
        """Time between the remaining marks: what is left of the day divided by the slots left.

        Recomputed from NOW rather than from the day's start, which is what makes an unspent mark
        compress the rest instead of vanishing.
        """
        if self.left() == 0:
            return dt.timedelta(0)
        return max(self.deadline() - self.now, dt.timedelta(0)) / self.left()

    def deadline(self) -> dt.datetime:
        """The day's last mark, as an instant."""
        return dt.datetime.combine(self.now.date(), LAST_MARK)

    def marks(self) -> list[dt.datetime]:
        """Every remaining mark, earliest first, the last of them the deadline itself."""
        return [self.now + self.interval() * step for step in range(1, self.left() + 1)]

    def lines(self) -> list[str]:
        """What a wake needs to read: the count, the compressed interval, and the marks themselves."""
        if self.left() == 0:
            return [f"{self.spent()} of {DAILY_SLOTS} spent today - no slot left, the next reset is 00:00Z"]
        hours = self.interval().total_seconds() / 3600
        marks = ", ".join(mark.strftime("%H:%MZ") for mark in self.marks())
        return [
            f"{self.spent()} of {DAILY_SLOTS} spent today, {self.left()} left",
            f"interval {hours:.2f}h - marks {marks}",
        ]


def main() -> None:
    """Print today's slot count and the compressed marks; exit 1 when the day is spent."""
    command = cli.Command(__doc__, logger)
    arguments = command.parsed()
    del arguments

    pace = SlotPace(LiveBoard.taken().dates, dt.datetime.now(dt.UTC))
    command.report(pace.lines())
    if pace.left() == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
