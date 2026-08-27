"""Give a published foreign checkpoint a holdout record: `python -m kneemri.inference.foreign_holdout`.

WHY IT EXISTS. `seat_value` prices a candidate by the record it left beside its checkpoint, and it does
not care who trained the weights — only that the record joins. A foreign member has no training run
here, so nothing ever writes one, and the one member on an axis no seat has ever occupied is the one
member the seat ruler cannot see.

WHY IT BORROWS A MEMBER'S STUDY LIST RATHER THAN REBUILDING A SPLIT. `HoldoutReading.aligned` refuses a
pair scored by different rulers, and a foreign checkpoint has no recipe to rebuild a split from: no
folds, no split seed, no label table. Reconstructing one from flags would be guessing at the very thing
the refusal exists to catch. So `--like` names a member that already has a record and this command
adopts its studies AND its truth verbatim. The join is then true by construction, and a foreign record
can never be paired against a member it does not share a ruler with, because it has no other ruler.

WHAT THE RECORD IS NOT. `source` is `DERIVED`: these weights never saw this holdout, so the array is not
a training-loop artefact, and every downstream interval built on it is indicative. It is also a reading
through OUR feed rather than theirs — `foreign_gate` is what prices that shortfall, and this command
should not be run on a key that gate has not been pointed at first.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from core import cli
from kneemri.inference.foreign_member import ForeignMember, ForeignReadings, WindowFeed
from kneemri.runs.holdout_record import DERIVED, UNKNOWN_EPOCH, HoldoutReading, HoldoutRecord

logger = logging.getLogger(__name__)


class ForeignHoldout:
    """One published member, read over the studies a member of ours already holds a record for."""

    def __init__(self, member: ForeignMember, feed: WindowFeed, device: torch.device) -> None:
        self.member = member
        self.feed = feed
        self.device = device

    @staticmethod
    def rows(order: list[str], studies: list[str]) -> NDArray[np.int64]:
        """Where each held-out study sits in the cache, refusing rather than silently reading fewer.

        A missing study would otherwise shrink the array and the record would still load — which is the
        shape of every mis-join this project has already paid for.
        """
        positions = {name: index for index, name in enumerate(order)}
        missing = [name for name in studies if name not in positions]
        if missing:
            message = f"{len(missing)} of {len(studies)} held-out studies are absent from this cache: {missing[:3]}"
            raise ValueError(message)
        return np.array([positions[name] for name in studies], dtype=np.int64)

    @staticmethod
    def borrowed(processed: Path, like: str) -> HoldoutReading:
        """The record this foreign reading will adopt, or a refusal naming the member that lacks one."""
        reading = HoldoutRecord(processed / f"{like}.pt").read()
        if reading is None:
            message = f"{like} has no holdout record, so there is no study list or ruler to borrow"
            raise ValueError(message)
        return reading

    def reading(self, borrowed: HoldoutReading, order: list[str]) -> HoldoutReading:
        """The foreign member's probabilities over the borrowed studies, under the borrowed truth."""
        rows = self.rows(order, borrowed.studies)
        values = ForeignReadings(self.device).predict(self.member.to(self.device), self.feed, rows.tolist())
        return HoldoutReading(
            studies=borrowed.studies,
            predictions=np.asarray(values, dtype=np.float64),
            truth=borrowed.truth,
            epoch=UNKNOWN_EPOCH,
            source=DERIVED,
        )


def main() -> None:
    """Write a foreign checkpoint's holdout record so `seat_value` can price the seat it would take."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--checkpoint", type=Path, required=True, help="a published foreign .pt")
    command.parser.add_argument("--key", required=True, help="the cache key foreign_gate scored best")
    command.parser.add_argument("--like", required=True, help="a member whose holdout studies and ruler to adopt")
    command.parser.add_argument("--name", required=True, help="the stem the record is written under")
    command.parser.add_argument("--device", default="cuda")
    arguments = command.parsed()

    processed = command.rooted(arguments).processed()
    adopted = ForeignHoldout.borrowed(processed, arguments.like)
    member = ForeignMember.rebuild(arguments.checkpoint)
    feed = WindowFeed.of(processed, arguments.key, member.spec.res)
    holdout = ForeignHoldout(member, feed, torch.device(arguments.device))

    reading = holdout.reading(adopted, WindowFeed.order(processed, arguments.key))
    HoldoutRecord(processed / f"{arguments.name}.pt").write(
        reading.studies, reading.predictions, reading.truth, reading.epoch, source=reading.source
    )
    command.report(
        [
            f"{arguments.name}: {len(reading.studies)} studies borrowed from {arguments.like}",
            f"read through {arguments.key} and written as {DERIVED}",
            "score it with `python -m kneemri.kaggle.seat_value --candidates "
            f"{arguments.name}.pt` -- the ruler lives there, not here",
        ]
    )


if __name__ == "__main__":
    main()
