"""What the ruler's uncertain-mention convention is worth, measured on the members we already have.

    python -m kneemri.evaluation.unk_surface
    python -m kneemri.evaluation.unk_surface --holdouts D:/models/rsna_knee_abnormality --top 15

THE CONVENTION IS A CONSTANT NOBODY ARGUED FOR. `report_labels_v2.csv` is `CORPUS_LABELS`, and every
holdout number in the ledger is scored against it. Its twelve finding columns take exactly five values,
and `0.28` accounts for 14445 of 52872 cells — 27.3% — joining 1:1 to `__conf 0.05` and `__verdict UNK`.
Binarised at `scores.MIDPOINT`, every one of those cells enters the ruler as a CONFIRMED NEGATIVE. That
is the U-Zeros convention, and this project adopted it by not noticing it was a choice.

WHY IT IS WORTH AN INSTRUMENT. Researcher D292 prices exactly this axis from the literature: on CheXpert,
holding images, architecture, split and labeler fixed and moving ONLY the uncertain-mention mapping,
weighted AUROC moves 0.070 (U-Zeros vs U-Ones) and 0.040 for CheXbert — a larger swing than the choice of
labeler itself, and 2.7x this project's significance bar.

WHAT THIS MODULE MEASURES, AND WHAT IT DOES NOT. It reads every holdout record twice: once as the ledger
reads it, and once with the uncertain cells DROPPED from both the truth and the predictions, per column.
The move between those two readings is not a gain and must never be quoted as one — dropping 29% of the
cells changes the population, and it removes the cases a report was equivocal about, which are the ones
a model is likeliest to be wrong on. The move bounds how much of a member's measured number is an
artefact of the convention, and nothing more.

THE PART THAT BEARS ON DECISIONS IS THE SPREAD, NOT THE OFFSET. Measured over the 318 members that are
actually on the v2 ruler, 2026-09-10: mean +0.0229, sd 0.0063, range -0.0158 to +0.0360. An offset that
were flat would be harmless — it would cancel in every comparison this project makes. It is not flat: its
sd sits AT the 0.0062 paired noise floor rather than under it, it re-ranks 8.8% of all member pairs, and
it changes three of the top ten. Every seat this project has chosen was chosen on a ruler carrying an
unstated convention that moves members unequally — but a spread this close to the floor prices that as a
RISK to seat ordering, not as a measured harm to any particular seat.

WHAT IT REFUSES. A record whose truth block carries almost no uncertain cells was not scored against
v2 at all — its arm passed a re-mapped table and the ruler moved with it. Five such records sit on
disk, and this module's first run ranked one of them in its top ten: `recal_holdannot_s2026` at
0.8964, the number `comp/members.declines` says in writing must never be quoted. The share is the
signature, `MINIMUM_UNCERTAIN` is the floor, and such a record is dropped from the sweep entirely
rather than shown with a caveat.

WHAT IT CANNOT SETTLE. Not whether a different convention would SCORE better — supervision and ruler are
the same file, so changing 0.28 moves the target and the scorer together, and there is no
radiologist-annotated set at scale here to break the tie. This instrument prices the convention's
FOOTPRINT on readings already taken; the A/B that would price the convention itself is a design question
recorded in knee-07gvn and still open.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.scores import MIDPOINT, LabelScores

logger = logging.getLogger(__name__)

UNCERTAIN = 0.28
"""The value `report_labels_v2.csv` assigns every cell whose `__verdict` is UNK. Not derived anywhere."""

TOLERANCE = 1e-6
"""How near a truth cell must sit to `UNCERTAIN` to be one. The table is a five-rung ladder, so the
nearest neighbouring rung is 0.40 away and any tolerance below that separates them identically."""

PREDICTIONS = "predictions"
TRUTH = "truth"
STUDIES = "studies"

MINIMUM_STUDIES = 200
"""Below this a record is a smoke run or a truncated frame, not a member reading."""

MINIMUM_UNCERTAIN = 0.10
"""The floor a truth block's uncertain share must clear to be `report_labels_v2.csv` at all.

v2 carries 27.3% uncertain cells; a table that re-maps them carries ~0. A record scored against a
re-mapped table was scored by a DIFFERENT RULER, so its area is not comparable to the ledger's and
the move this module measures is not defined for it — there is no convention left to price. Five
records on disk are in that state, and one of them is `recal_holdannot_s2026`, whose 0.8964
`comp/members.declines` says in writing must not be quoted. This module printed it in a top ten.
The share is the signature because the truth block does not record which table produced it."""

TOP = 10
"""How many members each ordering prints, since the question is whether the two orderings agree."""

RESOLVES = 2
"""Distinct labels a column needs before an area under its curve means anything."""


@dataclass(frozen=True)
class UnkReading:
    """One member read twice — as the ledger reads it, and with the uncertain cells dropped."""

    member: str
    area: float
    excluded: float
    uncertain_share: float
    columns: int

    def move(self) -> float:
        """What dropping the uncertain cells does to this member's number.

        Never a gain. It is the part of the member's measured area that is a property of the
        convention rather than of the member, and it is signed because a few members move DOWN.
        """
        return self.excluded - self.area

    def line(self) -> str:
        """The printed row, both readings side by side so the move is visible rather than inferred."""
        return (
            f"{self.member:<34} ledger {self.area:.4f}   no-UNK {self.excluded:.4f}"
            f"   move {self.move():+.4f}   unk {self.uncertain_share * 100:4.1f}%  cols={self.columns}"
        )


class UnkSurface:
    """Every holdout record on disk, read once as the ledger reads it and once without the UNK cells."""

    def __init__(self, uncertain: float = UNCERTAIN, cut: float = MIDPOINT) -> None:
        self.uncertain = uncertain
        self.cut = cut
        self.scores = LabelScores(cut=cut)

    def is_uncertain(self, truth: Float[NDArray[np.float64], " studies"]) -> Float[NDArray[np.bool_], " studies"]:
        """Which cells of one column carry the uncertain-mention constant."""
        return np.abs(truth - self.uncertain) < TOLERANCE

    def reading(
        self,
        member: str,
        predictions: Float[NDArray[np.float64], "studies findings"],
        truth: Float[NDArray[np.float64], "studies findings"],
    ) -> UnkReading | None:
        """This member read both ways, or `None` when no column resolves under both readings.

        A column is skipped independently in each reading, because dropping the uncertain cells can
        leave a rare column single-class — and averaging a different number of columns into the two
        means would compare two different metrics rather than two readings of one.
        """
        areas: list[float] = []
        excluded: list[float] = []
        uncertain = 0
        for index in range(min(truth.shape[1], len(TARGETS))):
            column, kept = truth[:, index], ~self.is_uncertain(truth[:, index])
            uncertain += int((~kept).sum())
            labels = (column >= self.cut).astype(np.int64)
            if min(len(np.unique(labels)), len(np.unique(labels[kept]))) < RESOLVES:
                continue
            areas.append(self.scores.area_under_curve(predictions[:, index], labels))
            excluded.append(self.scores.area_under_curve(predictions[kept, index], labels[kept]))
        if not areas:
            return None
        return UnkReading(
            member=member,
            area=float(np.mean(areas)),
            excluded=float(np.mean(excluded)),
            uncertain_share=uncertain / truth.size,
            columns=len(areas),
        )

    def sweep(self, holdouts: Path) -> list[UnkReading]:
        """Every readable holdout record under `holdouts`, ordered by the ledger's own reading."""
        readings: list[UnkReading] = []
        for record in sorted(holdouts.glob("*.holdout.npz")):
            reading = self.of(record)
            if reading is not None:
                readings.append(reading)
        return sorted(readings, key=lambda row: -row.area)

    def of(self, record: Path) -> UnkReading | None:
        """One record's reading, or `None` when the file is not a member reading at all.

        Refuses rather than guesses on a shape mismatch: a predictions block that does not match its
        truth block can only be joined by position, and a positional join produces a plausible number.
        Refuses a truth block below `MINIMUM_UNCERTAIN` for the same reason — it is a different ruler.
        """
        with np.load(record, allow_pickle=True) as held:
            if PREDICTIONS not in held or TRUTH not in held:
                return None
            predictions, truth = held[PREDICTIONS], held[TRUTH]
        if predictions.shape != truth.shape or truth.shape[0] < MINIMUM_STUDIES:
            return None
        if float(self.is_uncertain(truth.astype(np.float64)).mean()) < MINIMUM_UNCERTAIN:
            return None
        member = record.name.removesuffix(".holdout.npz")
        return self.reading(member, predictions.astype(np.float64), truth.astype(np.float64))

    @staticmethod
    def inversions(readings: list[UnkReading]) -> tuple[int, int]:
        """Pairs the two readings order differently, and pairs compared — the re-ranking, exactly.

        Counted rather than correlated because a rank correlation answers "mostly agrees" and the
        question a seat asks is "does THIS pair swap". Both are printed; only this one is decidable.
        """
        areas = np.array([row.area for row in readings])
        excluded = np.array([row.excluded for row in readings])
        swapped = 0
        for first in range(len(readings)):
            ledger = areas[first] - areas[first + 1 :]
            without = excluded[first] - excluded[first + 1 :]
            swapped += int(np.sum(ledger * without < 0))
        return swapped, len(readings) * (len(readings) - 1) // 2

    def report(self, readings: list[UnkReading], top: int = TOP) -> list[str]:
        """The two orderings, the move's distribution, and the re-ranking between them."""
        if not readings:
            return ["no holdout record carried both a predictions and a truth block"]
        moves = np.array([row.move() for row in readings])
        swapped, pairs = self.inversions(readings)
        by_excluded = sorted(readings, key=lambda row: -row.excluded)
        lines = [f"{len(readings)} members, cut {self.cut}, uncertain constant {self.uncertain}", ""]
        lines += [row.line() for row in by_excluded[:top]]
        lines += [
            "",
            f"move  mean {moves.mean():+.4f}  sd {moves.std():.4f}  min {moves.min():+.4f}  max {moves.max():+.4f}",
            f"re-ranking  {swapped} of {pairs} pairs swap = {swapped / max(pairs, 1) * 100:.1f}%",
            f"top {top} agrees on "
            f"{len({row.member for row in readings[:top]} & {row.member for row in by_excluded[:top]})}"
            f" of {top} members",
            "",
            "The move is NOT a gain. Dropping the uncertain cells drops 29% of the table and removes",
            "the studies a report was equivocal about. It bounds how much of a member's number belongs",
            "to the convention rather than to the member. The spread is what bears on a decision.",
        ]
        return lines


def main() -> None:
    """Price the ruler's uncertain-mention convention across every member already on disk."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True, help="directory of *.holdout.npz")
    command.parser.add_argument("--top", type=int, default=TOP, help="members printed per ordering")
    command.parser.add_argument("--cut", type=float, default=MIDPOINT, help="where a graded target cuts")
    arguments = command.parsed()
    surface = UnkSurface(cut=arguments.cut)
    command.report(surface.report(surface.sweep(arguments.holdouts), arguments.top))


if __name__ == "__main__":
    main()
