"""A draw that spends the epoch on ONE column's positives: `python -m kneemri.evaluation.column_spread`.

WHAT IT IS FOR. `column_headroom` says the block's remaining macro sits in the weak columns, and two
forms of the weak-column fix have now been priced: the DATA form (foreign tables carrying that column)
and the LOSS form (`--focus-label`, per-label weighting). Neither moved the macro. The third form is
the DRAW — spend more of the epoch on the studies that are positive there — and it is the only one of
the three that changes what the optimiser SEES rather than what it is told about what it saw.

THE WEIGHT IS THE VERDICT, NEVER THE SOFT VALUE. A labels table carries a probability and a `__verdict`
per finding, and the probability of an unread finding is a prior (0.28 here), not evidence. Weighting by
it would tilt the epoch toward the studies the reader said nothing about, which is the opposite of the
intended draw and would look like a working sampler the whole way.

THE CONTRAST IS AN ARGUMENT, NOT A TASTE. `Disagreed` normalises by the mean, so a spread of `ratio` on
positives and 1.0 elsewhere draws a positive exactly `ratio` times as often as a negative WHATEVER the
prevalence is — which is the property the disagreement spread did not have (`disagreed_sampler` found
its first alpha gave a 1.36x contrast, a null by construction, and had to be re-pitched at 1.9x). So the
knob is the contrast itself, and this prints the epoch MASS the named positives end up holding, which is
the number an arm header should pre-register against.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.findings import STUDY, TARGETS

logger = logging.getLogger(__name__)

VERDICT = "__verdict"
"""The suffix of the column carrying the reader's call, beside the finding's probability."""

POSITIVE = "YES"
"""The one verdict that counts as a positive. `UNK` is an unread finding, not a negative and not a
positive, and it draws with the negatives because there is nothing there to spend an epoch on."""

BASE = 1.0
"""What a study not positive on any named column weighs. The ratio is relative to this, so it is the
unit of the scale rather than a tunable — moving it and `ratio` together changes nothing."""


class ColumnSpread:
    """Per-study draw weights that favour the studies a named finding was actually read positive on."""

    def __init__(self, studies: list[str], positive: NDArray[np.bool_]) -> None:
        self.studies = studies
        self.positive = positive

    @classmethod
    def over(cls, labels: Path, columns: tuple[str, ...]) -> "ColumnSpread":
        """Which studies are read POSITIVE on any of the named findings, in the table's own order."""
        unknown = [name for name in columns if name not in TARGETS]
        if unknown:
            message = f"{unknown} are not findings; the columns are {TARGETS}"
            raise ValueError(message)
        table = pd.read_csv(labels)
        verdicts = np.array([table[f"{name}{VERDICT}"].to_numpy() == POSITIVE for name in columns])
        positive = np.asarray(verdicts.any(axis=0), dtype=np.bool_)
        return cls(studies=[str(name) for name in table[STUDY]], positive=positive)

    def spread(self, ratio: float) -> Float[NDArray[np.float64], " studies"]:
        """The weight per study: `ratio` on a positive, the unit elsewhere."""
        return np.where(self.positive, ratio, BASE).astype(np.float64)

    def mass(self, ratio: float) -> float:
        """The share of the epoch the named positives hold at this contrast — what an arm pre-registers.

        The draw is proportional to the weight, so this is the weight mass on the positives over the
        total, and it says the thing a contrast alone does not: a 6x tilt on a column read positive in
        2% of studies still spends under an eighth of the epoch there.
        """
        weights = self.spread(ratio)
        return float(weights[self.positive].sum() / weights.sum())


def main() -> None:
    """Write the spread `--sample-spread` reads, and print the mass it buys."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, required=True, help="the table whose verdicts decide")
    command.parser.add_argument("--column", action="append", required=True, help="a finding, repeatable")
    command.parser.add_argument("--ratio", type=float, required=True, help="how many times a positive draws")
    command.parser.add_argument("--out", type=Path, required=True, help="the npz `--sample-spread` reads")
    arguments = command.parsed()

    spread = ColumnSpread.over(arguments.labels, tuple(arguments.column))
    weights = spread.spread(arguments.ratio)
    np.savez(arguments.out, studies=np.array(spread.studies), spread=weights)
    command.report(
        [
            f"{len(spread.studies)} studies, {int(spread.positive.sum())} positive on {arguments.column}",
            f"prevalence {spread.positive.mean():.4f}, contrast {arguments.ratio}x",
            f"the named positives hold {spread.mass(arguments.ratio):.4f} of the epoch",
            f"wrote {arguments.out}",
        ]
    )


if __name__ == "__main__":
    main()
