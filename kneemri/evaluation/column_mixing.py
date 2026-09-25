"""Mix the twelve prediction COLUMNS into each other, priced honestly. `python -m kneemri.evaluation.column_mixing`.

THE ONE OPERATION THE METRIC LEAVES ALIVE. `kneemri-monotone` shipped a monotone transform of the
champion's table and read the champion's exact score, so the board metric is rank-based per column and
every within-column rescaling — calibration, smoothing, prevalence matching — is arithmetically dead.
A CROSS-COLUMN blend is not in that dead class: `Fracture := rescale(Fracture + 0.25 * Contusion)` mixes
a second finding's information into the first, which changes within-column ORDER and therefore can move
the score. That hand-tuned rule already reads +0.0632 on the gold Fracture column with p = 0.0022, but it
is a TRAINING-LABEL rule (`core.data.labels.table_correction`) applied before a model is fitted. This
module asks the different question: does the same structure pay at SERVE time, on the predictions, with
all twelve coefficients learned rather than one coefficient guessed?

WHAT IS FITTED. For each finding `j`, a ridge from all twelve prediction columns onto that finding's
truth. Twelve fits make a 12x12 mixing matrix; the identity matrix is exactly the table we ship today, so
the baseline is inside the model rather than beside it. Predictions are rank-normalised first — the metric
only reads order, so column scale is not information, and leaving it in would let a confident column
dominate a fit for reasons the score cannot see.

THE TRAP, AND WHAT IS DONE ABOUT IT. A 144-coefficient matrix fitted on the holdout that then reports the
number will win whatever the truth is; at 909 studies and as few as 62 Fracture positives that optimism is
larger than every effect this project measures. So nothing here is scored on studies it fitted: the
holdout is split in half, each half fits a matrix that scores the other, and the ridge penalty is chosen
by a NESTED split inside the fitting half. The baseline column is read on the SAME scoring halves, so the
candidate and its control share one ruler. This is `stack_fit`'s discipline, applied to columns instead of
members, and for the same reason: the earlier honest-selection failure there halved a gain that had been
reported as doubling with pool size.

MEASURED 2026-09-04, AND IT IS A NULL — the strong kind, with the ceiling taken. Seven fold-0 members
rank-averaged over the 909-study holdout (band_dense, band_span, base_dense, calib_span,
depth_regions_narrow, grade_ladder, through_dense). Mixing loses at every fixed penalty and the loss is
monotone in how much mixing is allowed: -0.0072 at 0.1, -0.0055 at 1, -0.0044 at 10, -0.0058 at 100,
-0.0022 at 1e3, +0.0001 at 1e5 (the identity, which is the control landing where it must). So the CEILING
of this method — the best penalty picked with hindsight, which no honest procedure could have chosen — is
still -0.0022, below the shipped table. Serve-time column mixing does not pay, and the reason is the one
the null was written for: the members' columns already carry each other's information.

AND AGAIN UNDER THE PUBLISHED ESTIMATOR, WHICH IS WHAT CLOSES THE MECHANISM. Those numbers are dense L2,
so alone they bound ridges, not cross-column structure. The published form (Srivastava et al.'s
Label-Graph Refinement, `A` zero-diagonal and L1-regularised, macro AUC) is `SparseGraph` in
`mixing_rules`, and it has the prior L2 cannot express: a couple of pairs real, the rest exactly zero.
It reads -0.0088 at 1e-4, -0.0064 at 1e-3, -0.0041 at 3e-3, -0.0019 at 1e-2, -0.0024 at 3e-2, and
+0.0000 at 0.1 — every cross term zeroed, the shipped table reproduced to the digit. Sparse ceiling
-0.0019 against dense -0.0022: the right prior buys three ten-thousandths, a third of the noise floor.
Two estimators bracketing the shrinkage prior from both ends and landing 0.0003 apart is a statement
about the MECHANISM. One of them would only have been a statement about ridges.

A SECOND READING, ABOUT THE PROCEDURE RATHER THAN THE METHOD. The nested penalty choice returns -0.0186,
worse than EVERY fixed penalty in its own grid including the two extremes. An inner half of about 227
studies cannot choose between six decades of shrinkage: it picks a loose penalty on noise and pays for it
out of sample. That is a fact about nested selection at this holdout size, not about mixing, and it is
worth carrying to the next thing that wants to select a hyperparameter on a fold of this corpus.

WHAT A NULL WOULD MEAN. That the members' columns already carry each other's information — that a trunk
reading Contusion has, by the time it writes Fracture, used what it saw. That is a real possibility and it
is why this is worth one CPU run and no card: the mechanism is cheap to refute and the refutation is
informative about every future cross-label proposal.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.member_pool import MemberPool
from kneemri.evaluation.mixing_rules import RULES, MixingRule, RidgeShrink
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.stack_fit import MemberHoldout
from kneemri.inference.vote import Vote

logger = logging.getLogger(__name__)

SPLIT_SEED = 0
"""The permutation that halves the holdout. Fixed, because a seed swept is a hyperparameter fitted."""

NESTED_SHARE = 0.5
"""How much of a fitting half fits the penalty candidates, the rest choosing between them. Half, because
the choosing part is what carries the selection error and it is already the smaller quantity."""


@dataclass(frozen=True)
class MixingResult:
    """One finding's before and after, with the study counts that make the pair readable."""

    finding: str
    scored: int
    positives: int
    plain: float
    mixed: float

    @property
    def delta(self) -> float:
        """The mixed column's area minus the shipped column's, on studies neither fitted."""
        return self.mixed - self.plain

    def line(self) -> str:
        """The printed row: the finding, both areas, the signed delta and the positives behind it."""
        return (
            f"{self.finding:<18} plain {self.plain:.4f}  mixed {self.mixed:.4f}  "
            f"{self.delta:+.4f}  n={self.scored} pos={self.positives}"
        )


class ColumnMixing:
    """Fit a 12x12 mixing matrix out of sample and report what it bought, per finding."""

    def __init__(self, seed: int = SPLIT_SEED, rule: MixingRule | None = None, fixed: float | None = None) -> None:
        self.seed = seed
        self.rule = rule or RidgeShrink()
        self.fixed = fixed

    def chosen_penalty(
        self,
        design: Float[NDArray[np.float64], "studies findings"],
        truth: Int[NDArray[np.int64], "studies findings"],
        column: int,
    ) -> float:
        """The ridge strength picked by a split INSIDE the fitting half, never by the scoring half.

        Picking it on the scoring half is the failure `stack_fit` already paid for once: it turns the
        penalty into twelve more fitted parameters and the reported gain into selection noise.
        """
        if self.fixed is not None:
            return self.fixed
        inner = int(len(design) * NESTED_SHARE)
        best, chosen = -np.inf, self.rule.penalties[0]
        for penalty in self.rule.penalties:
            weights = self.rule.row(design[:inner], truth[:inner, column], penalty, column)
            area = LabelScores.area_under_curve(design[inner:] @ weights, truth[inner:, column])
            if not np.isnan(area) and area > best:
                best, chosen = area, penalty
        return chosen

    def matrix(
        self,
        design: Float[NDArray[np.float64], "studies findings"],
        truth: Int[NDArray[np.int64], "studies findings"],
    ) -> Float[NDArray[np.float64], "findings findings"]:
        """The 12x12 matrix itself: one ridge row per finding, each at its own chosen penalty."""
        rows = [
            self.rule.row(design, truth[:, column], self.chosen_penalty(design, truth, column), column)
            for column in range(truth.shape[1])
        ]
        return np.stack(rows, axis=1)

    def scored(
        self,
        table: Float[NDArray[np.float64], "studies findings"],
        truth: Int[NDArray[np.int64], "studies findings"],
    ) -> list[MixingResult]:
        """Score mixed against shipped on studies the matrix never saw, both directions of the split.

        Both directions because half of 909 is small enough that one direction's number is a coin as much
        as a reading, and averaging the two costs nothing once the fits are being done at all.
        """
        normalised = Vote.ranks(table)
        drawn = np.random.default_rng(self.seed).permutation(len(normalised))
        halves = np.zeros(len(normalised), dtype=bool)
        halves[drawn[: len(normalised) // 2]] = True
        mixed = np.zeros_like(normalised)
        for side in (halves, ~halves):
            weights = self.matrix(normalised[side], truth[side])
            mixed[~side] = normalised[~side] @ weights
        return [
            MixingResult(
                finding=finding,
                scored=len(normalised),
                positives=int(truth[:, column].sum()),
                plain=LabelScores.area_under_curve(normalised[:, column], truth[:, column]),
                mixed=LabelScores.area_under_curve(mixed[:, column], truth[:, column]),
            )
            for column, finding in enumerate(TARGETS)
        ]


def main() -> None:
    """Rank-average a directory of member holdouts into one table, then price mixing its columns."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True)
    command.parser.add_argument(
        "--only", nargs="+", default=None, help="member names; every recorded member by default"
    )
    command.parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    command.parser.add_argument("--rule", choices=sorted(RULES), default=RidgeShrink.name)
    command.parser.add_argument(
        "--sweep",
        action="store_true",
        help="price every fixed penalty instead of choosing one nested; the honest number is the nested run",
    )
    parsed = command.parsed()
    names = MemberPool.at(parsed.holdouts).names(parsed.only)
    members = MemberHoldout.aligned([MemberHoldout.loaded(parsed.holdouts / f"{name}.holdout.npz") for name in names])
    table = np.mean([Vote.ranks(member.predictions) for member in members], axis=0)
    rule = RULES[parsed.rule]
    fixed = list(rule.penalties) if parsed.sweep else [None]
    for penalty in fixed:
        results = ColumnMixing(parsed.seed, rule, penalty).scored(table, members[0].truth)
        deltas = [result.delta for result in results]
        if penalty is None:
            for result in results:
                logger.info(result.line())
        label = "NESTED" if penalty is None else f"penalty {penalty:g}"
        logger.info(
            f"{rule.name} {label}: mean {np.mean(deltas):+.4f} worst {min(deltas):+.4f} best {max(deltas):+.4f}"
        )
    logger.info(f"{len(members)} members, {len(table)} studies")


if __name__ == "__main__":
    main()
