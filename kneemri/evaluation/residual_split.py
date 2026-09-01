"""What KIND of error the pool's residual is: `python -m kneemri.evaluation.residual_split`. `knee-y7v3`.

Every ruler this project owns reports an AREA, and an area cannot say what the mistakes ARE. Two pools with
the same 0.87 can be failing on opposite things — one missing the positives it has never seen enough of,
the other calling positives on studies that look like them — and the fix for those is not the same fix.

Three cuts, all on the studies every member held out:

 * MISSED against CALLED. At a prevalence-matched operating point (as many calls as there are positives,
   so the threshold is not a free parameter and FN equals FP by construction per finding), a finding whose
   errors concentrate in the rare positives is a SAMPLING problem; one whose errors are spread is not.

 * UNANIMOUS against SPLIT. For each cell the ensemble gets wrong, how many members get it wrong alone.
   An error every member makes cannot be removed by reseating, reweighting or a better combiner — it is
   the floor `uncertainty_split` measures as aleatoric, seen per-cell instead of per-bit. An error only a
   minority make is one the COMBINER threw away, and that is a different lane with a different cost.

 * PER FINDING, never pooled into one number. `2026-08-27_the-supervision-gap-is-four-findings-not-an-average`
   is the standing reason: the mean hides exactly the structure a decomposition exists to expose.

WHY PREVALENCE-MATCHED AND NOT 0.5. A threshold of 0.5 on a finding with 62 positives in 909 studies makes
almost no calls at all, and the FN/FP ratio it reports is a fact about the threshold. Matching the call
count to the positive count fixes the ruler across twelve findings whose prevalences span an order of
magnitude, at the cost that FN and FP are then equal per finding — which is why the interesting quantity
here is WHICH studies, and how many members agree, not the ratio.

WHAT IT CANNOT SEE. It reads only what the pool recorded, so a member fitted at `--fold -1` is absent by
construction (`UncertaintySplit.over` refuses those by name and this module borrows that pool). And the
unanimity share is a LOWER bound on what a different model class could fix, for the same reason the
epistemic term is: separately-trained members of one family are not an honest sample of all models.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float, Int
from numpy.typing import NDArray

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import MIDPOINT
from kneemri.evaluation.uncertainty_split import UncertaintySplit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResidualRow:
    """One finding's residual: how many positives, how many the ensemble missed, and who agreed."""

    label: str
    positives: int
    missed: int
    unanimous: int
    minority: int

    @property
    def unanimous_share(self) -> float:
        """The share of the ensemble's errors that EVERY member also makes — the reseating floor."""
        return self.unanimous / self.errors if self.errors else 0.0

    @property
    def minority_share(self) -> float:
        """The share a MINORITY of members get wrong — what a better combiner could still recover."""
        return self.minority / self.errors if self.errors else 0.0

    @property
    def errors(self) -> int:
        """Missed positives plus false calls, which are equal by construction at this operating point."""
        return self.missed * 2

    def line(self) -> str:
        """This finding's row."""
        return (
            f"  {self.label}: positives {self.positives:<4} missed {self.missed:<4} "
            f"unanimous {self.unanimous_share:.1%}  minority {self.minority_share:.1%}"
        )


class ResidualSplit:
    """The pool's mistakes, cut by whether the members agreed about them."""

    def __init__(self, pool: Pool) -> None:
        self.pool = pool

    @staticmethod
    def called(
        beliefs: Float[NDArray[np.floating], "studies targets"],
        positives: Int[NDArray[np.int64], " targets"],
    ) -> Bool[NDArray[np.bool_], "studies targets"]:
        """Call exactly as many studies positive per finding as there ARE positives.

        Rank rather than threshold, so the operating point is a property of the finding's prevalence and
        not a constant that means something different in each of the twelve columns.
        """
        order = np.argsort(-beliefs, axis=0)
        ranks = np.argsort(order, axis=0)
        return ranks < positives[np.newaxis, :]

    def truth(self) -> Int[NDArray[np.int64], "studies targets"]:
        """The pool's targets binarised at the same midpoint `LabelScores` uses."""
        return (np.asarray(self.pool.truth) > MIDPOINT).astype(np.int64)

    def mistakes(
        self, beliefs: Float[NDArray[np.floating], "studies targets"]
    ) -> Bool[NDArray[np.bool_], "studies targets"]:
        """Where a set of beliefs disagrees with the truth at the prevalence-matched operating point."""
        truth = self.truth()
        return self.called(beliefs, truth.sum(axis=0)) != truth.astype(bool)

    def per_member(self) -> Bool[NDArray[np.bool_], "members studies targets"]:
        """Every member's own mistakes, each at its OWN prevalence-matched point.

        Each member is thresholded on its own scores rather than the ensemble's, so a member that is
        merely mis-scaled does not read as a member that is wrong — the question here is which STUDIES a
        member gets wrong, and a monotone rescaling must not change that answer.
        """
        return np.stack([self.mistakes(np.asarray(member)) for member in self.pool.predictions])

    def per_label(self) -> list[ResidualRow]:
        """Each finding's row, in `TARGETS` order."""
        truth = self.truth()
        ensemble = self.mistakes(self.pool.combined(list(range(len(self.pool)))))
        members = self.per_member()
        agreeing = np.where(ensemble, members.sum(axis=0), 0)
        majority = len(self.pool) / 2.0
        return [
            ResidualRow(
                label=label,
                positives=int(truth[:, column].sum()),
                missed=int((ensemble[:, column] & truth[:, column].astype(bool)).sum()),
                unanimous=int((agreeing[:, column] == len(self.pool)).sum()),
                minority=int(((agreeing[:, column] > 0) & (agreeing[:, column] < majority)).sum()),
            )
            for column, label in enumerate(TARGETS)
        ]

    def report(self) -> list[str]:
        """The denominator, then one row per finding."""
        rows = self.per_label()
        errors = sum(row.errors for row in rows)
        unanimous = sum(row.unanimous for row in rows)
        return [
            f"{len(self.pool)} members over {len(self.pool.truth)} held-out studies; "
            f"prevalence-matched operating point per finding",
            *[row.line() for row in rows],
            f"  POOLED: {errors} errors, {unanimous / max(errors, 1):.1%} unanimous across all members",
        ]


def main() -> None:
    """Cut these members' residual by finding, by miss, and by how many members agreed."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--records",
        type=Path,
        nargs="+",
        required=True,
        help="the checkpoints to decompose, sharing ONE split; every one must carry a holdout record",
    )
    arguments = command.parsed()
    command.report(ResidualSplit(UncertaintySplit.over(list(arguments.records)).pool).report())


if __name__ == "__main__":
    main()
