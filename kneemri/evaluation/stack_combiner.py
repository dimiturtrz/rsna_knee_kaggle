"""Does a LEARNED combiner beat the flat vote? `python -m kneemri.evaluation.stack_combiner --members a.pt b.pt`

WHY THIS AXIS IS NOT THE ONE THAT WAS CLOSED. Nine board entries sit at 0.943-0.944 and every one of them
is the champion serving with one thing changed — a seat, a weight, the vote's rule, the pooling lens, the
calibration. `votefit` fitted the vote and `stackseat` swapped a member. Both move ONE scalar per member,
shared across all twelve findings, and every entry this project has shipped reduces the pool with a fixed
rule whose only freedom is that vector. This fits a weight per member PER FINDING, and optionally lets a
finding's score read what the members said about the OTHER eleven — a thing no vote can express at all,
since a vote sees one column at a time. The board has never seen one.

WHAT IT COSTS AND WHAT IT CANNOT SEE. No card: it reads the holdout arrays members already wrote. That is
also its ceiling — a member fitted at `--fold -1` held nothing out, so most of the champion is invisible
here exactly as it is to `seat_value`, and a gain measured on the scoreable pool is a statement about
that pool. `entry_offline` records the structural version: every board entry at 0.936 and above is
in-sample, so the offline ruler tops out below our own top band.

WHY THE ESTIMATE IS BLOCKED AND THE PENALTY IS CHOSEN INSIDE. A combiner fitted on the 909 holdout studies
and then scored on them reports its own training error, which is how a stacking result flatters itself by
several points. Rows split into contiguous blocks; each block is scored by a fit that never saw it, and
the flat vote is scored on the SAME rows so the two numbers answer one question. The ridge penalty is
picked on a split of the KEPT rows only, never on the block being reported — a penalty tuned against the
reporting rows is the same self-flattery one level up. Blocks and inner splits are contiguous rather than
sampled so that no seed enters a number somebody will later quote.

Members are ranked over all studies before fitting, held-out rows included. That is transductive, and it
is deliberate: `RankAverage`, the rule every ledger number was taken under, does exactly the same, so the
two sides of the comparison are matched. It does mean neither number would survive unchanged on a test
set arriving one study at a time.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import MIDPOINT, LabelScores

logger = logging.getLogger(__name__)

BLOCKS = 5
"""How many contiguous row blocks the honest estimate holds out one at a time. Five keeps each fit at 80%
of the studies, the smallest split where a per-finding fit over a dozen members is not itself the thing
being measured."""

PENALTIES = tuple(10.0**power for power in range(-3, 5))
"""The ridge grid, spanning from a penalty too small to bend the least-squares answer at all up to one
that crushes every weight toward zero — i.e. from the unregularised fit to the flat vote. It is a SPAN
rather than a value: the inner split picks the point, so no constant here is asserted to be right."""

BOTH_CLASSES = 2
"""A fit prices nothing without a positive and a negative on the rows it sees; below that the flat vote
stands rather than a score invented for a finding nobody could have learned."""


@dataclass(frozen=True)
class RidgeFit:
    """A fitted per-finding weight vector, kept beside the centring it was fitted under."""

    centre: Float[np.ndarray, " features"]
    coefficients: Float[np.ndarray, " features"]

    def applied(self, design: Float[np.ndarray, "rows features"]) -> Float[np.ndarray, " rows"]:
        """The combiner's score for each row — an ordering, which is all an AUC reads."""
        return (design - self.centre) @ self.coefficients


class StackCombiner:
    """A per-finding ridge over what the members said, priced against the pool's own vote rule."""

    def __init__(self, pool: Pool, blocks: int = BLOCKS, *, cross: bool = False) -> None:
        self.pool = pool
        self.blocks = blocks
        self.cross = cross
        self.scores = LabelScores()

    def ranked(self) -> Float[np.ndarray, "studies members targets"]:
        """Every member's prediction as a rank in [0, 1], the space the flat vote also works in."""
        stacked = np.stack([np.asarray(prediction, dtype=np.float64) for prediction in self.pool.predictions], axis=1)
        order = stacked.argsort(axis=0).argsort(axis=0)
        return order / max(stacked.shape[0] - 1, 1)

    def design(self, ranks: Float[np.ndarray, "studies members targets"], target: int) -> Float[np.ndarray, "s f"]:
        """The feature matrix for one finding: its own member columns, or every finding's when crossing."""
        chosen = ranks if self.cross else ranks[:, :, [target]]
        return chosen.reshape(chosen.shape[0], -1)

    def blocked(self, studies: int) -> list[NDArray[np.int64]]:
        """The row indices of each held-out block, contiguous so no seed enters the reading."""
        return [block for block in np.array_split(np.arange(studies), self.blocks) if block.size]

    @staticmethod
    def fittable(response: NDArray[np.int64]) -> bool:
        """Whether this finding carries both classes on these rows — otherwise there is nothing to learn."""
        return len(set(response.tolist())) >= BOTH_CLASSES

    @staticmethod
    def fitted(design: Float[np.ndarray, "rows features"], response: NDArray[np.int64], penalty: float) -> RidgeFit:
        """Closed-form ridge on centred features — no solver, so no convergence to argue about."""
        centre = design.mean(axis=0)
        centred = design - centre
        gram = centred.T @ centred + penalty * np.eye(centred.shape[1])
        return RidgeFit(centre, np.linalg.solve(gram, centred.T @ (response - response.mean())))

    def chosen(self, design: Float[np.ndarray, "rows features"], response: NDArray[np.int64]) -> float:
        """The penalty that scores best on the back half of the KEPT rows, never on the reported block."""
        cut = len(response) // 2
        inner, outer = slice(None, cut), slice(cut, None)
        if not (self.fittable(response[inner]) and self.fittable(response[outer])):
            return max(PENALTIES)
        graded = []
        for penalty in PENALTIES:
            fit = self.fitted(design[inner], response[inner], penalty)
            graded.append((self.scores.area_under_curve(fit.applied(design[outer]), response[outer]), penalty))
        return max(graded)[1]

    def stacked(self) -> Float[np.ndarray, "studies targets"]:
        """Every study's score, each produced by a fit that never saw its block."""
        truth = np.asarray(self.pool.truth, dtype=np.float64)
        ranks = self.ranked()
        flat = self.pool.combined(list(range(len(self.pool))))
        out = np.zeros_like(truth)
        for held in self.blocked(truth.shape[0]):
            kept = np.setdiff1d(np.arange(truth.shape[0]), held)
            for target in range(truth.shape[1]):
                design = self.design(ranks, target)
                response = (truth[kept, target] >= MIDPOINT).astype(np.int64)
                if not self.fittable(response):
                    out[held, target] = flat[held, target]
                    continue
                penalty = self.chosen(design[kept], response)
                out[held, target] = self.fitted(design[kept], response, penalty).applied(design[held])
        return out

    def reading(self) -> dict[str, float]:
        """The stacked mean AUC, the flat vote's over the same rows, and the difference between them."""
        truth = self.pool.truth
        stacked = self.scores.mean(self.stacked(), truth)
        voted = self.scores.mean(self.pool.combined(list(range(len(self.pool)))), truth)
        return {"stacked": stacked, "vote": voted, "delta": stacked - voted}


def main() -> None:
    """Print the learned combiner's mean AUC beside the flat vote's, over the same held-out rows."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--members", type=Path, nargs="+", required=True, help="member checkpoints")
    command.parser.add_argument("--blocks", type=int, default=BLOCKS, help="held-out row blocks")
    command.parser.add_argument(
        "--cross", action="store_true", help="let a finding read the members' other eleven columns"
    )
    arguments = command.parsed()

    pool = RecordMix(list(arguments.members)).per_member()
    combiner = StackCombiner(pool, arguments.blocks, cross=arguments.cross)
    reading = combiner.reading()

    command.report(
        [
            f"{len(pool)} members over {len(pool.truth)} studies, {arguments.blocks} blocks",
            f"features      {'every finding' if arguments.cross else 'own finding'}",
            f"flat vote     {reading['vote']:.4f}   ({pool.rule()})",
            f"learned stack {reading['stacked']:.4f}",
            f"delta         {reading['delta']:+.4f}",
            "a delta is a statement about the SCOREABLE pool; the champion's fold -1 seats are not in it",
        ]
    )


if __name__ == "__main__":
    main()
