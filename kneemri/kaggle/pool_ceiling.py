"""What the BEST LINEAR COMBINATION of this pool scores — the bound selection cannot answer.

`python -m kneemri.kaggle.pool_ceiling [--folds 10]`

WHY THIS EXISTS. Two readings say the pool is spent, and both of them are about SELECTION.
`priceable_blend` walks subsets and stops at 0.8764; `seat_requirement` prices every unseated member
against the best of those subsets and finds the best eighth seat worth +0.0002 against a +0.0117
requirement. Both hold members at EQUAL WEIGHT and ask which ones to keep. Neither answers the wider
question that decides whether external data is a necessity or a preference: **is +0.0117 reachable by
any weighting of what is already owned?**

If a cross-fitted stack over all 206 members lands near the same 0.875, then the ceiling is a property
of the POOL's information rather than of the rule used to combine it, and no combiner, weighting or
selection work can reach first place. That closes the last cheap route by measurement instead of by
inference, and it costs no card and no slot.

WHY CROSS-FITTED AND NOT IN-SAMPLE. An in-sample stack of 206 features on ~900 studies scores near 1.0
and bounds nothing — it is the overfit, not the ceiling. Weights are therefore fitted on nine tenths of
the studies and read on the tenth, stitched into one out-of-fold column per label, which is the honest
estimate of what a stack would DO. The penalty is chosen by looking at those out-of-fold scores, which
biases the reported number UP; that is deliberate. A number biased upward that still misses 0.8862 is
the decisive direction.

WHAT IT IS NOT. This is an instrument, not a proposal. The combiner and per-label weighting axes are
closed (`interpretations/kneemri/`), and nothing here is meant to ship: the number is a BOUND on the
pool, read under the same rank ruler as every other offline number in the ledger.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from core import cli
from core.bootstrap import StudyBootstrap
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.inference.vote import RankAverage
from kneemri.kaggle.priceable_blend import PriceableBlend

logger = logging.getLogger(__name__)

TARGET_SCORE = 0.8862
"""The offline score first place costs, from `2026-09-06_what-first-place-costs-in-member-points.md`."""

DEFAULT_FOLDS = 10
"""Ten, so the weights are fitted at nearly full data — the regime every ledger number was taken in."""

PENALTIES = (0.01, 0.1, 1.0, 10.0)
"""The stack's L2 grid. Read on the out-of-fold score, which biases the reported ceiling upward."""

MAX_ITERATIONS = 2000
"""Enough for a 200-feature logistic to converge; a warning here would be a silently truncated fit."""

SEED = 2026
"""The fold split, fixed, because a ceiling that moves with the shuffle is not a ceiling."""

BOTH_CLASSES = 2
"""A training fold needs a positive and a negative; one class alone carries no information to fit."""


@dataclass(frozen=True)
class CeilingReading:
    """One stack, at one penalty: what it scored out of fold, over how many members."""

    penalty: float
    score: float
    members: int


@dataclass(frozen=True)
class ColumnReading:
    """One finding: what the best stack reaches on it, and what the single best member reaches alone."""

    label: str
    stacked: float
    best_alone: float
    positives: int
    interval: tuple[float, float]

    @property
    def gain(self) -> float:
        """What weighting all 188 members buys over the best one of them, on this finding alone."""
        return self.stacked - self.best_alone

    @property
    def width(self) -> float:
        """How wide the finding's own sampling band is — a deficit narrower than this is not read."""
        return self.interval[1] - self.interval[0]


class PoolCeiling:
    """Fits every owned member as a weighted stack and reads what the pool can reach at best."""

    def __init__(self, models: Path, folds: int = DEFAULT_FOLDS) -> None:
        self.models = models
        self.folds = folds
        self.blend = PriceableBlend(models, "rank", 1)

    def pool(self) -> Pool:  # devtools-ignore: test-mirror
        """Every priceable member on the corpus ruler, on the studies they all share."""
        members = self.blend.on_the_corpus_ruler(self.blend.candidates())
        mix = RecordMix(members, foreign=True)
        mix.narrow()
        return mix.per_member(RankAverage())

    def features(self, pool: Pool) -> Float[np.ndarray, "studies members targets"]:
        """Each member's per-label RANKS, which is the space the ledger's ruler reads in.

        Ranks rather than raw sigmoids because members are calibrated differently and a stack fitted on
        raw outputs would spend its weights undoing that instead of weighting evidence. The ranking uses
        no labels, so taking it over the whole split leaks nothing the fold split has to protect.
        """
        return np.stack([RankAverage.ranks(member) for member in pool.predictions], axis=1)

    def out_of_fold(self, pool: Pool, penalty: float) -> NDArray[np.float64]:
        """The stack's prediction for every study, always made by weights that never saw that study.

        A label whose training fold holds one class gets a constant column: no information, scored as
        chance, rather than a crash on a rare finding — which is exactly where the folds go single-class.
        """
        features = self.features(pool)
        studies = features.shape[0]
        assignment = np.random.default_rng(SEED).permutation(studies) % self.folds
        stacked = np.full(pool.truth.shape, 0.5)
        for label in range(pool.truth.shape[1]):
            for fold in range(self.folds):
                held = assignment == fold
                targets = (pool.truth[~held, label] > MIDPOINT).astype(np.int64)
                if len(np.unique(targets)) < BOTH_CLASSES:
                    continue
                model = LogisticRegression(C=penalty, max_iter=MAX_ITERATIONS)
                model.fit(features[~held, :, label], targets)
                stacked[held, label] = model.predict_proba(features[held, :, label])[:, 1]
        return stacked

    def readings(self, pool: Pool) -> list[CeilingReading]:
        """The stack at each penalty, scored out of fold under the competition's own mean."""
        scores = LabelScores()
        return [
            CeilingReading(
                penalty=penalty,
                score=scores.mean(self.out_of_fold(pool, penalty), pool.truth),
                members=len(pool),
            )
            for penalty in PENALTIES
        ]

    def columns(self, pool: Pool, penalty: float) -> list[ColumnReading]:
        """Per finding: the stack against the best single member, weakest finding first.

        The mean hides where the missing points are. A finding whose stack sits far below the others is
        a place a NEW member could still pay; a finding where the stack cannot beat its own best member
        is one where more of the same evidence does nothing, however it is weighted.
        """
        scores = LabelScores()
        out_of_fold = self.out_of_fold(pool, penalty)
        stacked = scores.per_label(out_of_fold, pool.truth)
        alone = [scores.per_label(prediction, pool.truth) for prediction in pool.predictions]
        bands = StudyBootstrap().per_label(
            pool.truth.shape[0], lambda drawn: scores.per_label(out_of_fold[drawn], pool.truth[drawn])
        )
        positive = pool.truth > MIDPOINT
        readings = [
            ColumnReading(
                label=label,
                stacked=value,
                best_alone=max(one[label] for one in alone),
                positives=int(positive[:, column].sum()),
                interval=bands[label],
            )
            for column, (label, value) in enumerate(stacked.items())
        ]
        return sorted(readings, key=lambda reading: reading.stacked)

    def report(self, target: float = TARGET_SCORE, pool: Pool | None = None) -> list[str]:
        """The flat blend, the best stack, and the distance both of them still sit from first place."""
        pool = self.pool() if pool is None else pool
        scores = LabelScores()
        flat = scores.mean(pool.combined(list(range(len(pool)))), pool.truth)
        readings = self.readings(pool)
        best = max(readings, key=lambda reading: reading.score)
        lines = [
            f"{len(pool)} members, {pool.truth.shape[0]} shared studies, {self.folds}-fold cross-fitted",
            f"flat rank-average over all of them: {flat:.4f}",
        ]
        lines += [f"  stack at C={reading.penalty:<6g} out-of-fold {reading.score:.4f}" for reading in readings]
        lines.append(f"best stack: {best.score:.4f} at C={best.penalty:g} — chosen on the number it reports")
        lines.append(f"first place needs {target:.4f}: the best weighting of this pool is {best.score - target:+.4f}")
        lines.append("per finding, weakest first — stack vs the best single member:")
        lines += [
            f"  {reading.label:<20} stack {reading.stacked:.4f}  alone {reading.best_alone:.4f}"
            f"  {reading.positives:>4d}+  band {reading.width:.4f}"
            for reading in self.columns(pool, best.penalty)
        ]
        return lines


def main() -> None:
    """Read the pool's ceiling under any weighting, not merely under the subsets selection can reach."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="cross-fitting folds")
    command.parser.add_argument("--target", type=float, default=TARGET_SCORE, help="the score first place costs")
    arguments = command.parsed()
    root = command.rooted(arguments)
    command.report(PoolCeiling(root.models(), arguments.folds).report(arguments.target))


if __name__ == "__main__":
    main()
