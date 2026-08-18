"""Scoring predictions against targets — the competition's metric, and nothing else.

This lived on `Trainer` and did not belong there: the cohesion gate measured LCOM4=3 on that class, and
the scoring pair shared no state with the training loop. The practical symptom came first — analysis
scripts were calling `Trainer.area_under_curve(None, ...)`, passing `None` as `self`, because the
function needed no trainer and there was nowhere else to put it.

The metric is the unweighted mean of twelve per-label ROC AUCs, so everything here is rank-based. There
is deliberately no calibration, no threshold, and no probability transform anywhere in this module: AUC
is invariant under any strictly increasing map, so such a step could only add a bug.
"""

import numpy as np
from numpy.typing import NDArray

from kneemri.models.model import TARGETS

MIDPOINT = 0.5
"""Where a graded target is cut so a holdout AUC exists at all — never a submitted threshold."""


RESAMPLES = 1000
"""Bootstrap draws. Enough that the 2.5th percentile is stable to about a thousandth."""

BOOTSTRAP_SEED = 2026
"""Fixed, so an interval quoted in a bead is the interval anyone else recomputes."""


class LabelScores:
    """Per-label ROC AUC and its mean, computed by rank rather than by any library."""

    @staticmethod
    def area_under_curve(predictions: NDArray[np.float64], truth: NDArray[np.int64]) -> float:
        """ROC AUC by rank, which is its definition.

        A label with only one class present is undefined rather than zero: reporting 0.5 there would
        silently drag the mean toward chance on exactly the rare findings that matter most.

        TIES TAKE MIDRANKS, and they must. This was `argsort().argsort() + 1`, which are ORDINAL ranks:
        tied predictions are ordered by their position in the array, so a tie is settled by study order.
        A CONSTANT predictor — no information at all — scored 1.0 when the positives happened to sit
        late in the corpus and 0.0 when they sat early, instead of the 0.5 it must score.

        Raw model outputs are float64 sigmoids and effectively never tie, so single-member numbers are
        unaffected. RANK-AVERAGED ENSEMBLES ARE NOT: `RankAverage.combine` maps each member onto k/(n-1)
        and averages, so exact ties are common by construction — and that is precisely the number
        `member_curve` uses to choose which members to submit.
        """
        positives, negatives = int(truth.sum()), int((1 - truth).sum())
        if positives == 0 or negatives == 0:
            return float("nan")
        order = np.argsort(predictions, kind="mergesort")
        ranked = np.empty(len(predictions), dtype=np.float64)
        _, starts, counts = np.unique(predictions[order], return_index=True, return_counts=True)
        for start, count in zip(starts, counts, strict=True):
            # 1-based midrank of a tie group: the mean of the ordinal ranks it spans.
            ranked[order[start : start + count]] = start + (count + 1) / 2
        return float((ranked[truth == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))

    def per_label(
        self,
        predictions: NDArray[np.float64],
        targets: NDArray[np.float32],
        where: NDArray[np.bool_] | None = None,
    ) -> dict[str, float]:
        """One AUC per finding, against graded targets binarised at their midpoint.

        Binarising exists only so an AUC is defined; it decides which epoch is kept and never a
        submitted score, and the metric is invariant to any increasing map of the predictions anyway.

        `where` selects which (study, finding) pairs are scored at all, per finding. It exists because
        a pair the label reader marked UNKNOWN is not a negative: `Targets.load` leaves such a pair at
        the 0.5 default, which binarises to zero, so scoring a confidence-filtered table without a
        selection would silently assert that every unknown finding was ABSENT. Dropping the pair is the
        only honest reading of "we do not know", and it is what makes a lower-noise ruler possible.
        """
        scored = np.ones(predictions.shape, dtype=np.bool_) if where is None else where
        return {
            target: self.area_under_curve(
                predictions[scored[:, column], column].astype(np.float64),
                (targets[scored[:, column], column] > MIDPOINT).astype(np.int64),
            )
            for column, target in enumerate(TARGETS)
        }

    def paired_interval(
        self,
        first: NDArray[np.float64],
        second: NDArray[np.float64],
        targets: NDArray[np.float32],
        resamples: int = RESAMPLES,
    ) -> dict[str, tuple[float, float]]:
        """The 95% interval of the per-label DIFFERENCE between two arms, by bootstrap over studies.

        The unpaired bound - each label's own sampling error from its positive count - is what
        `run_comparison` prints today, and it is far too conservative to read a knob by. Two arms scored
        on the SAME 881 studies share that error almost entirely: if a study is hard, it is hard for
        both, and the DIFFERENCE does not care. Resampling studies with replacement and recomputing the
        difference each time measures what is actually left.

        THIS IS THE DIFFERENCE BETWEEN 'UNREAD' AND AN ANSWER. Every per-label claim this project has
        made was read bare, and when the intervals were finally applied, four arms with unrelated
        mechanisms had each 'improved' Fracture inside its +/-0.061 band, and the calibrated-targets arm
        produced twelve deltas of which none cleared. That arm's entire falsifiable claim was per-label.

        Seeded, because an interval that moves between runs cannot settle an argument - and because the
        unit-test gate refuses unseeded randomness, for the same reason.
        """
        generator = np.random.default_rng(BOOTSTRAP_SEED)
        studies = targets.shape[0]
        deltas: dict[str, list[float]] = {target: [] for target in TARGETS}
        for _ in range(resamples):
            drawn = generator.integers(0, studies, size=studies)
            left = self.per_label(first[drawn], targets[drawn])
            right = self.per_label(second[drawn], targets[drawn])
            for target in TARGETS:
                difference = right[target] - left[target]
                if not np.isnan(difference):
                    deltas[target].append(difference)
        return {
            target: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
            if values
            else (float("nan"), float("nan"))
            for target, values in deltas.items()
        }

    def mean(self, predictions: NDArray[np.float64], targets: NDArray[np.float32]) -> float:
        """The competition's score: the unweighted mean over the twelve findings.

        `nanmean`, so a label that is undefined on this split does not poison the whole number — but a
        caller reading only the mean should know how many labels were undefined, which `per_label`
        shows and this deliberately does not hide.
        """
        return self.mean_of(self.per_label(predictions, targets))

    @staticmethod
    def mean_of(per_label: dict[str, float]) -> float:
        """The same aggregation, for a caller that already holds the per-label numbers.

        Split out because `Attribution` scores under occlusion and then averaged the result itself, in
        its own copy of this line. An occlusion delta is only a delta ON THE COMPETITION SCORE while
        both sides aggregate the same way, and two copies of a rule are two rules waiting to differ.
        """
        return float(np.nanmean(list(per_label.values())))
