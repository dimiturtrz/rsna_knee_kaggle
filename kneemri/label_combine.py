"""Combine members PER LABEL, and say honestly what that is worth.

The metric is the unweighted mean of twelve independent per-label AUCs, so the combine that matches it
is per-label. What we ship is `Ensemble.combine` — one flat rank-average of every member, used for every
label. The only other thing in the repository is `member_curve.oracle`, which picks the best member per
label USING THE ANSWERS: a ceiling that prices this work and can never be shipped. Between the flat
average and the cheat there was nothing, and this module is that thing.

THE TRAP IT IS BUILT AROUND. Choosing members per label on the holdout that then reports the number is
selection optimism, and it is severe here: twelve choices made on 881 studies with as few as 62 Fracture
positives. The in-sample number will look good whether or not the strategy is real. So the deliverable
is not the combine — it is the combine WITH an estimator that fits the choice on studies it is not
scored on.

AND IT CARRIES ITS OWN CONTROL. Selecting the top `k` members per label at `k = every member` IS the
flat rank-average, so one k curve holds both the candidate and its matched baseline under the SAME
estimator. Comparing a split-half number against a whole-holdout one would compare two rulers.

No GPU: arithmetic over the arrays `HoldoutPredictions.cached` returns.
"""

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from kneemri.ensemble import Ensemble
from kneemri.model import TARGETS
from kneemri.scores import LabelScores

UNDEFINED = -np.inf
"""Where a label has one class present its AUC is nan, and a nan must sort LAST rather than compare
false against every candidate and land wherever the sort happens to leave it."""


class LabelCombine:
    """Per-label member selection, and the split-half estimate of what it buys."""

    def __init__(self, scores: LabelScores) -> None:
        self.scores = scores

    def ranking(
        self, per_member: list[NDArray[np.float64]], truth: Float[np.ndarray, "studies targets"]
    ) -> dict[str, list[int]]:
        """Members ordered best-first FOR EACH LABEL, by their AUC on the given studies.

        The studies are an argument rather than the whole holdout precisely so the caller can fit this
        on one half and apply it to the other — which is the only reason the number this module produces
        means anything.
        """
        per_label = [self.scores.per_label(member, truth) for member in per_member]
        order = range(len(per_member))
        return {
            target: sorted(order, key=lambda index: self._finite(per_label[index][target]), reverse=True)
            for target in TARGETS
        }

    def applied(
        self, per_member: list[NDArray[np.float64]], ranking: dict[str, list[int]], count: int
    ) -> NDArray[np.float64]:
        """Rank-average the top `count` members of each label into one table.

        Through `Ensemble.combine` rather than a second rank rule written here: the ensemble's voting is
        one decision with one home, and a per-label variant that ranked even slightly differently would
        report a number the submission path cannot reproduce. Each distinct subset is combined once and
        read for the labels that chose it, because twelve labels select from a handful of subsets.
        """
        combined = np.empty_like(per_member[0])
        already: dict[tuple[int, ...], NDArray[np.float64]] = {}
        for column, target in enumerate(TARGETS):
            chosen = tuple(ranking[target][:count])
            if chosen not in already:
                already[chosen] = Ensemble.combine([per_member[index] for index in chosen])
            combined[:, column] = already[chosen][:, column]
        return combined

    def halves(self, studies: int, seed: int) -> list[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """Two (fit, report) splits of the studies — each half fits once and reports once.

        Both directions rather than one, because a single split throws away half the evidence about a
        strategy whose whole question is whether it survives being chosen elsewhere.
        """
        order = np.random.default_rng(seed).permutation(studies)
        middle = studies // 2
        first, second = order[:middle], order[middle:]
        return [(first, second), (second, first)]

    def honest(
        self,
        per_member: list[NDArray[np.float64]],
        truth: Float[np.ndarray, "studies targets"],
        count: int,
        seed: int,
    ) -> float:
        """Fit the per-label choice on one half of the studies, score it on the other. Both ways.

        THIS IS THE NUMBER. Anything measured where it was chosen prices the holdout's noise as if it
        were signal, and with twelve choices over as few as 62 positives that noise is large enough to
        manufacture a whole result.
        """
        values = []
        for fit, report in self.halves(len(truth), seed):
            ranking = self.ranking([member[fit] for member in per_member], truth[fit])
            combined = self.applied([member[report] for member in per_member], ranking, count)
            values.append(self.scores.mean(combined, truth[report]))
        return float(np.mean(values))

    def curve(
        self, per_member: list[NDArray[np.float64]], truth: Float[np.ndarray, "studies targets"], seed: int
    ) -> list[str]:
        """Every k, in-sample beside split-half — and the last row IS the flat rank-average.

        Read the split-half column only, and read it against its own last row. The in-sample column is
        printed beside it so the SIZE of the selection optimism is visible rather than assumed: if the
        two columns diverge by more than the gain, the gain is the divergence.
        """
        lines = []
        whole = self.ranking(per_member, truth)
        for count in range(1, len(per_member) + 1):
            inside = self.scores.mean(self.applied(per_member, whole, count), truth)
            control = " (= the flat rank-average)" if count == len(per_member) else ""
            lines.append(
                f"top-{count:2d} per label:  in-sample {inside:.4f}   split-half "
                f"{self.honest(per_member, truth, count, seed):.4f}{control}"
            )
        return lines

    @staticmethod
    def _finite(value: float) -> float:
        """An undefined AUC, ordered rather than left to the sort's mercy."""
        return UNDEFINED if np.isnan(value) else value
