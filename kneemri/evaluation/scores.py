"""Scoring predictions against targets — the competition's metric, and nothing else.

This lived on `Trainer` and did not belong there: the cohesion gate measured LCOM4=3 on that class, and
the scoring pair shared no state with the training loop. The practical symptom came first — analysis
scripts were calling `Trainer.area_under_curve(None, ...)`, passing `None` as `self`, because the
function needed no trainer and there was nowhere else to put it.

The metric is the unweighted mean of twelve per-label ROC AUCs, so everything here is rank-based. There
is deliberately no calibration, no threshold, and no probability transform anywhere in this module: AUC
is invariant under any strictly increasing map, so such a step could only add a bug.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray

from core.bootstrap import STUDY_RESAMPLES, StudyBootstrap
from core.findings import TARGETS

MIDPOINT = 0.5
"""Where a graded target is cut so a holdout AUC exists at all — never a submitted threshold."""


@dataclass(frozen=True)
class ScoredArm:
    """One arm's predictions against its targets, with the bootstrap draws any interval on it takes."""

    predictions: Float[np.ndarray, "studies targets"]
    targets: Float[np.ndarray, "studies targets"]
    resamples: int = STUDY_RESAMPLES

    def drawn(self, rows: Int[np.ndarray, " rows"]) -> "ScoredArm":
        """The same arm on a subset of its studies, predictions and targets selected together."""
        return ScoredArm(self.predictions[rows], self.targets[rows], self.resamples)


@dataclass(frozen=True)
class ArmPair:
    """Two arms scored on the SAME studies against the same targets — what every paired interval reads."""

    first: Float[np.ndarray, "studies targets"]
    second: Float[np.ndarray, "studies targets"]
    targets: Float[np.ndarray, "studies targets"]
    resamples: int = STUDY_RESAMPLES

    def drawn(self, rows: Int[np.ndarray, " rows"]) -> "ArmPair":
        """The same pair on a subset of its studies, both arms and the targets selected together."""
        return ArmPair(self.first[rows], self.second[rows], self.targets[rows], self.resamples)

    def difference(self, measure: Callable[[NDArray[np.float64], NDArray[np.float32]], float]) -> float:
        """`second` minus `first` under one measure, so the order is stated once rather than per caller."""
        return measure(self.second, self.targets) - measure(self.first, self.targets)

    def first_arm(self) -> ScoredArm:
        """`first` alone, for the unpaired question the pair cannot answer."""
        return ScoredArm(self.first, self.targets, self.resamples)


class LabelScores:
    """Per-label ROC AUC and its mean, computed by rank rather than by any library."""

    def __init__(self, cut: float = MIDPOINT, column_cuts: Mapping[str, float] | None = None) -> None:
        """`cut` is where a graded target becomes a positive — the midpoint unless a caller argues otherwise.

        A PARAMETER because the reader's confidence levels are not evenly trustworthy and the project has
        now measured it: our 0.68 level is 38% gold-positive pooled (MCL 0 of 6, Fracture 1 of 4), so at
        the midpoint every one of those rows enters the ruler as a positive that the radiologist says is
        not there. A cut at 0.75 asks the same question of the same predictions with the false positives
        removed from the TRUTH rather than from the model.

        It is not a default anybody may change. Every number in the ledger was read at the midpoint, and a
        cut that moved globally would silently reprice the whole record; the argument is passed by the one
        reading that wants it, which is why it lives here and not in a config.

        `column_cuts` overrides `cut` for named findings, because the project's second ruler is NOT a
        scalar: `ledger75` reads ten columns at 0.75 and leaves Medial Meniscus and Baker's at the
        midpoint, whose 0.68 rung carries real positives (`ledger75.py:44-49`). Until 2026-09-08 that
        instrument existed only inside `LedgerSeventyFive`, which takes a `HoldoutReading` rather than
        arrays, so no scorer-taking caller — `seat_value`, `member_curve`, `member_distance` — could be
        asked for it at all. That is the mechanism behind the whole cut seam: not that 0.5 was chosen,
        but that it was the only thing reachable (knee-f9aao).
        """
        self.cut = cut
        self.column_cuts: Mapping[str, float] = column_cuts or {}

    def cut_for(self, target: str) -> float:
        """This finding's cut — its own override if it has one, else the instrument's."""
        return self.column_cuts.get(target, self.cut)

    def name(self) -> str:
        """What this instrument is called, for printing BESIDE a number it produced.

        A macro is not comparable across cuts — the recut moves the whole `honest12` roster by +0.026,
        about three times the board gap — so a number printed without this is a number waiting to be
        substituted into a term fitted under the other ruler. That happened twice on 2026-09-08.
        """
        return "ledger75" if self.column_cuts else f"cut{self.cut:g}"

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
        targets: NDArray[np.float32] | NDArray[np.int64],
        where: NDArray[np.bool_] | None = None,
    ) -> dict[str, float]:
        """One AUC per finding, against graded targets binarised at this instrument's `cut`.

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
                (targets[scored[:, column], column] > self.cut_for(target)).astype(np.int64),
            )
            for column, target in enumerate(TARGETS)
        }

    def paired_interval(self, pair: ArmPair) -> dict[str, tuple[float, float]]:
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

        def delta(rows: NDArray[np.int64]) -> dict[str, float]:
            draw = pair.drawn(rows)
            left, right = (self.per_label(arm, draw.targets) for arm in (draw.first, draw.second))
            return {target: right[target] - left[target] for target in TARGETS}

        return StudyBootstrap(pair.resamples).per_label(pair.targets.shape[0], delta)

    def paired_mean_interval(self, pair: ArmPair) -> tuple[float, float]:
        """The 95% interval of the difference in the COMPETITION SCORE between two arms, paired.

        The metric every decision here turns on is the mean of twelve AUCs, and until now nothing
        measured its uncertainty for a COMPARISON. `paired_interval` pairs but reports twelve separate
        labels; `mean_interval` reports the mean but is unpaired and answers a different question. So
        every roster and ensemble judgement in this project has been read against a floor borrowed from
        elsewhere.

        WHY THE BORROWED FLOOR IS THE WRONG RULER HERE. The 0.0062 this repo quotes bounds an ARM
        against an ARM - two separately trained models, so it carries training variance. Two ROSTERS
        built from the same frozen member predictions carry none: the arrays are already written, and
        the only uncertainty left is which studies were scored. Rosters sharing most of their members
        agree per study almost exactly, and the paired difference does not care about the difficulty
        they share. Using the arm-to-arm floor on a roster delta is conservative in the direction that
        makes a real difference unreadable, which is how the seat-count curve came to be called flat
        from its own numbers rather than from the board's.

        WHAT IT DOES NOT LICENSE. A tight interval is honest for ONE PAIR NAMED IN ADVANCE. Read on the
        winner of a search over rosters it is selection-optimistic - the same trap `LabelCombine` names,
        where twelve choices made on 909 studies look like twelve findings. A searched roster still owes
        the split-half treatment in `MemberCurve.honest`; this bounds the comparison, not the search.

        Seeded, and over STUDIES rather than labels, for the reasons `paired_interval` gives.
        """

        return StudyBootstrap(pair.resamples).interval(
            pair.targets.shape[0], lambda rows: pair.drawn(rows).difference(self.mean)
        )

    def mean_interval(self, arm: ScoredArm) -> tuple[float, float]:
        """The 95% interval of ONE arm's own mean, by bootstrap over studies.

        This is the UNPAIRED companion to `paired_interval`, and it answers a different question. The
        paired interval - and the 0.0062 floor this project quotes everywhere - measures how far two
        arms scored on the SAME holdout can differ by chance. It says nothing about how far a single
        arm's ABSOLUTE mean sits from the mean it would score on another 909 studies drawn the same
        way, because the shared per-study difficulty that the pairing cancels is exactly what dominates
        an absolute number.

        WHY THAT MATTERS HERE. Every cross-corpus comparison this project has made reads an absolute
        mean: a member's 0.8502 against a forum post's 0.942, a holdout number pushed through the
        board transfer, an ensemble's offline mean set beside its board score. None of those is a paired
        comparison and none of them may use the paired floor - the floor is the wrong ruler for them, and
        it is the narrower one, so using it overstates every such claim. This measures the right one.

        Bootstrap over STUDIES, not over labels: the twelve per-label AUCs are twelve readings of the
        same 909 studies, so resampling labels would treat one sample as twelve. Seeded for the same
        reason `paired_interval` is seeded.
        """
        return StudyBootstrap(arm.resamples).interval(
            arm.targets.shape[0], lambda rows: self.mean(arm.predictions[rows], arm.targets[rows])
        )

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
