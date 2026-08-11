"""The training targets: twelve graded scores per study, each with the confidence it was read at.

Only 58 of 4,407 studies carry the twelve annotations — 1.3%. Every study carries a radiology report,
and `test.csv` has no `Report` column, so text is available when fitting and absent when predicting.
Reports are therefore usable as targets, never as an inference-time input.

Two properties of the labels shape the loss:

**Graded, not thresholded.** The reporting radiologist and the annotator do not share a threshold — a
report saying *small joint effusion* can sit against a negative annotation because the annotator marked
only effusions they judged significant. "Term present implies positive" is wrong by construction; the
grade costs nothing, because the metric reads only order.

**Confidence is a weight, not a value.** A report that never mentions synovitis should pull weakly on
that head rather than assert a negative there, so the confidence multiplies the loss for that
(study, target) pair rather than moving the target toward 0.5.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from kneemri.model import TARGETS

CONFIDENCE_FLOOR = 0.25
"""What a finding the report never mentions still pulls with, as a fraction of a fully-confident one.

Ours dropped such targets entirely (weight 0.0); the public baseline uses `0.25 + 0.75*conf`. Silence
in a radiology report is weak evidence of ABSENCE — a knee report lists what was checked and found
intact — so dropping it discards evidence rather than declining to guess. Kept as a parameter because
the two behaviours are a measurable difference, not a matter of taste."""

ANNOTATED_WEIGHT = 3.0
"""Image-read annotations outweigh any derived row: they are the only labels read from the pixels rather
than from text, and there are 58 of them."""


class Targets:
    """Per-study scores and weights, from the report-derived table and the few image annotations."""

    def __init__(self, scores: NDArray[np.float32], weights: NDArray[np.float32], identifiers: list[str]) -> None:
        self.scores = scores
        self.weights = weights
        self.identifiers = identifiers
        self.position = {identifier: index for index, identifier in enumerate(identifiers)}

    @classmethod
    def load(cls, report_labels: Path, train_table: Path, identifiers: list[str], floor: float = 0.0) -> "Targets":
        """Read the derived table, then overwrite with annotations where a study has them.

        `floor` sets what an unmentioned finding still pulls with: 0.0 drops it (our original
        behaviour), `CONFIDENCE_FLOOR` reproduces the public baseline's `0.25 + 0.75*conf`.
        """
        # Aligned to OUR study order once per table, rather than looked up per (study, target). The
        # loop this replaces ran 4,407 x 12 = 52,884 pandas scalar lookups and re-scanned `.columns`
        # inside the inner loop; it also needed three `float(Any)` suppressions, because a `.at` scalar
        # is untyped. Whole columns are typed, so both the cost and the suppressions go together.
        index = pd.Index(identifiers)
        derived = pd.read_csv(report_labels).set_index("StudyInstanceUID").reindex(index)
        annotated = pd.read_csv(train_table).set_index("StudyInstanceUID").reindex(index)

        scores = np.full((len(identifiers), len(TARGETS)), 0.5, dtype=np.float32)
        weights = np.zeros((len(identifiers), len(TARGETS)), dtype=np.float32)
        # float64 for the confidence arithmetic, not float32. The loop this replaces computed
        # `floor + (1 - floor) * raw` in Python floats and stored the float32 result; doing the
        # multiply in float32 instead lands one ULP away on 22,102 of the pairs. Harmless to training,
        # but it would make this refactor unprovable, and an equality oracle is worth more than an ULP.
        ones = np.ones(len(identifiers), dtype=np.float64)
        for column, target in enumerate(TARGETS):
            if target in derived.columns:
                value = derived[target].to_numpy(dtype=np.float32)
                confidence = f"{target}__conf"
                # A missing confidence COLUMN and a missing confidence CELL both mean "fully
                # confident": the reader either does not report confidence at all, or reported none
                # for this row. Neither is evidence of doubt.
                raw = derived[confidence].to_numpy(dtype=np.float64) if confidence in derived.columns else ones
                said = ~np.isnan(value)
                scores[said, column] = value[said]
                weights[said, column] = floor + (1.0 - floor) * np.nan_to_num(raw, nan=1.0)[said]
            if target in annotated.columns:
                truth = annotated[target].to_numpy(dtype=np.float32)
                read = ~np.isnan(truth)
                scores[read, column] = truth[read]
                weights[read, column] = ANNOTATED_WEIGHT
        return cls(scores, weights, identifiers)

    @classmethod
    def pair(
        cls, labels: Path, eval_labels: Path | None, train_table: Path, identifiers: list[str], floor: float = 0.0
    ) -> tuple["Targets", "Targets"]:
        """The table to TRAIN on and the table to SCORE against, which are not always the same.

        Scoring a run against the labels it trained on makes two arms incomparable: a holdout AUC can
        rise because the labels agree with themselves more, not because the model improved. When the
        training labels are the variable under test, the ruler has to be held fixed — and ideally held
        at the table the interesting arm did NOT train on, so it earns no self-agreement credit.
        """
        training = cls.load(labels, train_table, identifiers, floor)
        if eval_labels is None:
            return training, training
        return training, cls.load(eval_labels, train_table, identifiers)

    def rows(self, positions: NDArray[np.int64]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Scores and weights for a batch of study positions."""
        return self.scores[positions], self.weights[positions]

    def coverage(self) -> dict[str, float]:
        """The share of studies carrying any weight per target — where the vocabulary is thin shows here."""
        return {target: float((self.weights[:, column] > 0).mean()) for column, target in enumerate(TARGETS)}
