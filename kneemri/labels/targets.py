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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from kneemri.models.model import TARGETS

CONFIDENCE_FLOOR = 0.25
"""What a finding the report never mentions still pulls with, as a fraction of a fully-confident one.

Ours dropped such targets entirely (weight 0.0); the public baseline uses `0.25 + 0.75*conf`. Silence
in a radiology report is weak evidence of ABSENCE — a knee report lists what was checked and found
intact — so dropping it discards evidence rather than declining to guess. Kept as a parameter because
the two behaviours are a measurable difference, not a matter of taste."""

MEASURED_AGREEMENT: tuple[tuple[float, float], ...] = (
    (0.08, 0.041),
    (0.28, 0.136),
    (0.68, 0.378),
    (0.82, 0.688),
    (0.94, 0.958),
)
"""Each level the reader emits, beside the rate the radiologist agreed at. See `Calibration`."""

ANNOTATED_WEIGHT = 3.0
"""Image-read annotations outweigh any derived row: they are the only labels read from the pixels rather
than from text, and there are 58 of them."""


@dataclass(frozen=True)
class Supervision:
    """What each report-derived level is WORTH, measured against the image annotations.

    The reader emits five values and we trained on them as if they were probabilities. Measured over
    the 57 fully-annotated studies in train.csv (696 cells, every label annotated), how often the
    radiologist actually agrees the finding is present:

        assigned   measured   95% CI            n
          0.08      0.041     [0.013, 0.068]   196
          0.28      0.136     [0.085, 0.186]   177
          0.68      0.378     [0.282, 0.474]    98
          0.82      0.688     [0.611, 0.764]   141
          0.94      0.958     [0.912, 1.004]    72

    FOUR OF FIVE fall outside their own interval, all four too high: the model was told 0.68 for cells
    that are positive 38% of the time. The levels themselves are sound — agreement is monotone across a
    span of 0.58, and the reader's `__conf` column is constant per verdict, so this gradation is real
    per-study information rather than certainty noise. It is the VALUES that were wrong.

    IT DOES NOT LEAK INTO THE NUMBER IT WILL BE JUDGED BY. The calibration is fitted on the 58
    image-read annotations; an arm is scored on the fold-0 holdout of the REPORT labels. Different
    rulers, so the comparison against wsfold0_e6 stays honest. What it does spend is some independence
    of the annotation-based check (`annot`), which is stated rather than hidden.
    """

    floor: float = 0.0
    """What a finding the report never mentions still pulls with. See CONFIDENCE_FLOOR."""

    calibrated: bool = False
    """Whether to replace each level with the rate it was measured at."""

    measured: tuple[tuple[float, float], ...] = MEASURED_AGREEMENT

    def applied(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        """Each level replaced by what it was measured to be worth; anything else left alone.

        Values the reader never emits pass through untouched rather than being snapped to a nearest
        level — a table with a different scale would otherwise be silently reinterpreted as this one.
        """
        if not self.calibrated:
            return value
        applied = value.copy()
        for level, rate in self.measured:
            applied[np.isclose(value, level)] = rate
        return applied


class Targets:
    """Per-study scores and weights, from the report-derived table and the few image annotations."""

    def __init__(self, scores: NDArray[np.float32], weights: NDArray[np.float32], identifiers: list[str]) -> None:
        self.scores = scores
        self.weights = weights
        self.identifiers = identifiers
        self.position = {identifier: index for index, identifier in enumerate(identifiers)}

    @classmethod
    def load(
        cls, report_labels: Path, train_table: Path, identifiers: list[str], supervision: "Supervision | None" = None
    ) -> "Targets":
        """Read the derived table, then overwrite with annotations where a study has them.

        `floor` sets what an unmentioned finding still pulls with: 0.0 drops it (our original
        behaviour), `CONFIDENCE_FLOOR` reproduces the public baseline's `0.25 + 0.75*conf`.
        """
        # Aligned to OUR study order once per table, rather than looked up per (study, target). The
        # loop this replaces ran 4,407 x 12 = 52,884 pandas scalar lookups and re-scanned `.columns`
        # inside the inner loop; it also needed three `float(Any)` suppressions, because a `.at` scalar
        # is untyped. Whole columns are typed, so both the cost and the suppressions go together.
        supervision = supervision or Supervision()
        floor = supervision.floor
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
                value = supervision.applied(value)
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
        cls,
        labels: Path,
        eval_labels: Path | None,
        train_table: Path,
        identifiers: list[str],
        supervision: "Supervision | None" = None,
    ) -> tuple["Targets", "Targets"]:
        """The table to TRAIN on and the table to SCORE against, which are not always the same.

        Scoring a run against the labels it trained on makes two arms incomparable: a holdout AUC can
        rise because the labels agree with themselves more, not because the model improved. When the
        training labels are the variable under test, the ruler has to be held fixed — and ideally held
        at the table the interesting arm did NOT train on, so it earns no self-agreement credit.

        A CALIBRATION IS EXACTLY THAT KIND OF VARIABLE, so it is applied to the training table ONLY.
        Recalibrating the ruler too would move every arm's score by construction and make the change
        unmeasurable — the AUC would rise because the targets agree with themselves more. That also
        means the two tables stop being the same object when a calibration is given, even though they
        are read from the same file.
        """
        supervision = supervision or Supervision()
        training = cls.load(labels, train_table, identifiers, supervision)
        if eval_labels is None:
            plain = Supervision(floor=supervision.floor)
            ruler = cls.load(labels, train_table, identifiers, plain) if supervision.calibrated else training
            return training, ruler
        return training, cls.load(eval_labels, train_table, identifiers)

    def rows(self, positions: NDArray[np.int64]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Scores and weights for a batch of study positions."""
        return self.scores[positions], self.weights[positions]

    def coverage(self) -> dict[str, float]:
        """The share of studies carrying any weight per target — where the vocabulary is thin shows here."""
        return {target: float((self.weights[:, column] > 0).mean()) for column, target in enumerate(TARGETS)}
