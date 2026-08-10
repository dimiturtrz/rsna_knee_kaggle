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
    def load(cls, report_labels: Path, train_table: Path, identifiers: list[str]) -> "Targets":
        """Read the derived table, then overwrite with annotations where a study has them."""
        derived = pd.read_csv(report_labels).set_index("StudyInstanceUID")
        annotated = pd.read_csv(train_table).set_index("StudyInstanceUID")

        scores = np.full((len(identifiers), len(TARGETS)), 0.5, dtype=np.float32)
        weights = np.zeros((len(identifiers), len(TARGETS)), dtype=np.float32)
        for row, identifier in enumerate(identifiers):
            for column, target in enumerate(TARGETS):
                if identifier in derived.index and target in derived.columns:
                    value = derived.at[identifier, target]
                    confidence = (
                        derived.at[identifier, f"{target}__conf"] if f"{target}__conf" in derived.columns else 1.0
                    )
                    if pd.notna(value):
                        scores[row, column] = float(value)  # pyrefly: ignore[bad-argument-type]
                        weights[row, column] = float(confidence) if pd.notna(confidence) else 1.0  # pyrefly: ignore[bad-argument-type]
                if identifier in annotated.index and target in annotated.columns:
                    truth = annotated.at[identifier, target]
                    if pd.notna(truth):
                        scores[row, column] = float(truth)  # pyrefly: ignore[bad-argument-type]
                        weights[row, column] = ANNOTATED_WEIGHT
        return cls(scores, weights, identifiers)

    def rows(self, positions: NDArray[np.int64]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Scores and weights for a batch of study positions."""
        return self.scores[positions], self.weights[positions]

    def coverage(self) -> dict[str, float]:
        """The share of studies carrying any weight per target — where the vocabulary is thin shows here."""
        return {target: float((self.weights[:, column] > 0).mean()) for column, target in enumerate(TARGETS)}
