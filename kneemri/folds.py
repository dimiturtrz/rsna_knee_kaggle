"""Splitting studies so a holdout number means something.

Some reports are byte-identical across studies — a template read for an unremarkable knee. Measured on
our corpus: 4,276 distinct reports over 4,407 studies, so 177 studies sit in a duplicate group, the
largest being 37 studies sharing one Turkish all-normal template.

Every study in such a group receives the same derived target vector, so splitting a group across the
divide scores the model on a target whose source it trained on. Assigning by a hash of the report text
keeps each group whole; the leak is only 4% of studies, and removing it costs nothing.
"""

import hashlib

import numpy as np
import pandas as pd
from numpy.typing import NDArray


class ReportFolds:
    """Study-to-fold assignment by report-text hash, so duplicate reports never straddle the divide."""

    def __init__(self, folds: int = 5, seed: int = 2026) -> None:
        self.folds = folds
        self.seed = seed

    def assign(self, train_table: str, identifiers: list[str]) -> NDArray[np.int64]:
        """A fold index per study, stable across runs and machines.

        The hash is of the report text, not of the study id: two studies sharing a report must land in
        the same fold, which is the whole point.
        """
        reports = pd.read_csv(train_table).set_index("StudyInstanceUID")["Report"].fillna("").to_dict()
        assignments = np.empty(len(identifiers), dtype=np.int64)
        for index, identifier in enumerate(identifiers):
            text = str(reports.get(identifier, identifier))
            digest = hashlib.sha1(f"{self.seed}:{text}".encode("utf-8")).hexdigest()
            assignments[index] = int(digest[:8], 16) % self.folds
        return assignments

    def holdout(self, assignments: NDArray[np.int64], fold: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Training and holdout positions for one fold — fixed rather than rotated within a run."""
        holdout = np.flatnonzero(assignments == fold)
        training = np.flatnonzero(assignments != fold)
        return training, holdout
