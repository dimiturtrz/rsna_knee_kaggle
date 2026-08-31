"""An ensemble's readings of the TRAINING studies, lined up with the corpus so a student can be taught them.

The labels this project trains on come off a report reader that emits five discrete levels, so two
studies read at the same level are indistinguishable to the pairwise term — and a finding present in
7% of studies leaves roughly 87% of its pairs on the same level. An ensemble scores continuously, so it
orders those pairs. That ordering is the whole of what is stored here: `Objective.pairs` reads
separation from a score and the gap from a logit, and an AUC reads nothing but order, so the file needs
no calibration, no temperature and no probabilities that mean anything.

**The teacher must not have seen the HOLDOUT**, which is a different and weaker constraint than the one
it is easy to write here. A teacher that trained on these same training studies is ordinary
distillation — that is the usual setup, and its readings on studies it fitted are partly memorised
rather than earned, which limits how much the student can learn but cannot manufacture a score. A
teacher that trained on the HOLDOUT is a broken ruler: it would leak the evaluation set's labels into
the student through this file, and the arm would read as a win with nothing behind it.

Six of the seven champion members were fitted with `--fold -1` and have therefore seen every study
including fold 0, so they CANNOT teach an arm that is scored on fold 0. The producer runs the fold-0
`_s2026` twins instead, which is a weaker ensemble (~0.869) bought for a holdout that stays honest.
"""

import os

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core.findings import TARGETS
from kneemri.teacher_file import TeacherFile

TEACHER = "teacher_targets"
"""The landing witness: whether readings ARRIVED, which is a separate fact from the term's weight.

`--distill-weight` says how hard to pull toward the teacher and `DISTILL_TERM` reports it, but a
weight is a number a run can carry while the array behind it never loaded. See `kneemri.training.landed`."""


class TeacherTargets:
    """Twelve teacher scores per study, in the corpus's own study order."""

    def __init__(self, scores: Float[np.ndarray, "n t"]) -> None:
        self.scores = scores

    @classmethod
    def from_file(cls, path: os.PathLike[str], identifiers: list[str], training: NDArray[np.int64]) -> "TeacherTargets":
        """Read the npz and reorder it into the CORPUS's study order, failing loudly on a gap or a shape.

        The reorder itself is `TeacherFile.aligned`, beside the format it reads, because the FEATURE
        loader performs the identical join and two spellings of it are how one consumer comes to
        disagree with the other about which row belongs to which study. What stays here is the part
        that is about TARGETS specifically: twelve columns, and nothing else is a teacher file.

        The column check is made BEFORE the alignment for the reason the alignment is made at all — a
        file with the wrong width is a file of the wrong KIND, most likely a features npz handed to
        `--teacher-targets`, and naming that is more useful than a shape error twelve layers down.
        """
        stored = np.load(path, allow_pickle=True)
        stored_scores = stored[TeacherFile.SCORES]
        if stored_scores.shape[1] != len(TARGETS):
            message = f"teacher readings have {stored_scores.shape[1]} columns, not the {len(TARGETS)} targets"
            raise ValueError(message)
        return cls(TeacherFile.aligned(stored[TeacherFile.IDENTIFIERS], stored_scores, identifiers, training))

    def rows(self, positions: NDArray[np.int64]) -> Float[np.ndarray, "b t"]:
        """The teacher's scores for a batch of study positions, indexed the way the labels are.

        The NaN check is the positive witness that the alignment held for exactly the studies used —
        cheap on a batch-sized array, and the only place an untaught study can be caught in the act.
        """
        rows = self.scores[positions]
        if np.isnan(rows).any():
            message = (
                f"{int(np.isnan(rows).any(axis=1).sum())} studies in this batch carry no teacher reading, "
                "so the run is fitting studies the teacher file does not cover"
            )
            raise ValueError(message)
        return rows
