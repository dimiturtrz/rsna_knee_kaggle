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

        Reordering rather than trusting the file's order is the discipline both the label loader and
        the report term follow, and it matters more here than in either: a misaligned teacher trains
        every study against ANOTHER STUDY'S ORDER while every log line, every gate and every ledger row
        continues to read normally, and the arm then reports a clean refutation of a mechanism that was
        never tested. There is no oracle downstream that would catch it, so the check is at load.

        ONLY THE TRAINING POSITIONS ARE REQUIRED TO BE PRESENT, and that is the file's whole shape: a
        teacher that read this run's holdout would be a teacher that had been SCORED on it, which is the
        leak this mechanism exists on the safe side of. Every other row is filled with NaN rather than
        with a plausible number, so a fold mismatch that reaches a batch surfaces as a NaN loss on the
        first step instead of as a quiet 0.5 the student can learn.
        """
        stored = np.load(path, allow_pickle=True)
        position = {str(name): index for index, name in enumerate(stored[TeacherFile.IDENTIFIERS])}
        taught = [identifiers[index] for index in training]
        missing = [name for name in taught if name not in position]
        if missing:
            message = (
                f"{len(missing)} of the {len(taught)} training studies have no teacher reading, first {missing[0]}"
            )
            raise KeyError(message)
        stored_scores = stored[TeacherFile.SCORES]
        if stored_scores.shape[1] != len(TARGETS):
            message = f"teacher readings have {stored_scores.shape[1]} columns, not the {len(TARGETS)} targets"
            raise ValueError(message)
        scores = np.full((len(identifiers), len(TARGETS)), np.nan, dtype=np.float32)
        scores[training] = stored_scores[[position[name] for name in taught]].astype(np.float32)
        return cls(scores)

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
