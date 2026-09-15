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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core.findings import TARGETS
from kneemri.teacher_file import TeacherFile
from kneemri.training.config_leaf import ConfigLeaf

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


@dataclass(frozen=True)
class TeacherConfig:
    """Which teacher a run distils from and how hard each of its signals pulls."""

    teacher_targets: Path | None = ConfigLeaf.of(
        default=None,
        says=(
            "npz of an ensemble's scores for the TRAINING studies, read by --distill-weight and by --dissent-"
            "weight. Its members must not have seen the holdout this arm is scored on: a --fold -1 member saw"
            " every study, so teaching from one leaks the evaluation labels into the student."
        ),
    )
    teacher_features: Path | None = ConfigLeaf.of(
        default=None,
        says=(
            "npz of a teacher's pooled REPRESENTATIONS of the TRAINING studies, read by --feature-weight. A "
            "different file from --teacher-targets and under a different array name, so handing one where the"
            " other belongs raises before the card is touched. Its members must not have seen the holdout "
            "this arm is scored on: a --fold -1 member saw every study, so teaching from one leaks the "
            "evaluation labels into the student."
        ),
    )
    distill_weight: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "add a pairwise term at this strength asking the student to reproduce an ENSEMBLE's ordering of "
            "these studies (0 = off). Not the usual KL on softened probabilities, which targets calibration "
            "the metric discards. The reader emits five levels, so a pair of studies on the SAME level "
            "teaches the label term nothing — and a finding present in 7%% of studies strands roughly 87%% of"
            " its pairs that way. A continuous teacher orders all of them, so the gain should concentrate on "
            "the rarest findings. Requires --teacher-targets."
        ),
    )
    feature_weight: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "add a cosine term at this strength pulling what the student READ from a study onto what a "
            "TEACHER read from it, through a learned projection between the two widths (0 = off). "
            "Distillation at the representation rather than at the verdict: --distill-weight teaches the "
            "teacher's twelve-way ORDERING, which is its head, and swin_distill_fit measured that at -0.0077 "
            "against a cold control. A large value here makes the label term a probe rather than a target, "
            "which is what turns a stage into a PRETRAIN whose product is an encoder. Requires --teacher-"
            "features."
        ),
    )
    dissent_weight: float = ConfigLeaf.of(
        default=0.0,
        says=(
            "add a pairwise ranking term at this strength whose pairs are re-weighted toward the ones an "
            "ENSEMBLE orders WRONG (0 = off). The same teacher file --distill-weight reads, for the opposite "
            "purpose: not to reproduce the pool's ordering but to spend the member's capacity where that "
            "ordering fails, which is the one way to raise ensemble ambiguity without raising member error. "
            "Anchored to the labels throughout — a pair is only ever paid for in the direction the reports "
            "order it. Requires --teacher-targets."
        ),
    )
