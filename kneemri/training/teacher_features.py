"""A teacher's per-study REPRESENTATIONS, lined up with the corpus so a student can be pulled onto them.

Beside `teacher_targets.py` and shaped like it, because the two answer the same question about different
objects: one carries twelve scores per study and the other a `d`-wide reading whose width is the
teacher's own. They share the file's join — `TeacherFile.aligned` — and nothing else, since a width is
not a fact this side may assert and twelve columns is.

**The teacher must not have seen the HOLDOUT.** A teacher fitted with `--fold -1` has read every study
including fold 0, so its features on those are memorised rather than inferred and a student pulled onto
them is scored on a holdout that flatters both. The fold-0 `_s2026` twins are the honest teachers;
enforcing it is the caller's job, which is the one assumption neither this file nor the objective can
check for itself.
"""

import os

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from kneemri.teacher_file import TeacherFile

FEATURE_TEACHER = "teacher_features"
"""The landing witness: whether REPRESENTATIONS arrived, which is separate from the term's weight.

`--feature-weight` says how hard to pull toward the teacher and `FEATURE_TERM` reports it, but a weight
is a number a run can carry while the array behind it never loaded. See `kneemri.training.landed`."""


class TeacherFeatures:
    """One teacher vector per study, in the corpus's own study order."""

    def __init__(self, embeddings: Float[np.ndarray, "n d"]) -> None:
        self.embeddings = embeddings

    @classmethod
    def from_file(
        cls, path: os.PathLike[str], identifiers: list[str], training: NDArray[np.int64]
    ) -> "TeacherFeatures":
        """Read the npz and reorder it into the CORPUS's study order, failing loudly on a gap.

        NO WIDTH IS CHECKED, and that is the difference from the targets loader rather than an omission:
        the teacher's width is whatever trunk produced the file, the projection is built from the number
        this class then reports, and a hard-coded expectation here would refuse the next teacher for
        being itself. What protects against the wrong KIND of file is the array NAME — a targets npz
        carries no `embeddings` key and raises before the card is touched.
        """
        stored = np.load(path, allow_pickle=True)
        aligned = TeacherFile.aligned(
            stored[TeacherFile.IDENTIFIERS], stored[TeacherFile.EMBEDDINGS], identifiers, training
        )
        return cls(aligned)

    def width(self) -> int:
        """How wide the teacher's reading is — the target side of the projection the objective builds."""
        return int(self.embeddings.shape[1])

    def rows(self, positions: NDArray[np.int64]) -> Float[np.ndarray, "b d"]:
        """The teacher's vectors for a batch of study positions, indexed the way the labels are.

        The NaN check is the positive witness that the alignment held for exactly the studies used, and
        it is the only place an untaught study can be caught in the act. Without it a study the file does
        not cover would descend a cosine against NaN, which poisons every weight in one step rather than
        naming what went wrong.
        """
        rows = self.embeddings[positions]
        if np.isnan(rows).any():
            message = (
                f"{int(np.isnan(rows).any(axis=1).sum())} studies in this batch carry no teacher reading, "
                "so the run is fitting studies the teacher file does not cover"
            )
            raise ValueError(message)
        return rows
