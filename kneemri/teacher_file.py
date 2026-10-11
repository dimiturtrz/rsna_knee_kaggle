"""The teacher-readings npz — the one place its writer and its reader agree on what is inside it.

It sits at the package root rather than beside either half because the halves are in packages that must
not see each other: the producer is an INFERENCE pass over fitted members and the consumer is a TRAINING
term, and `training is fitting only — serving and packaging never import it` is a contract this project
enforces. Spelling the array names twice would satisfy the linter and lose the thing that matters — a
rename would then surface as a `KeyError` against a file that cost thirty-six minutes of card to write.
"""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray


@dataclass(frozen=True)
class TeacherFile:
    """One teacher file on disk: the array names, and the single call that writes them under those names.

    The PATH is the instance because every call that touches the file is about one file — the writers
    and the merge each threaded it as their first argument. `aligned` stays a staticmethod: it reorders
    arrays already read, and knows nothing about where they came from.
    """

    path: os.PathLike[str]

    IDENTIFIERS: ClassVar[str] = "identifiers"
    """The study each row belongs to, stored beside the scores so a reader can REORDER rather than trust."""

    SCORES: ClassVar[str] = "scores"
    """One row per study, one column per target, in the corpus's twelve-target order."""

    EMBEDDINGS: ClassVar[str] = "embeddings"
    """One row per study, one column per feature of the TEACHER'S width — a representation, not a verdict.

    A DISTINCT NAME rather than a second file format, and that is the whole of its value: the two files
    are both `identifiers` plus a float matrix, so a shared name would let `--teacher-features
    raptor_teacher.npz` load twelve probabilities as a twelve-wide 'embedding', build a projection into
    it, descend a cosine, and report a representation arm that distilled logits. Under separate names
    the same mistake is a `KeyError` before the card is touched.
    """

    def write(self, identifiers: list[str], scores: Float[np.ndarray, "n t"]) -> None:
        """Write the two arrays under the names the reader looks them up by.

        A call rather than two constants the producer spells into its own `np.savez`: that call names its
        arrays as KEYWORDS, so a constant it never mentions could not keep it honest, and a shared call
        is the only form in which one half cannot drift from the other.
        """
        np.savez(self.path, identifiers=np.array(identifiers), scores=scores)

    def write_embeddings(self, identifiers: list[str], embeddings: Float[np.ndarray, "n d"]) -> None:
        """Write a teacher's REPRESENTATIONS under the name the feature side looks them up by.

        Beside `write` and not a flag on it, for the reason the class docstring gives: the array name is
        a KEYWORD at the one call that produces it, so the two formats cannot be confused by a caller
        that passed the wrong argument — they are different calls.
        """
        np.savez(self.path, identifiers=np.array(identifiers), embeddings=embeddings)

    @staticmethod
    def aligned(
        stored_identifiers: Sequence[object],
        values: Float[np.ndarray, "m d"],
        identifiers: list[str],
        training: NDArray[np.int64],
    ) -> Float[np.ndarray, "n d"]:
        """Reorder a teacher file's rows into the CORPUS's study order, NaN everywhere it does not teach.

        THE ONE COPY OF THE REORDER, and it is shared for the reason the module exists rather than for
        tidiness: a misaligned teacher trains every study against ANOTHER STUDY'S reading while every
        log line, gate and ledger row continues to read normally, and a second spelling of the join is
        exactly how one consumer would come to disagree with the other about which row is whose.

        ONLY THE TRAINING POSITIONS ARE REQUIRED TO BE PRESENT: a teacher that had read this run's
        holdout would be a teacher that had been SCORED on it, which is the leak both consumers exist on
        the safe side of. Every other row is NaN rather than a plausible number, so a fold mismatch that
        reaches a batch surfaces on the first step instead of as a quiet value the student can learn.
        """
        position = {str(name): index for index, name in enumerate(stored_identifiers)}
        taught = [identifiers[index] for index in training]
        missing = [name for name in taught if name not in position]
        if missing:
            message = (
                f"{len(missing)} of the {len(taught)} training studies have no teacher reading, first {missing[0]}"
            )
            raise KeyError(message)
        rows = np.full((len(identifiers), values.shape[1]), np.nan, dtype=np.float32)
        rows[training] = values[[position[name] for name in taught]].astype(np.float32)
        return rows

    def merged(self, parts: Sequence[os.PathLike[str]], array: str = SCORES) -> int:
        """Join shard files into this one, and return how many studies the joined file holds.

        A teacher pass is DECODE-bound — the card sat at 0% while pydicom worked — so the pass is run as
        several processes over disjoint slices of the same split. The join lives here, beside `write`, for
        the reason the module exists at all: the two array names are spelled in exactly one place, and a
        merge that spelled them itself would be the drift this file was written to make impossible.

        WHICH matrix is a parameter because both formats shard identically — the join is over the study
        axis and knows nothing about what a row means — and naming it at the call keeps a features merge
        from silently writing a `scores` key that the feature loader would then refuse to find.
        """
        loaded = [np.load(part) for part in parts]
        identifiers = np.concatenate([part[TeacherFile.IDENTIFIERS] for part in loaded])
        values = np.concatenate([part[array] for part in loaded])
        writers = {TeacherFile.SCORES: self.write, TeacherFile.EMBEDDINGS: self.write_embeddings}
        writers[array]([str(name) for name in identifiers], values)
        return len(identifiers)
