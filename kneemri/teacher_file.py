"""The teacher-readings npz — the one place its writer and its reader agree on what is inside it.

It sits at the package root rather than beside either half because the halves are in packages that must
not see each other: the producer is an INFERENCE pass over fitted members and the consumer is a TRAINING
term, and `training is fitting only — serving and packaging never import it` is a contract this project
enforces. Spelling the array names twice would satisfy the linter and lose the thing that matters — a
rename would then surface as a `KeyError` against a file that cost thirty-six minutes of card to write.
"""

import os
from collections.abc import Sequence

import numpy as np
from jaxtyping import Float


class TeacherFile:
    """The file format: the two array names, and the single call that writes them under those names."""

    IDENTIFIERS = "identifiers"
    """The study each row belongs to, stored beside the scores so a reader can REORDER rather than trust."""

    SCORES = "scores"
    """One row per study, one column per target, in the corpus's twelve-target order."""

    @staticmethod
    def write(path: os.PathLike[str], identifiers: list[str], scores: Float[np.ndarray, "n t"]) -> None:
        """Write the two arrays under the names the reader looks them up by.

        A call rather than two constants the producer spells into its own `np.savez`: that call names its
        arrays as KEYWORDS, so a constant it never mentions could not keep it honest, and a shared call
        is the only form in which one half cannot drift from the other.
        """
        np.savez(path, identifiers=np.array(identifiers), scores=scores)

    @staticmethod
    def merged(parts: Sequence[os.PathLike[str]], path: os.PathLike[str]) -> int:
        """Join shard files into one, and return how many studies the joined file holds.

        A teacher pass is DECODE-bound — the card sat at 0% while pydicom worked — so the pass is run as
        several processes over disjoint slices of the same split. The join lives here, beside `write`, for
        the reason the module exists at all: the two array names are spelled in exactly one place, and a
        merge that spelled them itself would be the drift this file was written to make impossible.
        """
        loaded = [np.load(part) for part in parts]
        identifiers = np.concatenate([part[TeacherFile.IDENTIFIERS] for part in loaded])
        scores = np.concatenate([part[TeacherFile.SCORES] for part in loaded])
        np.savez(path, identifiers=identifiers, scores=scores)
        return len(identifiers)
