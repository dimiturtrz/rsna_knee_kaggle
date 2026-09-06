"""OUR OWN pool's out-of-fold reading of the training studies, assembled into a teacher file.

`python -m kneemri.evaluation.oof_teacher --out <processed>/teacher_oof_corpus.npz`

WHY THIS IS NOT SELF-DISTILLATION. `unread_teacher` argues against teaching a student with our own
member, and it is right about the case it names: a member scoring studies it was itself fitted on hands
back memorised readings, so the arm measures training dynamics while wearing a label experiment's
clothes. The `oof_fold*` twins are the exception the argument does not cover. Each was fitted with one
fold held out and scored exactly that fold, so every reading in this file comes from a model that did
NOT train on the study it is reading. The union is the whole training set, read out of fold.

WHAT IT COVERS, AND THE HOLE IS THE POINT. Four files, 875 + 884 + 849 + 890 = 3498 studies — folds one
through four, which is precisely the training half. Fold 0 is absent because no `oof_fold0` exists and
none should: fold 0 is the holdout every arm is scored on, and a teacher that had read it would leak the
evaluation set's labels into the student through this file. `teacher_targets` states that constraint and
this file satisfies it by construction rather than by care. The corpus is 4407; the 909 not here are the
holdout, and a teacher covering all 4407 would be the broken one.

WHY IT IS WORTH A CARD HOUR WHEN THREE DISTILLATION ARMS HAVE READ FLAT. The three that ran taught from
`teacher_foreign_corpus` — a foreign pool, whose ordering the seat curve showed sits INSIDE our own
pool's consensus, so each increment of teacher weight pulled the student toward the pool and spent the
diversity a seat needs. This teacher is our pool's own out-of-fold posterior, which is a different
quantity from either: continuous where our labels are five discrete levels, and honest where our members'
in-fold readings are memorised. The mechanism it addresses is the one the ledger prices at ~20% of read
cells wrong and 38.5% of cell pairs tied — a continuous ordering breaks ties that a five-level table
cannot, on every cell rather than on the fraction a relabel run reaches.

IT IS ASSEMBLED, NOT COMPUTED. Every reading already exists on disk from the runs that produced the
folds, so this costs no card at all — which is why it is worth running before the arm that consumes it
rather than after.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from kneemri.teacher_file import TeacherFile

logger = logging.getLogger(__name__)

FOLDS = "oof_fold[1-4]_s2026.holdout.npz"
"""The out-of-fold twins, by the glob that finds them. Fold 0 is deliberately not matched."""

STUDIES = "studies"
"""How a holdout reduction spells its study column; a teacher file spells the same thing `identifiers`."""

PREDICTIONS = "predictions"
"""And its score column, which a teacher file spells `scores`. The two formats are joined here."""


@dataclass(frozen=True)
class OutOfFoldTeacher:
    """The four out-of-fold readings, concatenated into one table over the training studies."""

    parts: tuple[Path, ...]

    def readings(self) -> tuple[list[str], Float[NDArray[np.float64], "studies targets"]]:
        """Every study each part read, and the twelve scores it gave it, in the order the parts are listed.

        A study appearing twice would mean two folds both claimed it, which is a fold-assignment defect
        rather than a duplicate to average away, so it is refused rather than reduced.
        """
        names: list[str] = []
        scores: list[NDArray[np.float64]] = []
        for part in self.parts:
            stored = np.load(part, allow_pickle=True)
            names.extend(str(name) for name in stored[STUDIES])
            scores.append(stored[PREDICTIONS].astype(np.float64))
        repeated = len(names) - len(set(names))
        if repeated:
            message = f"{repeated} studies are read by more than one fold, so the folds are not disjoint"
            raise ValueError(message)
        return names, np.concatenate(scores, axis=0)

    def write(self, out: Path) -> list[str]:
        """Write the teacher file and report what went into it, per part."""
        names, scores = self.readings()
        TeacherFile.write(out, names, scores)
        read = [len(np.load(part, allow_pickle=True)[STUDIES]) for part in self.parts]
        return [
            *[f"  {part.stem:28s} {count} studies" for part, count in zip(self.parts, read, strict=True)],
            f"{len(names)} training studies, {scores.shape[1]} targets, out of fold by construction -> {out}",
        ]


def main() -> None:  # devtools-ignore: test-mirror
    """Assemble the out-of-fold teacher table from the fold twins already on disk."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--models", type=Path, default=None, help="where the fold twins live")
    command.parser.add_argument("--out", type=Path, required=True, help="destination teacher npz")
    arguments = command.parsed()
    root = command.rooted(arguments)
    models = arguments.models or root.models()
    parts = tuple(sorted(models.glob(FOLDS)))
    if not parts:
        message = f"no out-of-fold twins under {models} matching {FOLDS}"
        raise FileNotFoundError(message)
    for line in OutOfFoldTeacher(parts).write(arguments.out):
        logger.info(line)


if __name__ == "__main__":
    main()
