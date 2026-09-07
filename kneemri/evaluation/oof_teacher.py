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

FILL = "band_dense_s2026.holdout.npz"
"""The default fold-0 block for `--fill`, and it is a DIFFERENT teacher from the one this module writes
by default — see `main`'s `--fill` help and the module docstring's hole paragraph, which still describes
what `--out teacher_oof_corpus.npz` alone produces."""

STUDIES = "studies"
"""How a holdout reduction spells its study column; a teacher file spells the same thing `identifiers`."""

PREDICTIONS = "predictions"
"""And its score column, which a teacher file spells `scores`. The two formats are joined here."""

TRUTH = "truth"
"""The labels the block was scored against. Only the witness reads it; the teacher file carries scores."""

TINY = 1e-9
"""A prevalence floor, so a label no study in a fold carries divides rather than explodes."""

POSITIVE = 0.5
"""Where a probability becomes a called positive, for the witness's rate only. Nothing is thresholded
into the teacher — the file carries the continuous ordering, which is the whole reason it exists."""


@dataclass(frozen=True)
class BlockStats:
    """One block of the teacher, summarised per label, so a filled block can be compared to the folds.

    A teacher file carries an ORDERING, so the question a fill has to answer is not whether the filled
    block is accurate but whether it is the SAME KIND of reading as the blocks beside it. Four fold twins
    give four readings of one recipe, and the spread between them is the scatter a fifth block of the same
    kind would sit inside. A block outside it teaches a fifth of the corpus differently from the rest,
    which is a variable the consumer never asked for and could not separate from its own treatment.
    """

    name: str
    predicted: Float[NDArray[np.float64], " targets"]
    spread: Float[NDArray[np.float64], " targets"]
    positive: Float[NDArray[np.float64], " targets"]
    prevalence: Float[NDArray[np.float64], " targets"]

    @classmethod
    def of(cls, part: Path) -> "BlockStats":
        """Summarise one holdout reduction, per label."""
        stored = np.load(part, allow_pickle=True)
        scores = stored[PREDICTIONS].astype(np.float64)
        truth = stored[TRUTH].astype(np.float64)
        return cls(
            part.stem.removesuffix(".holdout"),
            scores.mean(axis=0),
            scores.std(axis=0),
            (scores > POSITIVE).mean(axis=0),
            truth.mean(axis=0),
        )

    def axes(self) -> dict[str, Float[NDArray[np.float64], " targets"]]:
        """The three ways one block can read unlike another, and only the last two are about the MODEL.

        `mean` is the raw level, which a fold with more positives in it raises without anything about the
        reader changing — so it is reported and never trusted alone. `calibrated` divides that level by
        the fold's own prevalence and asks whether the reader is differently confident for what is
        actually there. `spread` asks whether it separates harder, and prevalence cannot move it.
        """
        return {
            "mean": self.predicted,
            "calibrated": self.predicted / np.maximum(self.prevalence, TINY),
            "spread": self.spread,
        }

    def outside(self, scatter: tuple["BlockStats", ...]) -> dict[str, list[int]]:
        """Per axis, the labels where this block falls outside the range the given blocks span."""
        spanned = [block.axes() for block in scatter]
        outside: dict[str, list[int]] = {}
        for axis, values in self.axes().items():
            others = np.stack([block[axis] for block in spanned])
            beyond = (values < others.min(axis=0)) | (values > others.max(axis=0))
            outside[axis] = [int(label) for label in np.flatnonzero(beyond)]
        return outside

    @staticmethod
    def _witness(folds: tuple[Path, ...], fill: Path) -> list[str]:
        """State whether the filled block reads like the folds beside it, per label, before it is written."""
        scatter = tuple(BlockStats.of(part) for part in folds)
        filled = BlockStats.of(fill)
        labels = filled.predicted.size
        verdicts = [
            f"  {axis:11s} outside the fold-to-fold scatter on {len(beyond)}/{labels} labels {beyond}"
            for axis, beyond in filled.outside(scatter).items()
        ]
        return [*[block.line() for block in scatter], filled.line(), *verdicts]

    def line(self) -> str:
        """One row of the witness: the block's own averages over its labels."""
        return (
            f"  {self.name:24s} mean {self.predicted.mean():.4f}  spread {self.spread.mean():.4f}  "
            f"called+ {self.positive.mean():.4f}  true+ {self.prevalence.mean():.4f}"
        )


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
    command.parser.add_argument(
        "--twins",
        default=FOLDS,
        help=(
            "glob for the fold twins to assemble. The default set was fitted WITHOUT --exclude-fold, so "
            "every one of them trained on fold 0 and a student distilled from them and scored on fold 0 "
            "reads a number inflated by function mimicry. A clean set is globbed by its own name."
        ),
    )
    command.parser.add_argument("--out", type=Path, required=True, help="destination teacher npz")
    command.parser.add_argument(
        "--fill",
        nargs="?",
        const=FILL,
        default=None,
        help=(
            "fold-0 block to append, for a teacher a --fold -1 arm can consume; the default block is "
            f"{FILL}. A run SCORED on fold 0 must never read the result — write it to its own filename."
        ),
    )
    arguments = command.parsed()
    root = command.rooted(arguments)
    models = arguments.models or root.models()
    folds = tuple(sorted(models.glob(arguments.twins)))
    if not folds:
        message = f"no out-of-fold twins under {models} matching {arguments.twins}"
        raise FileNotFoundError(message)
    parts = folds
    if arguments.fill is not None:
        fill = models / arguments.fill
        for line in BlockStats._witness(folds, fill):  # noqa: SLF001 - its own class, called from this module's main
            logger.info(line)
        parts = (*folds, fill)
    for line in OutOfFoldTeacher(parts).write(arguments.out):
        logger.info(line)


if __name__ == "__main__":
    main()
