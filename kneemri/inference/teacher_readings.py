"""Write an ensemble's ordering of the TRAINING studies for a student to be taught:
`python -m kneemri.inference.teacher_readings`.

The consumer is `kneemri.training.teacher_targets`, and the whole of what it needs is ORDER — so the
members are combined with `RankAverage`, the rule the submission path actually ships, and the result is
rewritten as each study's POSITION in the corpus rather than as a probability. Two things follow from
that choice, and both are the point rather than a side effect:

**A rank is the only quantity all twelve labels share.** Effusion is positive in ~60% of studies and
Fracture in ~7%, so a probability separation of 0.1 means something different in each column, and the
distilling term weights its pairs by exactly that separation. A normalised rank gap is "what fraction
of the corpus sits between these two studies" in every column alike.

**Calibration is discarded on purpose**, which is the same argument `kneemri.inference.ensemble` makes
about the metric: an AUC is invariant to any monotone rewrite of a column, so a teacher's probabilities
carry no information its ranks do not, and they carry a per-label scale that would silently reweight
the term.

WHICH MEMBERS MAY TEACH is the one thing this command refuses to guess. A member fitted with `--fold -1`
saw every study, the holdout included, so distilling from it leaks the evaluation labels into the
student through this file and the arm reads as a win with nothing behind it. Pass the fold-aware twins.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.data.corpus.folds import ReportFolds
from core.paths import DataRoot
from core.wakelock import Wakelock
from kneemri.device import VisibleCards
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.holdout_slab import HoldoutSlab
from kneemri.inference.members import Member
from kneemri.inference.prediction_cache import PredictionCache
from kneemri.inference.vote import RankAverage
from kneemri.teacher_file import TeacherFile

logger = logging.getLogger(__name__)

STORE = "teacher_readings"
"""Its own cache directory, beside the holdout's rather than inside it: the two hold readings of
DIFFERENT studies, and one key collision would serve a member's holdout array as its training one."""


class TeacherReadings:
    """The inference pass that turns a set of fold-aware members into one ordering of the train split."""

    def __init__(self, root: DataRoot, device: torch.device, *, overlap: bool = False) -> None:
        self.root = root
        self.predictions = HoldoutPredictions(device)
        self.overlap = overlap

    def _split(self, identifiers: list[str], folds: ReportFolds, fold: int) -> NDArray[np.int64]:
        """The positions a member of this fold TRAINED on — the complement of what it was scored on.

        The training side, which is the opposite of every other caller of `holdout` in this repo, and
        deliberately so: these are exactly the studies the student will be fitted on, and exactly the
        studies whose ordering the labels cannot supply on their own.
        """
        training, _ = folds.holdout(folds.assign(str(self.root.raw() / "train.csv"), identifiers), fold)
        return training

    def _read(self, member: Member, positions: NDArray[np.int64]) -> Float[np.ndarray, "studies targets"]:
        """One member's probabilities for those studies, through the member's own reading of itself."""
        corpus, identifiers = member.cache(self.root.processed()).load()
        studies = [identifiers[position] for position in positions]
        window = member.window(corpus.slices, overlap=self.overlap)
        logger.info("%s: reading %d training studies at %d px", member.file.stem, len(studies), member.read_size)
        store = PredictionCache(self.root.processed() / STORE, studies, self.predictions.views)
        return self.predictions.cached(member, HoldoutSlab(corpus, positions), store, window)

    def teach(
        self, members: list[Member], identifiers: list[str], folds: ReportFolds, fold: int
    ) -> tuple[list[str], Float[np.ndarray, "studies targets"]]:
        """The studies a student will be fitted on, and the ensemble's ordering of them.

        One entry point rather than the three steps it is made of, because the steps are only correct
        in this order: the same `positions` that select the rows must select the identifiers written
        beside them, and a caller that re-derived either half could hand the student an ordering of
        the wrong studies — a mistake with no symptom, since both arrays would still be the right shape.
        """
        if not members:
            message = "no checkpoints to teach from — a teacher file with no members orders nothing"
            raise ValueError(message)

        positions = self._split(identifiers, folds, fold)
        with Wakelock():
            per_member = [self._read(member, positions) for member in members]
        return [identifiers[position] for position in positions], self.ordered(per_member)

    @staticmethod
    def ordered(per_member: list[Float[np.ndarray, "studies targets"]]) -> Float[np.ndarray, "studies targets"]:
        """The members' rank-average, rewritten as each study's position in [0, 1] per label.

        Rescaled rather than left in rank units so the array lands in the same space as the labels it
        sits beside — `Objective.pairs` reads a separation off it, and a separation of 900 against a
        label separation of 0.6 is not a different weighting, it is a different objective.
        """
        combined = RankAverage().combine(per_member)
        order = combined.argsort(axis=0).argsort(axis=0).astype(np.float32)
        return order / max(len(combined) - 1, 1)


def main() -> None:
    """Write the teacher file: `--checkpoints <fold-0 twins> --out teacher.npz`."""
    command = cli.Command(TeacherReadings.__doc__, logger)
    command.parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    command.parser.add_argument("--out", type=Path, required=True)
    command.parser.add_argument("--device", default=VisibleCards().device().type)
    command.parser.add_argument("--folds", type=int, default=5)
    command.parser.add_argument("--split-seed", type=int, default=2026)
    command.parser.add_argument("--fold", type=int, default=0, help="the fold these members held out, never -1")
    command.parser.add_argument("--overlap", action="store_true", help="read overlapping windows, as the kit does")
    arguments = command.parsed()

    if arguments.fold < 0:
        message = (
            f"--fold {arguments.fold} names a full-data fit, whose members saw the holdout too — "
            "teaching from one leaks the evaluation labels into the student"
        )
        raise ValueError(message)

    readings = TeacherReadings(command.rooted(arguments), torch.device(arguments.device))
    members = [Member.from_checkpoint(checkpoint) for checkpoint in arguments.checkpoints]
    _, identifiers = members[0].cache(readings.root.processed()).load()
    folds = ReportFolds(arguments.folds, arguments.split_seed)

    taught, scores = readings.teach(members, identifiers, folds, arguments.fold)
    TeacherFile.write(arguments.out, taught, scores)
    logger.info("%s: %d studies taught by %d members", arguments.out, len(scores), len(members))


if __name__ == "__main__":
    main()
