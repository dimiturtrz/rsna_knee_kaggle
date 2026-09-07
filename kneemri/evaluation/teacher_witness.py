"""Whether a distilled student tracks its teachers' IN-SAMPLE outputs on the fold it is scored on.

`python -m kneemri.evaluation.teacher_witness`.

WHAT IT WITNESSES. `oof_fold_k` (k=1..4) was fitted at `--fold k`, which trains on every OTHER fold —
fold 0 included. A student distilled from their targets on folds 1-4 and then SCORED on fold 0 has
learned to mimic four functions that each memorised fold 0's labels. `seat_clean_distill` measures what
REMOVING that costs; this measures whether the path exists at all, and it does so without spending a
training run: the two students are already on disk, and only the teachers' fold-0 outputs are missing.

RESIDUALS, NOT PROBABILITIES, ARE THE READING. A leaky teacher's fold-0 output is close to the truth,
so a good student agrees with it for the innocent reason that both are good. Correlating `p - y`
instead removes exactly that: what is left is the part of each reading the truth does not explain, and
two functions whose MISTAKES line up on studies neither was scored on are two functions related by
something other than the label. The control (`band_dense`, distilled from nothing) fixes the scale that
"related" is read against — it is the same architecture on the same fold with no teacher in its loss.

THE CLEAN TEACHERS ARE THE SECOND CONTROL, on the other axis. They never saw fold 0, so a student that
tracks the LEAKY set's residuals and not the CLEAN set's is tracking memorised labels rather than the
shared inductive bias of the family. Both comparisons are needed: the clean column alone would not
separate "distillation transfers a function" from "distillation transfers fold 0".

IT IS MINUTES OF CARD, and it explains the number the next arm measures rather than producing one of
its own. No score here licenses anything; it is an instrument.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.paths import DataRoot
from core.wakelock import Wakelock
from kneemri.device import VisibleCards
from kneemri.evaluation.foreign_price import ForeignPrice
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.member_readings import MemberReadings
from kneemri.inference.members import Member
from kneemri.inference.prediction_cache import PredictionCache

logger = logging.getLogger(__name__)

Reading = Float[NDArray[np.float64], "studies labels"]

CACHE = "witness_predictions"
"""Its own prediction store: these arrays are a fold the members were never scored on."""


@dataclass(frozen=True)
class TeacherWitness:
    """The arithmetic half — pure functions of arrays, so the table can be tested without a card."""

    truth: Reading

    @staticmethod
    def _pearson(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
        """One column pair's correlation, with a constant column reading 0 rather than nan."""
        if left.std() == 0.0 or right.std() == 0.0:
            return 0.0
        return float(np.corrcoef(left, right)[0, 1])

    def agreement(self, left: Reading, right: Reading) -> float:
        """Mean per-label correlation of two readings of the SAME studies, in the order given."""
        return float(np.mean([self._pearson(left[:, label], right[:, label]) for label in range(self.truth.shape[1])]))

    def mimicry(self, left: Reading, right: Reading) -> float:
        """The same correlation over what the truth does not explain — the reading this module exists for."""
        return self.agreement(left - self.truth, right - self.truth)

    def lines(self, readings: dict[str, Reading], teachers: dict[str, Reading]) -> list[str]:
        """One row per (reading, teacher set): raw agreement, residual mimicry, and the reading's own AUC."""
        rows = [
            f"{name:16s} macro AUC {ForeignPrice.macro(reading.astype(np.float32), self.truth.astype(np.float32)):.4f}"
            for name, reading in readings.items()
        ]
        pairs = [
            f"{name:16s} vs {group:12s} agreement {self.agreement(reading, mean):+.4f}"
            f"  mimicry {self.mimicry(reading, mean):+.4f}"
            for name, reading in readings.items()
            for group, mean in teachers.items()
        ]
        return [*rows, *pairs]


@dataclass(frozen=True)
class FoldReadings:
    """The card half — each checkpoint's probabilities over a fixed study list that is not its holdout."""

    root: DataRoot
    device: torch.device

    @staticmethod
    def aligned(record: Path, names: list[str]) -> tuple[Reading, Reading]:
        """A run's own holdout record, its rows put in the order the card read them, joined by NAME."""
        saved = np.load(record, allow_pickle=True)
        position = {str(name): index for index, name in enumerate(saved["studies"])}
        missing = [name for name in names if name not in position]
        if missing:
            message = f"{record.name} does not hold {len(missing)} of the studies read, the first {missing[0]}"
            raise ValueError(message)
        order = np.array([position[name] for name in names], dtype=np.int64)
        return saved["predictions"].astype(np.float64)[order], saved["truth"].astype(np.float64)[order]

    @staticmethod
    def mean(read: list[tuple[Path, Reading]], group: tuple[Path, ...]) -> Reading:
        """The mean output of one teacher set, taken over the members that set actually named."""
        wanted = {path.name for path in group}
        chosen = [array for path, array in read if path.name in wanted]
        if len(chosen) != len(wanted):
            message = f"{len(chosen)} of {len(wanted)} teachers were read — a member was skipped, not a mean"
            raise ValueError(message)
        return np.mean(chosen, axis=0)

    def over(  # devtools-ignore: test-mirror
        self, checkpoints: list[Path], studies: list[str]
    ) -> tuple[list[str], list[tuple[Path, Reading]]]:
        """The studies actually read, in corpus order, and every checkpoint's probabilities over them.

        The names are returned WITH the arrays because a caller holding its own study order would
        otherwise join by index onto a list this method built by scanning the corpus — the defect
        `AnnotatedReadings.readings` documents, in the one shape where the lengths always match.

        Read WITHOUT overlap, which is how each run's own holdout record was written: the reading being
        correlated has to be the same function as the one on disk, or the residuals differ by the window
        as well as by the teacher.
        """
        predictions = HoldoutPredictions(self.device)
        grouped = MemberReadings([Member.from_checkpoint(path) for path in checkpoints], self.root.processed())
        corpus, reference = grouped.reference()
        wanted = set(studies)
        rows = np.array([index for index, name in enumerate(reference) if name in wanted], dtype=np.int64)
        if rows.size == 0:
            message = "none of the requested studies are in the members' corpus"
            raise ValueError(message)
        names = [reference[index] for index in rows]
        cache = PredictionCache(self.root.processed() / CACHE, names, predictions.views)
        read: list[tuple[Path, Reading]] = []
        with Wakelock():
            for member, slab, held in grouped.slabs(rows, reference, corpus):
                window = member.window(held.slices, overlap=False)
                read.append((member.file, predictions.cached(member, slab, cache, window)))
                logger.info("%s: read over %d studies of a fold it was not scored on", member.file.name, len(names))
        return names, read


def main() -> None:  # devtools-ignore: test-mirror
    """Read both students' fold and both teacher sets over it, and print the witness table."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--student", type=Path, required=True, help="the distilled run's holdout record")
    command.parser.add_argument("--control", type=Path, required=True, help="an undistilled run on the same fold")
    command.parser.add_argument("--leaky", default="oof_fold[1-4]_s2026.pt", help="glob for the teachers that saw it")
    command.parser.add_argument("--clean", default="oof_clean_fold[1-4]_s2026.pt", help="glob for those that did not")
    command.parser.add_argument("--models", type=Path, default=None, help="where the teacher checkpoints live")
    command.parser.add_argument("--device", default=VisibleCards().device().type, choices=("cuda", "cpu"))
    arguments = command.parsed()
    root = command.rooted(arguments)
    models = arguments.models or root.models()
    globs = (("leaky", arguments.leaky), ("clean", arguments.clean))
    groups = {name: tuple(sorted(models.glob(glob))) for name, glob in globs}
    for name, group in groups.items():
        if not group:
            message = f"no {name} teachers under {models}"
            raise FileNotFoundError(message)
    student = np.load(arguments.student, allow_pickle=True)
    names, read = FoldReadings(root, torch.device(arguments.device)).over(
        [path for group in groups.values() for path in group], [str(one) for one in student["studies"]]
    )
    student_path, control_path = arguments.student, arguments.control
    distilled, truth = FoldReadings.aligned(student_path, names)
    control, _ = FoldReadings.aligned(control_path, names)
    witness = TeacherWitness(truth)
    readings = {student_path.name.split(".")[0]: distilled, control_path.name.split(".")[0]: control}
    means = {name: FoldReadings.mean(read, group) for name, group in groups.items()}
    command.report(witness.lines(readings, means))


if __name__ == "__main__":
    main()
