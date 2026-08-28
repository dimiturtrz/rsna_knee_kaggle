"""The raptor member's ordering over OUR corpus, written as a teacher file.

`python -m kneemri.kaggle.raptor_teacher --weights <checkpoint>` — score every study of the train
split through the foreign member and write `raptor_teacher.npz`, which `train --teacher-targets` reads.

WHY THIS TEACHER AND NOT THE ONE WE ALREADY HAVE. `distill_pairs` distilled our own pool into a student
of the same family and read +0.0016 with a seat value of −0.0003, and the ledger names the cause rather
than guessing at it: at 0.9159 nearest, the student sat closer to a seated member than any seated member
sits to its own nearest neighbour. A rank average cannot pay for an ordering it already holds. That is a
statement about that teacher, not about distillation.

WHAT MAKES THIS ONE DIFFERENT IS MEASURED, NOT ASSERTED. `raptor_v4_swa` served alone scored 0.923 on the
board, exactly level with our own best single on the same day — another architecture, another decode,
another training corpus, and equal quality by the only ruler that decides anything here. It is the first
teacher this project has held whose ordering the pool provably does not already contain.

IT IS ALSO THE ROUTE THE CHAMPION PIN NAMES. At the measured seat pass-through the member-quality lane
ceilings near 0.943, so the remaining routes are external data and stacking. A student taught by a foreign
model is the external-data route expressed as a member we can SEAT — which the foreign member itself
cannot be cheaply, since its decode costs a second forward path inside the kernel's budget.

WHAT THE READINGS ARE AND ARE NOT. The teacher was fitted on all 4,406 studies, so its scores here are
partly memorised. That is sound for copying an ORDERING and unsound for validating one: no number out of
this file is evidence about the teacher's accuracy, and the student's holdout stays clean because a
fold-0 target is never shown to it. It does mean the soft targets are sharper on train than an honest
model's would be, which is the mechanism by which too large a `--distill-weight` could hurt.

EVERY STUDY GETS A ROW. A study that will not decode contributes `FALLBACK_PROBABILITY` and a logged
traceback, exactly as it does on a submission — the asymmetry is the same one `RaptorEntry` names, and a
teacher file with a hole in it would be a silent training bug rather than a loud failure.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float

from core import cli
from core.findings import STUDY
from kneemri.device import VisibleCards
from kneemri.evaluation.raptor_feed import RaptorFeed, Records
from kneemri.evaluation.raptor_model import RaptorMember
from kneemri.kaggle.raptor_run import RaptorEntry
from kneemri.kaggle.submission import Split
from kneemri.teacher_file import TeacherFile

logger = logging.getLogger(__name__)

TRAIN = "train"
"""The split whose labels we hold, and the only one a teacher file is written over."""

TEACHER = "raptor_teacher.npz"
"""Named for its teacher rather than its fold: it covers every train study, and which of them a given
student withholds is the student's business. `teacher_fold0.npz` beside it is the self-distilled one."""


@dataclass(frozen=True)
class TeacherPass:
    """One pass of a foreign member over our corpus: where the studies are, and where the readings go."""

    competition: Path
    destination: Path
    split: Split = field(default_factory=lambda: Split(name=TRAIN))

    def identifiers(self) -> list[str]:
        """Every study in the split's own table order, as strings.

        The order is carried into the file rather than relied upon: `TeacherFile` stores the identifiers
        beside the scores precisely so a reader reorders instead of trusting, and a training run whose
        study order differs from this one is the normal case, not the exception.
        """
        table = pd.read_csv(self.competition / f"{self.split.name}.csv")
        return [str(study) for study in table[STUDY].tolist()]

    @staticmethod
    def sharded(identifiers: list[str], shard: int, shards: int) -> list[str]:
        """One process's slice of the split, strided rather than blocked.

        Strided because decode cost varies with a study's slice count and the corpus is not shuffled: a
        blocked split finishes its shards minutes apart, and every process that finishes early leaves a
        core idle while the slowest one still holds the whole pass open.
        """
        return identifiers[shard::shards]

    def records(self) -> Records:
        """The split's series rows, keyed the way the raptor decode wants them."""
        return RaptorFeed.records(self.competition / f"{self.split.name}_series.csv")

    def entry(self) -> RaptorEntry:
        """The scorer, pointed at the split whose labels we hold instead of the competition's test set.

        `Split(name="train")` exists for exactly this — the same path over studies we can score — so the
        foreign decode, its window count and its per-study fallback are the ones a submission uses, with
        no second reading of any of them written here.
        """
        return RaptorEntry(self.competition, working=self.destination.parent, split=self.split)

    def written(self, identifiers: list[str], scores: Float[np.ndarray, "n t"]) -> Path:
        """Write the readings under the two names the training side looks them up by."""
        TeacherFile.write(self.destination, identifiers, scores)
        logger.info("wrote %d studies x %d findings to %s", *scores.shape, self.destination)
        return self.destination


def main() -> None:
    """Score the train split through the foreign member and write its readings as a teacher file."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--weights", type=Path, required=True, help="the raptor checkpoint")
    command.parser.add_argument("--split", default=TRAIN, help="which split to teach over")
    command.parser.add_argument("--out", type=Path, default=None, help="destination npz")
    command.parser.add_argument("--studies", type=int, default=0, help="cap the pass, for a smoke")
    command.parser.add_argument("--shard", type=int, default=0, help="which stride of the split to score")
    command.parser.add_argument("--shards", type=int, default=1, help="how many processes share the pass")
    command.parser.add_argument("--merge", type=Path, nargs="*", help="join these shard files and exit")
    arguments = command.parsed()
    root = command.rooted(arguments)

    destination = arguments.out or root.processed() / TEACHER
    if arguments.merge:
        logger.info("merged %d studies into %s", TeacherFile.merged(arguments.merge, destination), destination)
        return

    teaching = TeacherPass(root.raw(), destination, Split(name=str(arguments.split)))
    identifiers = TeacherPass.sharded(teaching.identifiers(), arguments.shard, arguments.shards)
    if arguments.studies:
        identifiers = identifiers[: arguments.studies]
    logger.info("teaching over %d %s studies", len(identifiers), arguments.split)

    device = str(VisibleCards().device())
    member = RaptorMember.loaded(arguments.weights, device)
    scores = teaching.entry().predictions(identifiers, teaching.records(), member, device)
    teaching.written(identifiers, scores.astype(np.float32))


if __name__ == "__main__":
    main()
