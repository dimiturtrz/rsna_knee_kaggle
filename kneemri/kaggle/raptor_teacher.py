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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from jaxtyping import Float

from core import cli
from core.findings import STUDY
from kneemri.device import VisibleCards
from kneemri.evaluation.raptor_feed import LAYOUTS, RAPTOR, Layout, RaptorFeed, Records
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

    FILE: ClassVar[str] = TEACHER
    ARRAY: ClassVar[str] = TeacherFile.SCORES
    VERB: ClassVar[str] = "score"
    GERUND: ClassVar[str] = "teaching"

    competition: Path
    destination: Path
    split: Split = field(default_factory=lambda: Split(name=TRAIN))
    layout: Layout = RAPTOR

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
        return RaptorEntry(self.competition, working=self.destination.parent, split=self.split, layout=self.layout)

    def written(self, path: Path, identifiers: list[str], scores: Float[np.ndarray, "n t"]) -> Path:
        """Write one member's readings under the two names the training side looks them up by.

        The path is a parameter rather than always `self.destination` because a fleet pass writes one
        file per member out of a single decode, and the alternative — a `TeacherPass` per member — would
        re-read the split's tables thirteen times to name thirteen files.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        TeacherFile(path).write(identifiers, scores)
        logger.info("wrote %d studies x %d findings to %s", *scores.shape, path)
        return path

    @staticmethod
    def fleet(weights: Sequence[Path], device: str, layout: Layout = RAPTOR) -> dict[str, RaptorMember]:
        """Every checkpoint loaded at once, keyed by its filename stem.

        They fit together because a member is a backbone and a forty-line head: thirteen of them are a
        few gigabytes on a card that holds thirty-two, and holding them all is what lets one decode feed
        all of them. The stem is the key because it is what the teacher file is then named for, so the
        distance report reads back the checkpoint rather than an index into an argument list.
        """
        return {path.stem: RaptorMember.loaded(path, device, layout=layout) for path in weights}

    @staticmethod
    def honest(checkpoint: Path) -> None:
        """Admit any teacher: this pass copies an ORDERING, which a memorised reading still carries.

        See the module docstring — the readings are unsound for validating a teacher and sound for copying
        one, so there is nothing about a checkpoint this pass could refuse.
        """
        del checkpoint

    def taught(
        self, identifiers: list[str], records: Records, members: Mapping[str, RaptorMember], device: str
    ) -> Mapping[str, Float[np.ndarray, "n w"]]:
        """What this pass takes out of every member: the head's verdicts, through the submission's scorer."""
        return self.entry().readings(identifiers, records, members, device)

    def kept(self, path: Path, identifiers: list[str], values: Float[np.ndarray, "n w"]) -> Path:
        """Write one member's take under the array name this pass's readers look it up by."""
        return self.written(path, identifiers, values)

    @classmethod
    def main(cls, doc: str | None, log: logging.Logger) -> None:
        """The command both passes are: read the flags, then merge shards, or read a shard and write it.

        One body because the two commands differ in nothing but the facts the class already holds — its
        default file, its array name, its refusal, and what it takes out of a member and writes.
        """
        command = cli.Command(doc, log)
        command.parser.add_argument("--weights", type=Path, nargs="+", required=True, help="raptor checkpoints")
        command.parser.add_argument("--fleet", type=Path, default=None, help="write one file per checkpoint here")
        command.parser.add_argument("--split", default=TRAIN, help="which split to teach over")
        command.parser.add_argument("--out", type=Path, default=None, help="destination npz")
        command.parser.add_argument("--studies", type=int, default=0, help="cap the pass, for a smoke")
        command.parser.add_argument("--shard", type=int, default=0, help=f"which stride of the split to {cls.VERB}")
        command.parser.add_argument("--shards", type=int, default=1, help="how many processes share the pass")
        command.parser.add_argument("--merge", type=Path, nargs="*", help="join these shard files and exit")
        command.parser.add_argument(
            "--layout", choices=sorted(LAYOUTS), default="raptor", help="the decode it was fitted on"
        )
        arguments = command.parsed()
        root = command.rooted(arguments)

        destination = arguments.out or root.processed() / cls.FILE
        if arguments.merge:
            joined = TeacherFile(destination).merged(arguments.merge, cls.ARRAY)
            log.info("merged %d studies into %s", joined, destination)
            return

        for checkpoint in arguments.weights:
            cls.honest(checkpoint)
        layout = LAYOUTS[arguments.layout]
        passing = cls(root.raw(), destination, Split(name=str(arguments.split)), layout)
        identifiers = cls.sharded(passing.identifiers(), arguments.shard, arguments.shards)
        if arguments.studies:
            identifiers = identifiers[: arguments.studies]
        log.info("%s over %d %s studies", cls.GERUND, len(identifiers), arguments.split)

        device = str(VisibleCards().device())
        members = cls.fleet(arguments.weights, device, layout)
        log.info("loaded %d members onto %s", len(members), device)

        for name, values in passing.taught(identifiers, passing.records(), members, device).items():
            passing.kept(
                arguments.fleet / f"{name}.npz" if arguments.fleet else destination,
                identifiers,
                values.astype(np.float32),
            )


def main() -> None:
    """Score the train split through the foreign member and write its readings as a teacher file."""
    TeacherPass.main(__doc__, logger)


if __name__ == "__main__":
    main()
