"""The raptor member's READING of our corpus, written as a feature-teacher file.

`python -m kneemri.kaggle.raptor_features --weights <checkpoint>` — pass every study of the train split
through the foreign member and write `raptor_features.npz`, which `train --teacher-features` reads.

WHAT IS IN IT AND WHY IT IS NOT THE OTHER FILE. `raptor_teacher.npz` holds the twelve numbers the
teacher's HEAD emits; this holds the `d`-wide vector that head was applied to, meaned over the finding
axis. The distinction is the whole of branch (b): `swin_distill_fit` distilled the logits and read
-0.0077 against a cold control, and the pre-registered suspect for that was the LEVEL of the target —
twelve calibrated verdicts are a narrow thing to pretrain a trunk on, a representation is not.

THE TEACHER MUST NOT HAVE TRAINED ON THE STUDIES IT TEACHES, and this is the same honesty constraint
`objective.py` states for the logit path. A member fitted with `--fold -1` has read every study including
fold 0, so its features on those are memorised rather than inferred, and a student pulled onto them is
then scored on a holdout that flatters both. The fold-0 `_s2026` twins are the honest teachers.
ENFORCING IT IS THE CALLER'S JOB — `honest` refuses the case that is checkable (one of OUR checkpoints
that records `fold: -1`), and a foreign checkpoint records no recipe at all, so for the raptor members
nothing here can check anything and the constraint travels only as this paragraph.

A STUDY THAT WILL NOT DECODE GETS NaN, not a fallback. `raptor_teacher` writes 0.5 there because a
submission must emit a number for every row; a teacher file is read only where it teaches, and there is
no "average representation" that means anything. `TeacherFile.aligned` and `TeacherFeatures.rows` both
refuse a NaN that reaches a batch, so an undecodable study is a loud failure at load rather than a
direction the student is quietly pulled toward.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from core import cli
from kneemri.device import VisibleCards
from kneemri.evaluation.raptor_feed import RaptorFeed, Records
from kneemri.evaluation.raptor_model import RaptorMember
from kneemri.kaggle.raptor_teacher import TRAIN, TeacherPass
from kneemri.kaggle.submission import Split
from kneemri.teacher_file import TeacherFile

logger = logging.getLogger(__name__)

FEATURES = "raptor_features.npz"
"""Named for what it holds rather than for its teacher, because the array inside it is what a mistaken
invocation confuses: a file called `raptor_teacher` is one `--teacher-` flag away from the wrong one."""

PROGRESS_EVERY = 100
"""How often the pass says where it is. The same cadence `RaptorEntry` logs at, and for its reason: a
pass over 4,406 studies is half an hour, and a silent half hour is indistinguishable from a hung one."""

FULL_FOLD = -1
"""The recorded fold of a member fitted on EVERY study — the one case `honest` can refuse by reading."""


class FeaturePass(TeacherPass):
    """One pass of a member over our corpus, writing what it READ rather than what it decided.

    A subclass because the split tables, the sharding, the records and the fleet load are the same facts
    about the same pass; only what is taken out of each member and the array it is written under differ.
    """

    def embeddings(
        self, identifiers: list[str], records: Records, members: dict[str, RaptorMember], device: str
    ) -> dict[str, Float[np.ndarray, "n d"]]:
        """Every member's reading of every study, from ONE decode of each.

        The decode is what this pass is bound by — the card sits near idle while pydicom works — so the
        loop is over studies on the outside and members on the inside, exactly as `RaptorEntry.readings`
        arranges it for the scoring pass.

        A DECODE FAILURE LEAVES NaN AND CARRIES ON, logged with its traceback. Stopping the pass would
        throw away thirty minutes of decode over one unreadable study, and substituting a value would
        put a direction in the file that no member ever read.
        """
        feed = RaptorFeed(self.split.series(self.competition))
        rows = {name: np.full((len(identifiers), member.width()), np.nan) for name, member in members.items()}
        for index, study in enumerate(identifiers):
            try:
                volume, mask = feed.study(study, records)
            except Exception:
                logger.exception("study %s could not be decoded — it will carry no teacher reading", study)
                continue
            for name, member in members.items():
                try:
                    rows[name][index] = member.represented(volume, mask, device)
                except Exception:
                    logger.exception("member %s could not read study %s", name, study)
            if (index + 1) % PROGRESS_EVERY == 0:
                logger.info("read %d of %d studies", index + 1, len(identifiers))
        return rows

    def wrote(self, path: Path, identifiers: list[str], embeddings: Float[np.ndarray, "n d"]) -> Path:
        """Write one member's representations under the name the feature side looks them up by.

        A SEPARATE METHOD from `written` rather than a flag on it, mirroring `TeacherFile.write_embeddings`
        and for the same reason: the array name is what keeps a targets file from being loaded as a
        features file, and a shared writer parameterised by it is one argument away from writing either.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        TeacherFile(path).write_embeddings(identifiers, embeddings)
        logger.info("wrote %d studies x %d features to %s", *embeddings.shape, path)
        return path

    @staticmethod
    def honest(checkpoint: Path) -> None:
        """Refuse a teacher this project fitted on EVERY study — the checkable half of the constraint.

        Only ours record a recipe, so this catches exactly one mistake: reaching for a `--fold -1` member
        of our own because it is the strongest thing on disk. A foreign checkpoint carries no recipe and
        passes silently, which is not a clean bill of health — see the module docstring. Read rather than
        asserted about, because a checkpoint's own record is the only evidence available at this point.
        """
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        recipe = state.get("recipe") if isinstance(state, dict) else None
        if isinstance(recipe, dict) and int(recipe.get("fold", 0)) == FULL_FOLD:
            message = (
                f"{checkpoint.name} records fold {FULL_FOLD}, so it was fitted on every study including "
                "the holdout a student taught from it would be scored on — its readings there are "
                "memorised, and the student's score would flatter both"
            )
            raise ValueError(message)


def main() -> None:
    """Read the train split through the member and write its representations as a feature-teacher file."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--weights", type=Path, nargs="+", required=True, help="raptor checkpoints")
    command.parser.add_argument("--fleet", type=Path, default=None, help="write one file per checkpoint here")
    command.parser.add_argument("--split", default=TRAIN, help="which split to teach over")
    command.parser.add_argument("--out", type=Path, default=None, help="destination npz")
    command.parser.add_argument("--studies", type=int, default=0, help="cap the pass, for a smoke")
    command.parser.add_argument("--shard", type=int, default=0, help="which stride of the split to read")
    command.parser.add_argument("--shards", type=int, default=1, help="how many processes share the pass")
    command.parser.add_argument("--merge", type=Path, nargs="*", help="join these shard files and exit")
    arguments = command.parsed()
    root = command.rooted(arguments)

    destination = arguments.out or root.processed() / FEATURES
    if arguments.merge:
        joined = TeacherFile(destination).merged(arguments.merge, TeacherFile.EMBEDDINGS)
        logger.info("merged %d studies into %s", joined, destination)
        return

    for checkpoint in arguments.weights:
        FeaturePass.honest(checkpoint)
    reading = FeaturePass(root.raw(), destination, Split(name=str(arguments.split)))
    identifiers = FeaturePass.sharded(reading.identifiers(), arguments.shard, arguments.shards)
    if arguments.studies:
        identifiers = identifiers[: arguments.studies]
    logger.info("reading over %d %s studies", len(identifiers), arguments.split)

    device = str(VisibleCards().device())
    members = FeaturePass.fleet(arguments.weights, device)
    logger.info("loaded %d members onto %s", len(members), device)

    for name, embeddings in reading.embeddings(identifiers, reading.records(), members, device).items():
        reading.wrote(
            arguments.fleet / f"{name}.npz" if arguments.fleet else destination,
            identifiers,
            embeddings.astype(np.float32),
        )


if __name__ == "__main__":
    main()
