"""Which of our studies a foreign fold checkpoint has already seen. `python -m kneemri.evaluation.foreign_split`.

A PUBLISHED CHECKPOINT IS NOT PRICEABLE UNTIL ITS SPLIT IS. Scoring someone else's model on studies it
trained on measures memory, not skill, and it reads HIGH — which is the direction that gets a member
shipped. This project has already voided one board reading that way: `raptor_solo`'s 0.8938 came off a
holdout it was 98.6% in-sample on, and that single cell was carrying a whole claimed effect.

MATTIAANGELI'S SPLIT IS PUBLIC, AND IT IS AN MD5. Their training program does not use `KFold`,
`GroupKFold` or `train_test_split` — searching for that vocabulary finds nothing and reads like an
absence of a split. It is a hash of the REPORT TEXT (`bend_the_knee_to_dinov3_ensembled.py:1508`):

    grp = md5(report_text or study_uid)[:8] as int, mod 5      fold f trains on grp != f, holds out grp == f

Which is deterministic, groups studies that share a report, and is reproducible from `train.csv` alone.
So fold f's checkpoint is honestly priceable on EXACTLY the studies with `grp == f`, and across the five
folds that covers every study with a member that never saw it. Their family goes from board-or-nothing to
measurable — but only through this file, and only one fold at a time.

NEVER QUOTE A FOREIGN NUMBER OVER A FULL HOLDOUT. Over our 909-study holdout their fold 0 is in-sample on
700 of them. A "foreign solo AUC" computed on the 909 is not a weaker version of the right number; it is
the `raptor_solo` defect with a different name.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Bool, Int
from numpy.typing import NDArray

from core import cli
from core.findings import STUDY

logger = logging.getLogger(__name__)

REPORT = "Report"
FOLDS = 5
"""Their `% 5`. Not a knob — changing it stops describing their checkpoints."""
DIGEST_CHARS = 8
"""Their `hexdigest()[:8]`, read as hex. Truncating differently produces a different, wrong partition."""


@dataclass(frozen=True)
class ForeignSplit:
    """The fold each study falls in under a foreign kernel's own partition."""

    fold: Int[NDArray[np.int64], " studies"]
    studies: list[str]

    @staticmethod
    def _group(text: str) -> int:
        """Their hash, transcribed once: md5 of the report text, first eight hex digits, mod five."""
        return int(hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:DIGEST_CHARS], 16) % FOLDS

    @classmethod
    def of(cls, studies: list[str], labels: pd.DataFrame) -> "ForeignSplit":
        """Assign each study its fold, falling back to the study UID exactly where their `rep.get(s, s)` does.

        A study ABSENT from the label table hashes its own UID, so the fold is still defined and no row is
        dropped silently. A study PRESENT with an empty report hashes the empty string — their `.fillna('')`
        runs before their `.get`, so absent and blank are different cases, and collapsing them would move
        every blank-report study to a different fold than the one their checkpoint actually held out.
        """
        table = labels.drop_duplicates(subset=STUDY).set_index(STUDY)
        reports = table[REPORT].fillna("") if REPORT in table.columns else pd.Series(dtype=str)
        text = [reports.get(study, study) for study in studies]
        return cls(fold=np.array([cls._group(one) for one in text], dtype=np.int64), studies=list(studies))

    def out_of_sample(self, fold: int) -> Bool[NDArray[np.bool_], " studies"]:
        """The studies fold `fold`'s checkpoint held OUT — the only ones it may be scored on."""
        return self.fold == fold

    def sizes(self) -> dict[int, int]:
        """How many of these studies fall in each fold; an empty fold means the wrong table was joined."""
        return {index: int((self.fold == index).sum()) for index in range(FOLDS)}

    def lines(self, fold: int) -> list[str]:
        """The contamination statement a foreign reading has to carry, both denominators spelled out."""
        clean = int(self.out_of_sample(fold).sum())
        total = len(self.studies)
        return [
            f"fold {fold}: out-of-sample on {clean} of {total} studies ({clean / total:.1%}), "
            f"IN-SAMPLE on {total - clean} ({(total - clean) / total:.1%})",
            f"fold sizes {self.sizes()}",
            f"a solo AUC over all {total} would be the raptor_solo defect; score fold {fold} on its {clean}",
        ]


def main() -> None:
    """State how much of a holdout a foreign fold has already seen, before anything is scored on it."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, required=True)
    command.parser.add_argument("--studies", type=Path, required=True, help="a .npz or .csv carrying StudyInstanceUID")
    command.parser.add_argument("--fold", type=int, default=0)
    arguments = command.parsed()

    if arguments.studies.suffix == ".npz":
        studies = [str(one) for one in np.load(arguments.studies, allow_pickle=True)["studies"]]
    else:
        studies = pd.read_csv(arguments.studies)[STUDY].astype(str).tolist()
    command.report(ForeignSplit.of(studies, pd.read_csv(arguments.labels)).lines(arguments.fold))


if __name__ == "__main__":
    main()
