"""Stamp a `corpus` column onto a labels table without touching its labels:
`python -m core.data.corpus.corpus_column --labels <table> --tags <tagged table> --out <table>`.

WHY THIS EXISTS. `--corpus-oversample` refuses to run without a `corpus` column in the labels table
(`kneemri/training/run_setup.py:214`), and `merged3_labels.csv` — the table `merged3_corpus.sh` trains
on, and therefore the only table whose control score is on disk — does not carry one.

WHY NOT THE TAGGED TABLE ITSELF. `merged3_tagged_labels.csv` carries the column AND RECALIBRATED
FINDING VALUES: ACL 0.2995 -> 0.2233, MCL 0.2141 -> 0.1311, Effusion 0.5009 -> 0.3387, Synovitis
0.3394 -> 0.1915, Fracture 0.2423 -> 0.1332. Training on it would confound the draw policy with label
calibration — the one axis this project has measured positive at BOTH the default and the dense corner
(+0.0129 / +0.0161) — and the delta would price neither.

WHY A JOIN AND NEVER A COLUMN ASSIGNMENT. Both tables hold the same studies, in different row order. An
assignment by position mislabels the corpus of most rows, and every value it writes still looks like a
corpus name, so nothing downstream can notice. The merge is on `StudyInstanceUID` and refuses anything
short of one-to-one coverage.
"""

import logging
from pathlib import Path

import pandas as pd

from core import cli
from core.data.corpus.merged_cache import CORPUS
from core.data.corpus.study_metadata import IDENTIFIER

logger = logging.getLogger(__name__)


class CorpusTags:
    """The corpus tag table, and the one join that may stamp it onto a table of labels."""

    @classmethod
    def tagged(cls, labels: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
        """`labels` with the `corpus` column of `tags` joined on the study identifier, values untouched.

        Refuses a `labels` frame that already carries the column, a `tags` frame that does not, and any
        identifier the tag table cannot answer — a partial join would silently drop rows from the corpus
        the oversample is meant to weight.
        """
        if CORPUS in labels.columns:
            message = f"the labels table already carries a `{CORPUS}` column; nothing to join"
            raise ValueError(message)
        if CORPUS not in tags.columns:
            message = f"the tag table carries no `{CORPUS}` column: {sorted(tags.columns)[:8]}"
            raise ValueError(message)

        joined = labels.merge(tags[[IDENTIFIER, CORPUS]], on=IDENTIFIER, how="left", validate="one_to_one")
        missing = joined[CORPUS].isna().sum()
        if missing:
            message = f"{missing} of {len(joined)} studies have no tag in the tag table; the join is not one-to-one"
            raise ValueError(message)
        return joined


def main() -> None:
    """Write the joined table and report the row count each corpus contributed."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, required=True, help="table to stamp, values untouched")
    command.parser.add_argument("--tags", type=Path, required=True, help="table carrying the corpus column")
    command.parser.add_argument("--out", type=Path, required=True, help="where the joined table is written")
    arguments = command.parsed()

    joined = CorpusTags.tagged(pd.read_csv(arguments.labels), pd.read_csv(arguments.tags))
    joined.to_csv(arguments.out, index=False)
    counts = joined[CORPUS].value_counts()
    tally = [f"  {name} {count}" for name, count in counts.items()]
    command.report([f"{arguments.out}: {len(joined)} studies", *tally])


if __name__ == "__main__":
    main()
