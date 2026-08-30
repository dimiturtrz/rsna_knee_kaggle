"""Whether two label tables score the same holdout the same way — the ruler behind every recorded AUC.

A table is not a different RULER because it has a different name. It is one when the truth an AUC is
computed against changes, and only two things do that: a cell crossing `ASSERTS_PRESENCE`, which flips
which class it belongs to, or a cell gaining or losing a value, which changes whether the pair is scored
at all. A re-map that keeps every cell on its own side of the midpoint changes what is TRAINED on and
nothing about what is SCORED.

WHY IT IS WORTH MEASURING RATHER THAN ASSUMING. `Targets.pair` scores a run against its own `--labels`
when `--eval-labels` is unset, so an arm re-mapping its targets moves its own ruler and prints a number
no other run was measured against. `recal_holdannot` printed `0.8964` that way against a control's
`0.8487`, and the +0.0477 is two rulers rather than a result; re-scored on the shared table it is
+0.0009. Reading the difference off the FILENAME would have been wrong in the other direction too —
`merged_labels.csv` is a column union whose rows are byte-identical to ours, and calling it a moved ruler
would have voided a roster member's number for nothing.

MEASURED 2026-08-30 over the 4,406 shared studies, each count one the repo had already written down
elsewhere: `merged_labels.csv` moves 0 cells, `confident.csv` moves 14,445 by presence (the UNK cells it
clears), `report_labels_v3.csv` moves 6,461 by crossing (the mild cells `Supervision` documents).
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core.data.labels.targets import ASSERTS_PRESENCE

STUDY_COLUMN = "StudyInstanceUID"
DERIVED_COLUMNS = ("__conf", "__verdict")
"""A label table's graded columns are every column but the study id and these two per-target companions."""

DEFAULT_LABELS = "report_labels_v2.csv"
"""The table every ruler in this project is written against, by basename — the root varies per machine."""


@dataclass(frozen=True)
class Ruler:
    """One label table, asked what it would score differently from another over their shared studies."""

    table: Path

    def graded(self) -> pd.DataFrame:
        """The table's per-target VALUES, indexed by study — its confidence and verdict columns dropped."""
        frame = pd.read_csv(self.table).set_index(STUDY_COLUMN)
        return frame[[column for column in frame.columns if not column.endswith(DERIVED_COLUMNS)]]

    def moved_cells(self, other: "Ruler") -> int:
        """How many holdout truth cells `other` scores differently from this table, over shared studies."""
        mine, theirs = self.graded(), other.graded()
        shared = mine.index.intersection(theirs.index)
        mine, theirs = mine.loc[shared], theirs.loc[shared]
        crossed = (mine > ASSERTS_PRESENCE) != (theirs > ASSERTS_PRESENCE)
        presence = mine.isna() != theirs.isna()
        return int(crossed.to_numpy().sum() + presence.to_numpy().sum())
