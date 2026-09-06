"""Which training labels the reader AND the model both contradict. `python -m kneemri.evaluation.label_contradiction`.

THE LEVER WAS NEVER "TRUST THE READER". A second reader that disagrees with the label is one opinion
against another, and `reader_agreement` has already priced that opinion: over the whole 696-cell
radiologist set v3 is wrong on 121 cells, so a correction driven by the reader alone would carry its own
error rate straight into the training labels. What is different here is the CONJUNCTION — a cell where
the report reader and a model that never saw that study BOTH contradict the given label. Two
independent instruments agreeing on a specific cell is a much stronger statement than either alone, and
it is cheap: both readings already exist on disk.

THE DENOMINATOR IS STATED TWICE ON PURPOSE. Every count below is reported over ALL cells (studies x
twelve findings) and over the POSITIVE cells, because a correction that touches 2% of all cells and 20%
of the positives is a different proposition from one that touches 2% of both. Three readings in one
session were mis-sized by quoting a rate computed inside the subset that happened to parse.

THE PREDICTION SIDE MUST BE OUT OF SAMPLE. It is read from `.holdout.npz` records, which hold a run's
predictions for the studies it held OUT. A model's opinion about a study it trained on is not an
independent instrument, it is a memory of the label being questioned.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float
from numpy.typing import NDArray

from core import cli
from core.findings import STUDY, TARGETS
from kneemri.runs.holdout_record import HoldoutRecord

logger = logging.getLogger(__name__)

SILENT = "UNK"
POSITIVE = "YES"
VERDICT_SUFFIX = "__verdict"
MIDPOINT = 0.5
"""Where a graded label or a predicted probability becomes a positive, matching `LabelScores`."""


@dataclass(frozen=True)
class LabelContradiction:
    """Per cell: what the reader said, what an out-of-sample model predicted, what the label claims."""

    spoke: Bool[NDArray[np.bool_], "studies labels"]
    read_positive: Bool[NDArray[np.bool_], "studies labels"]
    predicted: Float[NDArray[np.float64], "studies labels"]
    truth: Float[NDArray[np.float32], "studies labels"]

    @staticmethod
    def read_table(path: Path) -> pd.DataFrame:
        """The reader's table, indexed by study — the join key every caller uses."""
        return pd.read_csv(path).drop_duplicates(subset=STUDY).set_index(STUDY)

    @classmethod
    def align(cls, table: pd.DataFrame, records: list[Path]) -> "LabelContradiction":
        """Pair the reader's verdicts with the mean out-of-sample prediction over the shared studies.

        The predictions are averaged across records BEFORE the comparison, which is the same LogitAverage
        the pool is scored under — one member's opinion is noisier than the pool's, and the point of the
        conjunction is that the second instrument be a strong one.
        """
        readings = [reading for record in records if (reading := HoldoutRecord(record).read()) is not None]
        if not readings:
            message = f"none of the {len(records)} named records carries a holdout array"
            raise ValueError(message)
        shared = sorted(set.intersection(*[set(reading.studies) for reading in readings]) & set(table.index))
        stacked = []
        for reading in readings:
            order = {study: position for position, study in enumerate(reading.studies)}
            stacked.append(reading.predictions[[order[study] for study in shared]])
        first = readings[0]
        order = {study: position for position, study in enumerate(first.studies)}
        truth = first.truth[[order[study] for study in shared]]
        rows = table.loc[shared]
        verdicts = np.column_stack([rows[f"{target}{VERDICT_SUFFIX}"].to_numpy(str) for target in TARGETS])
        return cls(
            spoke=verdicts != SILENT,
            read_positive=verdicts == POSITIVE,
            predicted=np.mean(stacked, axis=0),
            truth=truth,
        )

    def labelled_positive(self) -> Bool[NDArray[np.bool_], "studies labels"]:
        """The cells the TRAINING label calls positive — the denominator a correction is priced against."""
        return self.truth.astype(np.float64) >= MIDPOINT

    def candidates(self) -> Bool[NDArray[np.bool_], "studies labels"]:
        """Cells where the reader SPOKE, and reader and model contradict the label on the same side.

        Both instruments must land on the same side of the label, not merely differ from it: a reader
        calling positive and a model calling negative is two disagreements that cancel, and correcting
        the cell to either one of them is a coin toss dressed as evidence.
        """
        model_positive = self.predicted >= MIDPOINT
        agree = self.read_positive == model_positive
        return self.spoke & agree & (self.read_positive != self.labelled_positive())

    def per_finding(self) -> pd.DataFrame:
        """One row per finding: candidates, and the two denominators they are a share OF."""
        candidates, positives = self.candidates(), self.labelled_positive()
        rows = [
            {
                "finding": target,
                "candidates": int(candidates[:, index].sum()),
                "cells": int(candidates.shape[0]),
                "positives": int(positives[:, index].sum()),
                "share_cells": float(candidates[:, index].mean()),
                "share_positives": float(candidates[candidates[:, index] & positives[:, index], index].size)
                / max(int(positives[:, index].sum()), 1),
            }
            for index, target in enumerate(TARGETS)
        ]
        return pd.DataFrame(rows).sort_values("share_positives", ascending=False)

    def lines(self) -> list[str]:
        """The census, both denominators spelled out, worst-affected finding first."""
        table = self.per_finding()
        total, cells = int(table["candidates"].sum()), int(self.spoke.size)
        head = [
            f"{total} candidate cells of {cells} ({total / cells:.1%}), "
            f"over {self.spoke.shape[0]} studies and {len(TARGETS)} findings",
            f"the reader spoke on {self.spoke.mean():.1%} of cells; a cell it left silent can never be a candidate",
        ]
        return head + [
            f"  {row.finding:<18} {row.candidates:>4} cells  {row.share_cells:>6.2%} of all  "
            f"{row.share_positives:>6.2%} of its {row.positives} positives"
            for row in table.itertuples()
        ]


def main() -> None:
    """Count the cells a correction arm would touch, over both denominators."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--table", type=Path, required=True)
    command.parser.add_argument("--records", type=Path, nargs="+", required=True)
    arguments = command.parsed()

    table = LabelContradiction.read_table(arguments.table)
    command.report(LabelContradiction.align(table, arguments.records).lines())


if __name__ == "__main__":
    main()
