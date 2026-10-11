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
from kneemri.runs.holdout_record import HoldoutReading, HoldoutRecord

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
        """Pair the reader's verdicts with the mean out-of-sample prediction, over the UNION of the records.

        Each study is averaged across the records that HOLD IT OUT, and studies covered by one record are
        kept beside studies covered by five. Averaging first is the same LogitAverage the pool is scored
        under: one member's opinion is noisier than the pool's, and the conjunction is only worth drawing
        if the second instrument is a strong one.

        UNION RATHER THAN INTERSECTION, because the records that cover the CORPUS are complementary by
        construction. Every one of the 263 fold-0 records on disk holds out the same 909 studies, so an
        intersection was the whole answer while fold 0 was the only fold — and it silently became the
        empty set the moment `oof_folds` produced four disjoint fifths, which is the exact set of records
        the corpus-wide census needs. A census that cannot read complementary folds cannot leave the
        holdout, and a correction that never leaves the holdout cannot move a training label.
        """
        readings = [reading for record in records if (reading := HoldoutRecord(record).read()) is not None]
        if not readings:
            message = f"none of the {len(records)} named records carries a holdout array"
            raise ValueError(message)
        covered = sorted(set().union(*[set(reading.studies) for reading in readings]) & set(table.index))
        opinions, truths = cls._opinions(readings, covered)
        stacked = [np.mean(opinions[study], axis=0) for study in covered]
        truth = np.stack([truths[study] for study in covered])
        rows = table.loc[covered]
        verdicts = np.column_stack([rows[f"{target}{VERDICT_SUFFIX}"].to_numpy(str) for target in TARGETS])
        return cls(
            spoke=verdicts != SILENT,
            read_positive=verdicts == POSITIVE,
            predicted=np.stack(stacked),
            truth=truth,
        )

    @staticmethod
    def _opinions(
        readings: list[HoldoutReading], covered: list[str]
    ) -> tuple[dict[str, list[NDArray[np.float64]]], dict[str, NDArray[np.float32]]]:
        """Every record's prediction for each covered study, and the truth the last record to hold it carried."""
        opinions: dict[str, list[NDArray[np.float64]]] = {study: [] for study in covered}
        truths: dict[str, NDArray[np.float32]] = {}
        held = (
            (study, reading, position)
            for reading in readings
            for position, study in enumerate(reading.studies)
            if study in opinions
        )
        for study, reading, position in held:
            opinions[study].append(reading.predictions[position].astype(np.float64))
            truths[study] = reading.truth[position]
        return opinions, truths

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
