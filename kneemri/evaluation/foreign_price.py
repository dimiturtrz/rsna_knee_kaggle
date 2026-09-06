"""Price a foreign fold family honestly: each fold on the studies it held out.

`python -m kneemri.evaluation.foreign_price`.

TWO READINGS, AND THEY ANSWER DIFFERENT QUESTIONS.

PER-FOLD is the solo number: fold *f* scored on `grp == f` only, roughly a fifth of the holdout. It says
how good one of their checkpoints is, and it is the only number that may be compared to one of ours.

FOLD-MATCHED POOLED is over every study, each one scored by the ONE fold that never saw it. It covers the
whole holdout without a single contaminated cell, and it is what a member built from this family would be
worth as a seat. It is NOT their ensemble: their kernel rank-means five folds over studies that four of
them trained on, which is exactly the reading this module exists to refuse.

Their `A5` rank-mean is deliberately not reproduced here. Combining folds is a serving decision; combining
them over our holdout would mix four in-sample opinions into every study and read high, which is the
`raptor_solo` defect wearing an ensemble's clothes.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.findings import STUDY, TARGETS
from kneemri.evaluation.foreign_split import FOLDS, ForeignSplit
from kneemri.evaluation.protocol_prior import Ordering
from kneemri.runs.holdout_record import FOREIGN

logger = logging.getLogger(__name__)

NAN_CEILING = 0.05
"""How much absent truth a corpus reading may carry before the table is the wrong file, not a gap."""


@dataclass(frozen=True)
class ForeignPrice:
    """Per-fold and fold-matched macro AUC for a family of foreign fold checkpoints."""

    predictions: Float[NDArray[np.float32], "folds studies labels"]
    truth: Float[NDArray[np.float32], "studies labels"]
    split: ForeignSplit

    @staticmethod
    def macro(
        scores: Float[NDArray[np.float32], "studies labels"], truth: Float[NDArray[np.float32], "studies labels"]
    ) -> float:
        """Mean AUC over the labels that are not constant on this subset; a constant label is nan, never 0.5."""
        columns = [Ordering.auc(scores[:, label], truth[:, label]) for label in range(truth.shape[1])]
        return float(np.nanmean(columns))

    def solo(self, fold: int) -> float:
        """Fold `fold` on the studies it held out — the only cells it may be read on."""
        clean = self.split.out_of_sample(fold)
        return self.macro(self.predictions[fold][clean], self.truth[clean])

    def pooled(self) -> float:
        """Every study scored by the one fold that never saw it: full coverage, no contaminated cell."""
        return self.macro(self.matched(), self.truth)

    def matched(self) -> Float[NDArray[np.float32], "studies labels"]:
        """The fold-matched column: every study's score from the one fold that never saw it.

        This is the only combination of theirs that may enter our seating. It is also a FLOOR on what a
        served member would do, because their kernel rank-means five folds per study and this is one.
        """
        return self.predictions[self.split.fold, np.arange(len(self.split.studies))]

    @staticmethod
    def corpus_rows(table: pd.DataFrame, studies: list[str]) -> list[int]:
        """The served positions whose study the truth table actually carries, in the served order.

        The corpus scan serves every study the SERIES tables name; the label table is a different file
        with its own coverage, and joining them by assumption is how a reading acquires a denominator
        nobody stated. The positions this returns are the denominator, and the caller prints it.
        """
        known = {str(one) for one in table[STUDY]}
        return [index for index, study in enumerate(studies) if study in known]

    @staticmethod
    def truth_of(
        holdout: Path | None, studies: list[str], table: pd.DataFrame
    ) -> Float[NDArray[np.float32], "studies labels"]:
        """The truth for the served studies: our holdout record's, or a per-study label table's.

        A holdout npz is the reading when the subject is the 909 — it carries the exact truth our members
        were scored against, and its study ORDER is the contract this module holds the serving to. The
        CORPUS reading has no such file by construction: their five folds hold out all 4407 between them
        by THEIR split, and nothing on our side ever wrote that list. There a label table is the truth,
        which is the point rather than a fallback — the census this feeds asks which of the table's own
        cells two independent instruments both contradict.

        THE TABLE IS NOT `train.csv`, AND THIS REFUSES IT. The competition's own file carries a label for
        58 of our 4407 studies: 52188 of its 52884 target cells are nan, every target column nan on 4349
        rows. Passed here it produced a fold-matched 0.8211 that read like a family being worth 0.06 less
        than it is, on a truth that was almost entirely absent. What our members are actually scored
        against is the reader's table (`labels/report_labels_v2.csv`), which is complete and soft-valued.
        """
        if holdout is None:
            rows = table.drop_duplicates(subset=STUDY).astype({STUDY: str}).set_index(STUDY)
            truth = rows.loc[studies, list(TARGETS)].to_numpy(np.float32)
            absent = float(np.isnan(truth).mean())
            if absent > NAN_CEILING:
                message = f"{absent:.3%} of the truth cells are nan: this table does not label these studies"
                raise ValueError(message)
            return truth
        ours = np.load(holdout, allow_pickle=True)
        if studies != [str(one) for one in ours["studies"]]:
            message = "the served studies are not our holdout's studies, in order"
            raise ValueError(message)
        return ours["truth"].astype(np.float32)

    def lines(self) -> list[str]:
        """The reading, with the denominator on every row because that is what makes it quotable."""
        sizes = self.split.sizes()
        return [
            *[
                f"fold {fold}: {self.solo(fold):.4f} solo on its {sizes[fold]} out-of-sample studies"
                for fold in range(FOLDS)
            ],
            f"fold-matched pooled: {self.pooled():.4f} over all {len(self.split.studies)} studies, none contaminated",
            "their own rank-mean over these studies is NOT reported: four of five folds trained on each one",
        ]


def main() -> None:
    """Read a foreign per-fold prediction file and say what the family is worth on studies it never saw."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--predictions", type=Path, required=True, help="the npz foreign_folds wrote")
    command.parser.add_argument(
        "--holdout",
        type=Path,
        default=None,
        help="our holdout npz, for truth and study order; omit for a corpus scan, whose truth is --labels",
    )
    command.parser.add_argument("--labels", type=Path, required=True, help="train.csv: their split hashes its Report")
    command.parser.add_argument(
        "--truth",
        type=Path,
        default=None,
        help="the per-study label table a corpus scan is scored against (labels/report_labels_v2.csv)",
    )
    command.parser.add_argument(
        "--emit", type=Path, default=None, help="write the fold-matched column as a holdout record, priceable as a seat"
    )
    arguments = command.parsed()

    served = np.load(arguments.predictions, allow_pickle=True)
    studies = [str(one) for one in served["studies"]]
    if [str(one) for one in served["labels"]] != list(TARGETS):
        message = f"their label order is not ours: {list(served['labels'])}"
        raise ValueError(message)
    table = pd.read_csv(arguments.labels)
    predictions = served["predictions"]
    if arguments.holdout is None:
        if arguments.truth is None:
            message = "a corpus scan needs --truth: --labels is their split's Report source, not a label table"
            raise SystemExit(message)
        labelled = pd.read_csv(arguments.truth)
        rows = ForeignPrice.corpus_rows(labelled, studies)
        logger.info("corpus reading over %d of %d served studies the truth table labels", len(rows), len(studies))
        studies = [studies[index] for index in rows]
        predictions = predictions[:, rows]
        truth = ForeignPrice.truth_of(None, studies, labelled)
    else:
        truth = ForeignPrice.truth_of(arguments.holdout, studies, table)
    price = ForeignPrice(predictions, truth, ForeignSplit.of(studies, table))
    if arguments.emit:
        np.savez(
            arguments.emit,
            studies=np.array(studies),
            predictions=price.matched(),
            truth=truth,
            epoch=np.array(-1),
            source=np.array(FOREIGN),
        )
        logger.info("wrote %s", arguments.emit)
    command.report(price.lines())


if __name__ == "__main__":
    main()
