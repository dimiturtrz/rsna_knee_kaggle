"""Write a label table with the reader's unsure calls removed.

`python -m kneemri.labels.confident_labels --labels report_labels_v2.csv --out confident.csv`

The field's strongest public notebooks spend most of their lines on a multilingual rule-based report
reader, which says label quality is where they believe the score is. Our own label experiment measured
−0.0008 between two tables and was run once, against a noise floor since measured at 0.006 — so it could
not have detected an effect in either direction, and we do not actually know whether label content moves
our score at all.

Porting a two-thousand-line lexicon to find out is the expensive way round. This is the cheap way: take
the table we already have and remove what its own reader marked `UNK`, which is 20% of English pairs and
34% of non-English ones. A model trained on what is left is trained on strictly less, but cleaner,
supervision. If that cannot move the number by more than the noise floor, the label axis is not our
lever whatever anyone else spends their lines on — and if it can, the port is worth costing.

Removal here means clearing the VALUE, not zeroing the confidence: `Targets.load` skips a target with no
value and gives it weight 0, which is what "we do not know" should mean. Zeroing the confidence would
instead leave the floor pulling toward a value the reader never asserted.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from kneemri.models.model import TARGETS

UNSURE = "UNK"

logger = logging.getLogger(__name__)


class ConfidentLabels:
    """A report-derived table with the reader's unsure verdicts struck out."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table

    def struck(self) -> pd.DataFrame:
        """A copy whose `UNK` targets carry no value, and so no training weight."""
        out = self.table.copy()
        for target in TARGETS:
            verdict = f"{target}__verdict"
            if verdict in out.columns:
                out.loc[out[verdict] == UNSURE, target] = pd.NA
        return out

    def struck_share(self) -> dict[str, float]:
        """What fraction of each target's rows the reader was unsure about."""
        return {
            target: float((self.table[f"{target}__verdict"] == UNSURE).mean())
            for target in TARGETS
            if f"{target}__verdict" in self.table.columns
        }


def main() -> None:
    """Write the struck-out table beside the original."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    labels = ConfidentLabels(pd.read_csv(arguments.labels))
    shares = labels.struck_share()
    logger.info("struck per target: %s", {name: round(value, 3) for name, value in shares.items()})
    logger.info("mean struck share: %.3f", sum(shares.values()) / max(len(shares), 1))
    labels.struck().to_csv(arguments.out, index=False)
    logger.info("wrote %s", arguments.out)


if __name__ == "__main__":
    main()
