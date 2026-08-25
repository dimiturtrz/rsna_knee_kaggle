"""How precisely each label's AUC can be measured at all: `python -m core.data.labels.label_power`.

Every per-label claim this project has made was read off a single holdout without an error bar, and
two of them shaped a plan. "MCL 0.755 and Fracture 0.784 are our worst labels, worth +0.0173 if
lifted" survived a re-pricing against the literature and was then refuted by our own folds: Fracture
is 0.784 on fold 0, 0.903 on fold 1 and 0.864 on fold 2 under an IDENTICAL recipe.

The cause is not instability in the model. It is that an AUC estimated from 62 positives has a
standard error near 0.04, so two holdouts disagree by a tenth without anything being wrong. This
module puts that number beside every label, so a per-label delta can be compared to the precision of
the instrument that produced it.

It also explains the project's noise floor rather than merely restating it. Twelve per-label AUCs, each
with its own sampling error, are averaged into the reported metric; the average of independent errors
shrinks by about the square root of twelve, which lands close to the 0.0062 spread measured across
seeds. The mean is readable BECAUSE its parts are noisy and many.
"""

import logging
import math
from pathlib import Path

import pandas as pd

from core import cli
from core.findings import TARGETS

logger = logging.getLogger(__name__)

FOLDS = 5
TYPICAL_AUC = 0.85
"""Where our labels actually sit, so the error bar is quoted where it will be read."""


class LabelPower:
    """The measurement precision of each label's AUC, from how many positives carry it.

    Uses the Hanley-McNeil standard error for an AUC, which is the standard analytic estimate and needs
    only the positive and negative counts — no predictions, so it can be run before a model exists and
    is a property of the CORPUS rather than of any run.
    """

    def __init__(self, labels: Path, folds: int = FOLDS) -> None:
        self.labels = labels
        self.table = pd.read_csv(labels)
        self.folds = folds

    def positives(self) -> dict[str, int]:
        """How many studies each label is asserted positive on, corpus-wide.

        Names the file when the columns are not there, because the obvious guess at "the label table"
        is the competition's own train.csv, and indexing it raised `KeyError: 'ACL__verdict'` nine
        frames deep in pandas — an error about a column, from a caller who asked about a path.
        """
        missing = [f"{target}__verdict" for target in TARGETS if f"{target}__verdict" not in self.table.columns]
        if missing:
            message = f"{self.labels} is not a verdict table: no {', '.join(missing[:3])}"
            raise ValueError(message)
        return {target: int((self.table[f"{target}__verdict"] == "YES").sum()) for target in TARGETS}

    def standard_error(self, positives: int, negatives: int, auc: float = TYPICAL_AUC) -> float:
        """Hanley-McNeil standard error of an AUC estimated from these counts.

        Returns NaN rather than 0 when a class is empty: an AUC is undefined there, and a zero would
        read as perfect precision — the most dangerous possible answer.
        """
        if positives <= 0 or negatives <= 0:
            return float("nan")
        first = auc / (2 - auc)
        second = 2 * auc**2 / (1 + auc)
        variance = (auc * (1 - auc) + (positives - 1) * (first - auc**2) + (negatives - 1) * (second - auc**2)) / (
            positives * negatives
        )
        return math.sqrt(variance)

    def report(self) -> str:
        """Each label's holdout positives and the precision of its AUC, worst-measured first."""
        counted = self.positives()
        studies = len(self.table) // self.folds
        rows = []
        for target, total in counted.items():
            positives = total // self.folds
            error = self.standard_error(positives, studies - positives)
            rows.append((error, target, positives))
        lines = [f"holdout of {studies} studies (one fold of {self.folds}); AUC error at {TYPICAL_AUC:.2f}"]
        for error, target, positives in sorted(rows, reverse=True):
            lines.append(f"  {target:18s} {positives:4d} positives   SE {error:.4f}   95% +/-{1.96 * error:.3f}")
        errors = [error for error, _, _ in rows if not math.isnan(error)]
        combined = math.sqrt(sum(error**2 for error in errors)) / len(errors)
        lines.append(f"\nmean of {len(errors)} labels: SE {combined:.4f} — the floor a single arm can be read against")
        return "\n".join(lines)


def main() -> None:
    """Report per-label measurement precision for the corpus."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, required=True)
    command.parser.add_argument("--folds", type=int, default=FOLDS)
    arguments = command.parsed()
    logger.info("%s", LabelPower(arguments.labels, arguments.folds).report())


if __name__ == "__main__":
    main()
