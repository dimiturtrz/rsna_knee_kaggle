"""Combine several report readers into one label table: `python -m core.data.labels.table_blend`.

WHY THIS EXISTS, AND IT IS NOT THE REASON IT LOOKS LIKE. Scored against the 58 radiologist-annotated
studies by `kneemri.evaluation.supervision_gap`, our table reads 0.8658 and the best public LLM table
reads 0.8927, which invites the conclusion that we read reports worse than the forum does. We do not.
Quantise that table onto our own five levels and it falls to 0.8445 — BELOW ours. At equal resolution our reader wins by
+0.021.

THE GAP IS TIE MASS, NOT READER QUALITY. `Supervision.MEASURED_AGREEMENT` emits five levels, so 38.5%
of all cell pairs are ties; an 85-level LLM probability ties 17.3%. AUC breaks no ties, so a coarse
scale is penalised for being coarse. The blend below is therefore a TIE-BREAKING device first and a
noise-cancelling one second — and the measurement says exactly that:

    ours + w*(foreign - MIDPOINT), w = 0.02   0.8874
                                   w = 0.05   0.8874
                                   w = 0.10   0.8874
                                   w = 0.25   0.8872
    mean(ours, steven)                        0.8885
    mean(ours, steven, lixin)                 0.8912

The w-invariance IS the mechanism proof. AUC is rank-based, so any weight small enough to keep our
ordering dominant yields the SAME ordering, and the whole +0.0216 is tie-breaking. A tie-break that
paid for itself by reweighting would move when the weight moved.

WHAT THE NUMBERS DO NOT SAY. `MEAN` over three readers reaches 0.8912 and the best single foreign table
reads 0.8927, so the blend does not beat its best input — which is what averaging independent readers
is supposed to do. On 58 studies and ~696 cells that difference has no computed interval and none of
these should be quoted as separated. The honest claim is narrow: blending recovers most of what our
scale's coarseness costs, and it does so without adopting a reader we cannot audit.

LEAKED TABLES ARE REFUSED BY NAME. Three public tables score exactly 1.0000 against the gold set
because they BAKE IT IN — `yunusgmsoy/report_labels_v5`, `gxpixel/train_labels_final` (which ships
`gold_58.csv` and an `is_gold` column) and `rayanbabur/train_targets_soft` (which ships
`gold_benchmark_58.csv`). A perfect score against the ruler is the signature of a table that has read
the ruler, so `LEAKED` blocks them here rather than leaving the trap for the next reader of the numbers.

TRAINING ON THE RESULT NEEDS `--eval-labels`. Our fold-0 holdout is scored against `report_labels_v2`
itself, so an arm that trains on a blend and is scored unpinned measures DISAGREEMENT WITH V2, not
supervision quality. Pin the ruler to v2 and let the 58 gold and the board be the honest tests.
"""

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from core import cli
from core.data.labels.reader_agreement import MIDPOINT
from core.findings import STUDY, TARGETS

logger = logging.getLogger(__name__)

CONFIDENCE_SUFFIX = "__conf"
"""Confidence columns are carried from the base table untouched. They weight the loss rather than
setting the target, and a foreign reader's confidence is confidence in ITS extraction, not in ours."""

TIEBREAK_WEIGHT = 0.05
"""How far a foreign reader may move a base cell under `TIEBREAK`. Any value small enough to keep the
base ordering dominant gives the same AUC — measured at 0.02, 0.05 and 0.10 — so this is a legibility
choice, not a tuned one. At 0.25 the ordering starts to move and the score falls."""

LEAKED: tuple[str, ...] = ("report_labels_v5", "train_labels_final", "train_targets_soft")
"""Tables that score 1.0000 against the gold 58 because they contain it. Refused by filename stem."""

MEAN = "mean"
TIEBREAK = "tiebreak"


class LabelBlend:
    """One base table plus foreign readers of the same reports, combined on the base table's index.

    The base table's row set is the output's row set. A foreign reader that is missing a study or a
    finding contributes nothing there rather than shrinking the corpus — a blend that silently dropped
    the rows one reader missed would change the training denominator, which is this project's most
    repeated error.
    """

    def __init__(self, base: pd.DataFrame, foreign: Sequence[pd.DataFrame]) -> None:
        self.base = base.set_index(STUDY) if STUDY in base.columns else base
        self.foreign = [table.set_index(table.columns[0]).reindex(self.base.index) for table in foreign]

    def findings(self) -> list[str]:
        """The twelve findings, as the base table carries them.

        Taken from `TARGETS` rather than from "every column that is not a confidence" — a real label
        table also carries `Report` and `report_language`, and asking pandas to average a string column
        is how the first run of this module died.
        """
        return [column for column in TARGETS if column in self.base.columns]

    def readings(self, finding: str) -> list[pd.Series]:
        """Every reader's column for one finding — the base first, then whichever foreigns carry it."""
        return [self.base[finding]] + [table[finding] for table in self.foreign if finding in table.columns]

    @staticmethod
    def read(path: Path) -> pd.DataFrame:
        """One reader's table, refusing by name any table known to contain the gold set it is scored on."""
        if path.stem in LEAKED:
            raise ValueError(f"{path.stem} bakes the gold 58 into its rows — see LEAKED")
        return pd.read_csv(path)

    @staticmethod
    def _mean(readings: list[pd.Series]) -> pd.Series:
        """Equal-weight mean over the readers that carry this finding, ignoring the ones that do not."""
        return pd.concat(readings, axis=1).mean(axis=1)

    @staticmethod
    def _tiebreak(readings: list[pd.Series]) -> pd.Series:
        """The base's ordering, nudged by the foreign consensus only far enough to separate equal cells."""
        base = readings[0]
        if len(readings) == 1:
            return base
        nudge = pd.concat(readings[1:], axis=1).mean(axis=1).fillna(MIDPOINT) - MIDPOINT
        return base + TIEBREAK_WEIGHT * nudge

    STRATEGIES: ClassVar[dict[str, Callable[[list[pd.Series]], pd.Series]]] = {MEAN: _mean, TIEBREAK: _tiebreak}
    """Declaration-only dispatch, so a new way to combine readers is a function and never a branch."""

    def blended(self, mode: str) -> pd.DataFrame:
        """The combined table, one column per finding, with the base's confidence columns carried over."""
        combine = self.STRATEGIES[mode]
        table = self.base.copy()
        for finding in self.findings():
            table[finding] = combine(self.readings(finding))
        return table.reset_index()

    def coverage(self) -> list[str]:
        """How much of the base each foreign reader actually covers, per reader, over the base's rows.

        Stated because a foreign table that aligned on nothing blends to the base unchanged and reports
        no error — the failure mode this module hit on its first run, where a positional join wrote NaN
        into every cell and the ruler answered a flat 0.5.
        """
        rows = len(self.base)
        lines = [f"{rows} base studies, {len(self.findings())} findings, {len(self.foreign)} foreign readers"]
        for position, table in enumerate(self.foreign):
            shared = [finding for finding in self.findings() if finding in table.columns]
            present = int(table[shared].notna().any(axis=1).sum()) if shared else 0
            lines.append(f"  reader {position}: {len(shared):3d} shared findings, {present}/{rows} studies aligned")
        return lines


def main() -> None:
    """Blend a base label table with one or more foreign readers of the same reports."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--base", type=Path, required=True, help="the table whose rows and scale lead")
    command.parser.add_argument("--foreign", type=Path, nargs="+", required=True, help="other readers, same reports")
    command.parser.add_argument("--mode", choices=sorted(LabelBlend.STRATEGIES), default=MEAN)
    command.parser.add_argument("--out", type=Path, required=True, help="where the blended table is written")
    arguments = command.parsed()

    blend = LabelBlend(LabelBlend.read(arguments.base), [LabelBlend.read(path) for path in arguments.foreign])
    command.report(blend.coverage())
    table = blend.blended(arguments.mode)

    destination = arguments.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    levels = int(np.unique(np.round(table[blend.findings()[0]].dropna().to_numpy(), 6)).size)
    command.report([f"{len(table)} rows -> {destination}", f"{levels} distinct levels on {blend.findings()[0]}"])


if __name__ == "__main__":
    main()
