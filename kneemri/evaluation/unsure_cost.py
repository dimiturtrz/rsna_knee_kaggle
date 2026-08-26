"""What an `UNK` cell is actually worth: `python -m kneemri.evaluation.unsure_cost`.

THE QUESTION, AND IT IS ANSWERABLE. The competition's hidden test carries no report — `test.csv` holds
one column, `StudyInstanceUID` — so report text is TRAIN-TIME-ONLY supervision. A label cell is not a
fact we have to reproduce; it is a teaching signal whose strength we choose. That makes one number
decisive: on the studies a radiologist annotated, how often is a cell POSITIVE, split by whether our
reader said it could read that cell at all.

WHAT IS AT STAKE. `Supervision.MEASURED_AGREEMENT` emits 0.08/0.28/0.68/0.82/0.94, and a cell the reader
marked `UNK` is written at **0.28** — the mild-negative level. So every cell the reader admitted it could
not read is taught as a soft NO. If the radiologist's positive rate on those same cells is near the
decided cells' rate, that default is not a neutral placeholder: it is a systematic bias, applied to a
fifth of the English pairs and a third of the non-English ones, on exactly the findings where the image
carries more than the text does.

WHAT THIS MODULE DOES NOT DO. It does not flip a cell and it does not propose a value. Nobody in this
loop can read an MRI, so the only licensed move is to change how much the trainer BELIEVES a cell
(`Targets` already multiplies each cell's loss by its `__conf` weight). This measures the size of the
distortion so that move can be priced; the arm itself is knee-1ww2.

THE DENOMINATOR IS SMALL AND IT IS STATED EVERY TIME. `train.csv` annotates a study on all twelve
findings or on none, and there are 58 such studies — 1.32% of the corpus, 696 cells before the split.
An `UNK` share of a fifth leaves ~140 cells on one side. Cells within a study share a report and a
radiologist, so they are NOT independent draws; the interval here is bootstrapped over STUDIES, the
same protocol `reader_agreement` and `table_separation` are read under, and it is about twice the width
a cell-wise interval would report.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Bool, Float

from core import cli
from core.bootstrap import StudyBootstrap
from core.data.labels.confident_labels import UNSURE
from core.findings import TARGETS
from kneemri.evaluation.supervision_gap import PRESENT, SupervisionGap

logger = logging.getLogger(__name__)

UNSURE_VALUE = 0.28
"""What `Supervision.MEASURED_AGREEMENT` writes into a cell the reader could not decide. Quoted here as
the thing being judged, not as a knob — changing it is not the move this module argues for."""


@dataclass(frozen=True)
class UnsureSplit:
    """One finding's annotated cells, split by whether the reader was sure it had read them.

    A value rather than printed lines because the DECISION reads off the difference between the two
    positive rates, and a caller that re-derives it from prose gets it wrong — which is how the pinned
    board gap came to read 0.008 against the API's 0.015.
    """

    finding: str
    unsure_positive: float
    decided_positive: float
    unsure_cells: int
    decided_cells: int

    @property
    def taught_error(self) -> float:
        """How far the 0.28 default sits from what the radiologist actually found on those cells.

        Signed: positive means the default UNDER-states the finding, which is the direction that costs
        us, because it teaches a soft NO where the image may carry a YES.
        """
        return self.unsure_positive - UNSURE_VALUE


class UnsureCost:
    """The radiologist's verdict on the cells our reader declined to decide."""

    def __init__(self, gap: SupervisionGap | None = None) -> None:
        self.gap = gap if gap is not None else SupervisionGap()

    def unsure(self, labels: Path, identifiers: list[str]) -> Bool[np.ndarray, "studies targets"]:
        """Which annotated cells the reader marked `UNK`, in the annotations' own row and column order.

        A target with no `__verdict` column reads as decided everywhere rather than raising: a foreign
        reader that does not report its own confidence has not thereby become unsure of everything.
        """
        table = pd.read_csv(labels).set_index("StudyInstanceUID").reindex(identifiers)
        columns = [target for target in TARGETS if target in table.columns]
        verdicts = [
            (table[f"{target}__verdict"] == UNSURE).to_numpy()
            if f"{target}__verdict" in table.columns
            else np.zeros(len(identifiers), dtype=bool)
            for target in columns
        ]
        return np.stack(verdicts, axis=1)

    @staticmethod
    def _rate(annotations: Float[np.ndarray, "studies targets"], mask: Bool[np.ndarray, "studies targets"]) -> float:
        """The radiologist's positive rate over the selected cells, or NaN where none were selected."""
        selected = annotations[mask]
        return float((selected >= PRESENT).mean()) if selected.size else float("nan")

    def split(
        self,
        annotations: Float[np.ndarray, "studies targets"],
        unsure: Bool[np.ndarray, "studies targets"],
        finding: str,
        column: int,
    ) -> UnsureSplit:
        """One finding's two positive rates, over the cells the reader did and did not decide."""
        flag = unsure[:, column]
        return UnsureSplit(
            finding=finding,
            unsure_positive=self._rate(annotations[:, [column]], flag[:, None]),
            decided_positive=self._rate(annotations[:, [column]], ~flag[:, None]),
            unsure_cells=int(flag.sum()),
            decided_cells=int((~flag).sum()),
        )

    def overall(
        self,
        annotations: Float[np.ndarray, "studies targets"],
        unsure: Bool[np.ndarray, "studies targets"],
        *,
        resamples: int,
    ) -> tuple[float, tuple[float, float]]:
        """The pooled unsure-minus-decided positive rate, bracketed by resampling STUDIES.

        Pooled across findings because per-finding the unsure cells number in the tens; the pooled figure
        is the one the interval can carry, and the per-finding table below is read as a ranking only.
        """

        def measure(drawn: Bool[np.ndarray, " draw"]) -> float:
            rows, flags = annotations[drawn], unsure[drawn]
            return self._rate(rows, flags) - self._rate(rows, ~flags)

        point = self._rate(annotations, unsure) - self._rate(annotations, ~unsure)
        return point, StudyBootstrap(resamples=resamples).interval(annotations.shape[0], measure)

    def report(
        self,
        annotations: Float[np.ndarray, "studies targets"],
        unsure: Bool[np.ndarray, "studies targets"],
        *,
        resamples: int,
    ) -> list[str]:
        """Per finding, then the pooled difference with its interval and the taught-error verdict."""
        columns = list(TARGETS)[: annotations.shape[1]]
        lines = [
            f"{annotations.shape[0]} annotated studies, {annotations.size} cells, "
            f"{int(unsure.sum())} marked {UNSURE}, taught at {UNSURE_VALUE}"
        ]
        splits = [self.split(annotations, unsure, finding, column) for column, finding in enumerate(columns)]
        lines += [
            f"{split.finding:<24} unsure {split.unsure_cells:4d} pos {split.unsure_positive:.3f}   "
            f"decided {split.decided_cells:4d} pos {split.decided_positive:.3f}   "
            f"taught error {split.taught_error:+.3f}"
            for split in sorted(splits, key=lambda split: -split.unsure_cells)
        ]
        point, (floor, ceiling) = self.overall(annotations, unsure, resamples=resamples)
        verdict = "SEPARATED" if floor > 0.0 or ceiling < 0.0 else "unresolved"
        lines.append(f"pooled unsure - decided: {point:+.4f}  95% [{floor:+.4f}, {ceiling:+.4f}]  {verdict}")
        return lines


def main() -> None:
    """Score the radiologist's annotations on the cells our reader declined to decide."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--train", type=Path, required=True, help="the competition's train.csv")
    command.parser.add_argument("--labels", type=Path, required=True, help="our report-derived table")
    command.parser.add_argument("--resamples", type=int, default=1000)
    arguments = command.parsed()

    cost = UnsureCost()
    identifiers, annotations = cost.gap.annotated(arguments.train)
    command.report(cost.report(annotations, cost.unsure(arguments.labels, identifiers), resamples=arguments.resamples))


if __name__ == "__main__":
    main()
