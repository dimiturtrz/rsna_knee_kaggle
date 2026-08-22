"""How much label structure survives in a model's RESIDUAL — the gate on a label-correlation head.

WHY THIS EXISTS BEFORE THE HEAD DOES. `knee-w8h6` proposes refining our twelve independent sigmoids with
a learned interaction over the label axis, because nothing in 107 runs has told the model that an ACL
tear drags effusion and contusion with it. That structure is real and clinical. But our twelve heads
already read ONE pooled study feature, so every correlation the IMAGE predicts is already available to
all twelve of them. What a label-interaction block can add is only the correlation left in the RESIDUAL —
what the image did not explain.

So the cheap question that gates the expensive build is not "do the labels correlate" (they do, hugely)
but "do the ERRORS correlate". If the residuals are near-independent, the shared feature has already
absorbed the joint structure and a refinement head has nothing to mine.

WHAT THIS MODULE DOES NOT ANSWER. A correlation matrix says structure EXISTS; it does not say the
structure is USABLE out of sample, because it is computed on the same rows it describes. That second
question is `label_probe.HeldOutProbe`, and it is the one that carries the verdict — this module is the
context that verdict is read against.

THE DENOMINATOR IS THE WHOLE POINT AND IT IS NOT THE ARRAY'S SHAPE. A pair the report never decided sits
at the 0.28 UNKNOWN default, so a residual computed there is the model's distance from a prior rather
than from a fact — and the corpus is 15.8% decided on Synovitis and 43.6% on Fracture. Scoring the full
909x12 rectangle would compute most of this matrix on cells nobody labelled. Every number below is taken
over DECIDED pairs only, pairwise for the matrices, and the counts are reported beside the values so the
denominator is visible rather than implied.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float
from numpy.typing import NDArray

from core.findings import TARGETS
from kneemri.evaluation.scores import MIDPOINT

UNKNOWN = "UNK"
"""The verdict column's value for a finding the report never decided. See the module docstring."""

VERDICT_SUFFIX = "__verdict"


@dataclass(frozen=True)
class CouplingReading:
    """One label pair's coupling, in the truth and in the residual, over the pairs BOTH labels decided."""

    first: str
    second: str
    truth: float
    residual: float
    studies: int


class LabelCoupling:
    """A model's label structure, split into what the image explained and what it left behind."""

    def __init__(
        self,
        predictions: Float[NDArray[np.float64], "studies labels"],
        truth: Float[NDArray[np.float32], "studies labels"],
        known: Bool[NDArray[np.bool_], "studies labels"],
    ) -> None:
        self.predictions = predictions
        self.positive = (truth > MIDPOINT).astype(np.float64)
        self.known = known
        self.residual = self.positive - predictions

    @staticmethod
    def decided(table: Path, studies: list[str]) -> Bool[NDArray[np.bool_], "studies labels"]:
        """Which (study, finding) pairs the report actually decided — this reading's DENOMINATOR.

        It lives on the coupling rather than in a mask class of its own because it is not a general
        service: it answers the one question the module docstring says every number here must be taken
        over, and the probe below takes the same mask from the same call.

        Aligned to a holdout record's STUDY ORDER, not to the table's, and joined by name rather than by
        position for the reason `HoldoutReading.aligned` gives: a fold change or a dropped study reindexes
        one side while leaving both shapes plausible, and the arithmetic then succeeds on a silently
        mismatched join.
        """
        rows = table.read_text(encoding="utf-8").splitlines()
        header = next(iter(rows)).split(",")
        columns = [header.index(f"{target}{VERDICT_SUFFIX}") for target in TARGETS]
        identity = header.index("StudyInstanceUID")
        decided = {
            fields[identity]: np.array([fields[column] != UNKNOWN for column in columns], dtype=np.bool_)
            for fields in (row.split(",") for row in rows[1:])
        }
        missing = [study for study in studies if study not in decided]
        if missing:
            message = f"{len(missing)} holdout studies are absent from {table.name}, first {missing[0]!r}"
            raise KeyError(message)
        return np.stack([decided[study] for study in studies])

    def pairs(self) -> list[CouplingReading]:
        """Every label pair, most residually-coupled first.

        Sorted by the RESIDUAL rather than the truth because the truth ordering is a property of knee
        pathology and is the same for every model, while the residual ordering is the only part that
        says anything about this one.
        """
        readings = [
            self.pair(first, second) for first in range(len(TARGETS)) for second in range(first + 1, len(TARGETS))
        ]
        return sorted(readings, key=lambda reading: -abs(reading.residual))

    def pair(self, first: int, second: int) -> CouplingReading:
        """Truth and residual correlation for one label pair, over the studies BOTH labels decided."""
        both = self.known[:, first] & self.known[:, second]
        return CouplingReading(
            first=TARGETS[first],
            second=TARGETS[second],
            truth=self.correlation(self.positive[both, first], self.positive[both, second]),
            residual=self.correlation(self.residual[both, first], self.residual[both, second]),
            studies=int(both.sum()),
        )

    @staticmethod
    def correlation(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
        """Pearson correlation, NaN when either side is constant.

        Constant rather than zero for `LabelScores.area_under_curve`'s reason: a label with one class
        present in the overlap is UNDEFINED, and reporting 0.0 would quietly average a non-measurement
        in with the measurements on exactly the rare findings this question is about.
        """
        spread = float(left.std()) * float(right.std())
        return float("nan") if spread == 0.0 else float(((left - left.mean()) * (right - right.mean())).mean() / spread)
