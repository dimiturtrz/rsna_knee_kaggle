"""Does fp16 SERVING tie enough studies for the vote rule to matter? `Fp16Ties`.

`python -m kneemri.kaggle.fp16_ties`

WHY THIS EXISTS. `Vote.ranks` was ordinal until `fc8a8db1` and is midranks now. The fix was argued from a
mechanism rather than measured: a member emitting one value everywhere votes a full sweep from 0 to 1
ordered by corpus position under ordinal ranks, and fp16 serving — eleven mantissa bits — collapses a
saturated sigmoid onto a handful of distinct values, so the ties that never occur in float64 training
arrays are reachable at submission time. Measured on the training arrays the tie mass is 7.9e-05 and the
blend mean does not move, which says the fix is free; it does NOT say the fix is worth anything.

THE INSTRUMENT. Take the greedy-optimal seven, round-trip each member through float16 exactly as the
kernel serves it, and read four numbers: the tie mass and the blended score under BOTH rank rules, at
both precisions. The float64 rows are the control the fix was already checked against. The float16 rows
are the question — how much of a score does the ordinal rule lose once the values it ranks are the ones
that actually ship.

WHAT IT DECIDES. A slot. If the ordinal rule loses nothing at fp16, then the board number was never
paying for this defect and a submission that re-ships the champion tuple under the fixed vote would buy
nothing; the slot goes elsewhere. If it loses something, that loss is a gain already banked in code, and
the re-ship is an instrument worth a slot because the board is the only place it can be confirmed.

WHAT IT DOES NOT ANSWER. Serving also runs the trunk in fp16, and this reads only the vote: the member
predictions here are the float64 ones cast down, not predictions a half-precision forward pass produced.
So the tie mass is a FLOOR on the served one, and the cost it prints is a floor on the served cost.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float

from core import cli
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.vote import RankAverage, Vote
from kneemri.kaggle.seat_requirement import DEFAULT_BASE

logger = logging.getLogger(__name__)

SERVED = np.float16
"""The precision the kernel runs at, by owner directive: bf16 training and fp16 serving."""

TRAINING = "float64 (training)"
SERVING = "float16 (serving)"


@dataclass(frozen=True)
class TieReading:
    """One precision: how much of the corpus ties, and what each rank rule scores on it."""

    precision: str
    tie_mass: float
    midrank_score: float
    ordinal_score: float

    def cost(self) -> float:
        """What the ORDINAL rule loses against midranks here — the number the fix is worth."""
        return self.midrank_score - self.ordinal_score


class Fp16Ties:
    """Reads both rank rules at both precisions on one pool and one truth."""

    def __init__(self, models: Path, base: tuple[str, ...] = DEFAULT_BASE) -> None:
        self.models = models
        self.base = base
        self.scores = LabelScores()

    def seated(self) -> Pool:  # devtools-ignore: test-mirror
        """The greedy-optimal seven on their shared studies, on the corpus ruler."""
        mix = RecordMix([self.models / f"{name}.pt" for name in self.base], foreign=True)
        mix.narrow()
        return mix.per_member(RankAverage())

    @staticmethod
    def served(prediction: Float[np.ndarray, "studies targets"]) -> Float[np.ndarray, "studies targets"]:
        """The member as the KERNEL sees it: float64 that has been through fp16 and back."""
        return prediction.astype(SERVED).astype(np.float64)

    @staticmethod
    def tie_mass(prediction: Float[np.ndarray, "studies targets"]) -> float:
        """The share of study-label cells whose value is shared with at least one other study.

        Per label rather than over the flattened array, because the rank rules order each label on its
        own and a value repeated across two different findings is not a tie either rule can see.
        """
        tied = 0
        for label in range(prediction.shape[1]):
            _, counts = np.unique(prediction[:, label], return_counts=True)
            tied += int(counts[counts > 1].sum())
        return tied / prediction.size

    @staticmethod
    def ordinal(prediction: Float[np.ndarray, "studies targets"]) -> Float[np.ndarray, "studies targets"]:
        """Ranks EXACTLY as `Vote.ranks` spelled them before `fc8a8db1`, kept so the fix has a control.

        A claim about what the old rule cost needs the old rule to measure against, and the old rule is
        no longer anywhere in the tree.
        """
        studies = prediction.shape[0]
        ranked = np.empty_like(prediction)
        for label in range(prediction.shape[1]):
            ranked[:, label] = prediction[:, label].argsort().argsort()
        return ranked / max(studies - 1, 1)

    def reading(
        self,
        predictions: list[Float[np.ndarray, "studies targets"]],
        truth: Float[np.ndarray, "studies targets"],
        precision: str,
    ) -> TieReading:
        """Both rules blended over the same members, so the only difference between the two scores is the rule."""
        midrank = np.mean([Vote.ranks(member) for member in predictions], axis=0)
        ordinal = np.mean([self.ordinal(member) for member in predictions], axis=0)
        return TieReading(
            precision=precision,
            tie_mass=float(np.mean([self.tie_mass(member) for member in predictions])),
            midrank_score=self.scores.mean(midrank, truth),
            ordinal_score=self.scores.mean(ordinal, truth),
        )

    def readings(self, pool: Pool) -> list[TieReading]:
        """The control precision and the served one, in that order."""
        return [
            self.reading(pool.predictions, pool.truth, TRAINING),
            self.reading([self.served(member) for member in pool.predictions], pool.truth, SERVING),
        ]

    def report(self, readings: list[TieReading] | None = None) -> list[str]:
        """The four numbers, and what they say about spending a slot on the re-ship."""
        readings = self.readings(self.seated()) if readings is None else readings
        lines = [
            f"the greedy-optimal {len(self.base)}, ranked by both rules at both precisions:",
            "  precision                tie mass     midranks      ordinal      ordinal cost",
        ]
        lines += [
            f"  {reading.precision:<22}{reading.tie_mass:10.6f}{reading.midrank_score:13.6f}"
            f"{reading.ordinal_score:13.6f}{reading.cost():+18.6f}"
            for reading in readings
        ]
        served = readings[-1]
        lines.append(
            f"the midrank fix is worth {served.cost():+.6f} at serving precision, against a tie mass of "
            f"{served.tie_mass:.6f}"
        )
        return lines


def main() -> None:
    """Price the rank-rule fix at the precision that ships, before a slot is spent confirming it."""
    command = cli.Command(__doc__, logger)
    arguments = command.parsed()
    root = command.rooted(arguments)
    command.report(Fp16Ties(root.models()).report())


if __name__ == "__main__":
    main()
