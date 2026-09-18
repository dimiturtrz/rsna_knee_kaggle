"""How big a mutual-supervision term would be, measured against the supervised term it is added to.

`python -m kneemri.evaluation.peer_signal --holdouts <dir> --members <stem> <stem> ...`

WHAT IT DECIDES. Deep mutual learning and codistillation add a peer term to the loss: each model in a
cohort is pulled toward its peers' posteriors as well as toward the labels. Whether that can do anything
at all rests on one quantity nobody had measured here — the size of the peer disagreement relative to
the residual against truth on the same rows. If two members agree tightly, the mutual gradient is a
rounding error beside the label gradient and a K-way cohort buys an expensive copy of one model.

WHY THE RATIO AND NOT THE KL. A KL in isolation has no scale: 0.02 nats is neither large nor small until
something says what it is being added to. Both terms are reported against the SAME rows and the same
label table, so `ratio` is the number that transfers — and it is the number a loss weight is chosen in.

WHAT IT IS NOT. Disagreement is NECESSARY for a mutual term to carry anything and it is not SUFFICIENT.
`distill_pairs` (+0.0016, under the paired floor) used this project's own fold-0 twins, which read a
ratio in this same band, and mutual training still did not extract it. A ratio here licenses a run; it
never predicts a gain, and no slot may be spent on the strength of this number.

WHY IT IS A FLOOR ON WHAT A COHORT WOULD SEE. This reads CONVERGED members on HELD-OUT rows, which is
the hardest place to find disagreement — each has already fit the training corpus, and the training-set
posteriors of a cohort early in its schedule are further apart than these. So a ratio measured here
understates a cohort's, and a ratio near zero here would be the stronger result of the two.

This is distinct from `kneemri.evaluation.member_distance`, which reads no labels by construction because
its question is seat redundancy and the truth vector would drag that reading back onto the quality axis.
Here the truth is the scale, and leaving it out would produce exactly the unitless number nobody can act
on that the other module's docstring warns about.
"""

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from kneemri.runs.holdout_record import SUFFIX, HoldoutReading, HoldoutRecord

logger = logging.getLogger(__name__)

PROBABILITY_FLOOR = 1e-6
"""How far a posterior is held off 0 and 1 before a log touches it. A member that saturates a column is
reporting certainty, not an infinity, and an unclamped KL would report the clamp's absence as the signal."""

SYMMETRIC_HALVES = 2
"""The symmetric KL averages the two directions. Named because neither model is the teacher here — that
asymmetry is what every distillation row in the ledger already tested, and all of them came back null."""


@dataclass(frozen=True)
class PeerSignal:
    """One pair's disagreement, carrying the supervised residual that gives it a scale."""

    left: str
    right: str
    studies: int
    peer: float
    supervised: float
    symmetric_kl: float
    correlation: float

    def ratio(self) -> float:
        """Peer term over supervised term — what a cohort's loss weight would be chosen against."""
        return self.peer / self.supervised if self.supervised else 0.0

    def line(self) -> str:
        """One row, both terms beside each other, because either alone is unreadable."""
        return (
            f"{self.left} vs {self.right}  n={self.studies}  "
            f"peer={self.peer:.4f}  supervised={self.supervised:.4f}  "
            f"ratio={self.ratio():.3f}  sym-KL={self.symmetric_kl:.5f}  corr={self.correlation:.4f}"
        )


class PeerComparison:
    """Members in, one signal per pair out, joined on the studies they actually share."""

    def __init__(self, readings: dict[str, HoldoutReading]) -> None:
        self._members = {stem: self._rows(reading) for stem, reading in readings.items()}

    @staticmethod
    def _rows(
        reading: HoldoutReading,
    ) -> tuple[
        dict[str, Float[NDArray[np.float64], " labels"]],
        dict[str, Float[NDArray[np.float64], " labels"]],
    ]:
        """A member's predictions and truth, keyed by study, so pairs join on names and never on order."""
        names = list(reading.studies)
        predictions = dict(zip(names, reading.predictions, strict=True))
        truth = dict(zip(names, reading.truth, strict=True))
        return predictions, truth

    @staticmethod
    def _divergence(
        left: Float[NDArray[np.float64], "studies labels"],
        right: Float[NDArray[np.float64], "studies labels"],
    ) -> Float[NDArray[np.float64], "studies labels"]:
        """Bernoulli KL per label, clamped, because every column here is an independent abnormality."""
        held = np.clip(left, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
        against = np.clip(right, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
        return held * np.log(held / against) + (1 - held) * np.log((1 - held) / (1 - against))

    def between(self, left: str, right: str) -> PeerSignal:
        """The pair's disagreement and the residual it has to be read against, on their shared studies."""
        left_predictions, left_truth = self._members[left]
        right_predictions, _ = self._members[right]
        shared = sorted(set(left_predictions) & set(right_predictions))
        if not shared:
            message = f"{left} and {right} share no studies - they were held out on different draws"
            raise ValueError(message)

        held = np.array([left_predictions[name] for name in shared])
        against = np.array([right_predictions[name] for name in shared])
        truth = np.array([left_truth[name] for name in shared])
        divergence = self._divergence(held, against) + self._divergence(against, held)

        return PeerSignal(
            left=left,
            right=right,
            studies=len(shared),
            peer=float(np.abs(held - against).mean()),
            supervised=float(np.abs(held - truth).mean()),
            symmetric_kl=float(divergence.mean() / SYMMETRIC_HALVES),
            correlation=float(np.corrcoef(held.ravel(), against.ravel())[0, 1]),
        )

    def across(self) -> list[PeerSignal]:
        """Every pair, so a cohort is judged on its weakest disagreement and not on its best one."""
        return [self.between(left, right) for left, right in combinations(self._members, 2)]


def main() -> None:
    """Report how large a mutual term would be beside the supervised one, for every pair given."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True, help="directory of .holdout.npz records")
    command.parser.add_argument("--members", nargs="+", required=True, help="the members' stems, two or more")
    arguments = command.parsed()

    readings = {}
    for stem in arguments.members:
        reading = HoldoutRecord(arguments.holdouts / f"{stem}{SUFFIX}").read()
        if reading is None:
            message = f"{stem} left no holdout record - a member that predicted nothing has no peer signal"
            raise SystemExit(message)
        readings[stem] = reading

    command.report([signal.line() for signal in PeerComparison(readings).across()])


if __name__ == "__main__":
    main()
