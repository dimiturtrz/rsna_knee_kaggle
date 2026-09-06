"""What a member's own board score implies for every weight it could be seated at.

`python -m kneemri.kaggle.mix_curve --solo 0.914`.

WHY THIS EXISTS. `foreignseat_v1` seated the foreign DINOv3 family at 6/13 and scored 0.929 where the
identical seven without it scored 0.944. Offline the same seat reads +0.0178 at the same weight. Three
dials moved at once and no offline ruler we own could name which carried the drop, so a slot went to the
family ALONE — the one quantity every branch of that argument is a function of. This module is what turns
that single number into the whole curve, so the next weight does not cost a slot to price.

THE MODEL, AND WHAT IS ASSUMED. A rank-mean blend of two producers at weight `w` is not the linear mix
of their scores: diversity buys a bump. Offline that bump is measured, not assumed —

    synergy(w) = blend(w) - [(1 - w) * base + w * solo]

— over the ten weights `seat_weight` served (`logs/seat_weight_served.log`). The board curve is then the
line through the two board anchors plus that same bump:

    board(w) = (1 - w) * board_base + w * board_solo + synergy(w)

The ONE assumption is that the bump transfers. It is stated here rather than buried because the whole
value of the instrument is that a measured solo tests it: with the board blend already known at one
weight, the solo reading says whether the offline synergy survived, and `transferred` reports the
fraction that did. A fraction near 1.0 means the offline ruler prices this axis; near 0.0 means diversity
that is real on our 909 buys nothing on the hidden split, which is a finding about the RULER and not
about the member.

WEIGHTS ARE SEATS. `seat_weight` reports repetitions — a candidate at `k` copies beside `n` members — and
a weight is the fraction those copies take of the final table, `k / (n + k)`. The two are the same
quantity spelled differently, and every number here is the fraction.
"""

import logging
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli

logger = logging.getLogger(__name__)

SERVED_BASE = 8
"""Members in the roster `seat_weight` priced the foreign candidate against, in `seat_weight_served.log`."""

BOARD_BASE = 0.942
"""`onegroup_v1` — our seven at the collapsed read the foreign entries serve. NOT the champion's 0.944.

The champion serves three read groups and no entry carrying a foreign producer fits three groups in nine
hours, so every foreign entry inherits `onegroup_v1`'s measured -0.002. Charging one of them against
0.944 would charge it for a cost another slot already priced.
"""

BOARD_BLEND = 0.929
"""`foreignseat_v1`, ref 56060097: our seven plus the foreign family at 6/13, RankAverage."""

BLEND_WEIGHT = 6.0 / 13.0
"""The weight `BOARD_BLEND` was served at — six foreign seats beside our seven."""


@dataclass(frozen=True)
class SeatCurve:
    """A candidate's offline seat curve: what the blend scored at each number of copies.

    `base` is the roster without the candidate and `solo` is the candidate alone on the same studies, so
    the curve has both endpoints and the bump between them is a measurement rather than a fit.
    """

    base: float
    solo: float
    blends: tuple[float, ...]
    members: int = SERVED_BASE

    @property
    def weights(self) -> Float[NDArray[np.float64], " copies"]:
        """The fraction of the final table each seat count takes."""
        copies = np.arange(1, len(self.blends) + 1, dtype=np.float64)
        return copies / (self.members + copies)

    def linear(self) -> Float[NDArray[np.float64], " copies"]:
        """What the blend would score at each weight if mixing bought nothing but the average."""
        weights = self.weights
        return (1.0 - weights) * self.base + weights * self.solo

    def synergy(self) -> Float[NDArray[np.float64], " copies"]:
        """The bump: how far each measured blend sits above its own linear mix."""
        return np.asarray(self.blends, dtype=np.float64) - self.linear()

    def synergy_at(self, weight: float) -> float:
        """The bump at an arbitrary weight, interpolated between the served ones.

        Outside the served range the nearest measured bump is held rather than extrapolated: the curve
        is flat-to-falling above six seats, and a line fitted through its top would invent a rise.
        """
        return float(np.interp(weight, self.weights, self.synergy()))


@dataclass(frozen=True)
class MixCurve:
    """The board curve a measured solo implies, and the synergy question it answers."""

    curve: SeatCurve
    board_base: float = BOARD_BASE

    def board_at(self, solo: float, weight: float, *, synergy_share: float = 1.0) -> float:
        """What the board would read at `weight`, given the member's own board score.

        `synergy_share` scales the offline bump: 1.0 assumes it transfers whole, 0.0 assumes the board
        pays only the linear mix. Both are worth reading, because the gap between them is the entire
        disagreement this instrument exists to expose.
        """
        return (1.0 - weight) * self.board_base + weight * solo + synergy_share * self.curve.synergy_at(weight)

    def implied_solo(self, board_blend: float, weight: float, *, synergy_share: float = 1.0) -> float:
        """The member's board score that a MEASURED blend implies, under a given synergy assumption.

        Run at share 1.0 and 0.0 it brackets the member: the two answers are what the solo slot chooses
        between.
        """
        without = board_blend - (1.0 - weight) * self.board_base - synergy_share * self.curve.synergy_at(weight)
        return without / weight

    def transferred(self, solo: float, board_blend: float, weight: float) -> float:
        """The FRACTION of the offline bump the board actually paid, once the solo is known.

        This is the reading the slot buys. Near 1.0 the offline seat ruler prices this axis and the
        -0.013 was the member. Near 0.0 the diversity is real on the 909 and worth nothing on the hidden
        split, which condemns the ruler rather than the member. Negative means the blend scored BELOW
        its own linear mix, which no offline reading of this family predicts at all.
        """
        bump = self.curve.synergy_at(weight)
        if bump == 0.0:
            return float("nan")
        return float((board_blend - (1.0 - weight) * self.board_base - weight * solo) / bump)

    def best_weight(self, solo: float, *, synergy_share: float = 1.0) -> tuple[float, float]:
        """The weight with the highest implied board score, and that score.

        Zero is included as a candidate, because "seat it at nothing" is an answer this curve is allowed
        to give and is the one a member below the base earns whenever the bump does not transfer.
        """
        weights = np.concatenate(([0.0], self.curve.weights))
        scores = [self.board_at(solo, float(w), synergy_share=synergy_share) for w in weights]
        best = int(np.argmax(scores))
        return float(weights[best]), float(scores[best])

    def lines(self, solo: float) -> list[str]:
        """The report: the two bracketing curves, the transfer fraction, and the best weight under each."""
        rows = [
            f"solo {solo:.4f} on the board, base {self.board_base:.4f}, "
            f"measured blend {BOARD_BLEND:.4f} at weight {BLEND_WEIGHT:.4f}",
            f"offline synergy at that weight: {self.curve.synergy_at(BLEND_WEIGHT):+.4f}",
            f"the board paid {self.transferred(solo, BOARD_BLEND, BLEND_WEIGHT):.2f} of it",
            "",
            "weight   synergy transfers whole   no synergy at all",
        ]
        for weight in self.curve.weights:
            whole = self.board_at(solo, float(weight), synergy_share=1.0)
            none = self.board_at(solo, float(weight), synergy_share=0.0)
            rows.append(f"{weight:.4f}   {whole:.4f}                  {none:.4f}")
        for share, name in ((1.0, "whole"), (0.0, "none")):
            weight, score = self.best_weight(solo, synergy_share=share)
            rows.append(f"best weight with synergy {name}: {weight:.4f} at {score:.4f}")
        return rows


FOREIGN_SERVED = SeatCurve(
    base=0.8835,
    solo=0.8812,
    blends=(0.8908, 0.8954, 0.8982, 0.8999, 0.9009, 0.9013, 0.9014, 0.9013, 0.9011, 0.9007),
)
"""`logs/seat_weight_served.log`, RankAverage: the foreign family against the served base of eight.

Transcribed rather than recomputed because the log is the artefact the arm left, and a recomputation
would price a member the entry did not serve. The base is the roster without the candidate and the solo
is `foreign_dinov3_matched.holdout.npz` pooled fold-matched over all 909 studies.
"""


def main() -> None:  # devtools-ignore: test-mirror
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--solo", type=float, required=True, help="the member's own board score")
    arguments = command.parsed()
    for line in MixCurve(FOREIGN_SERVED).lines(arguments.solo):
        logger.info(line)


if __name__ == "__main__":
    main()
