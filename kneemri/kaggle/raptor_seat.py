"""Seating the raptor member beside our own, as a seventh seat in rank space.

WHY RANK SPACE AND NOT PROBABILITY SPACE. The two tables come out of unrelated heads: ours is a
logit-average over six members sharing one decode, theirs is a sigmoid off a per-finding attention pool
over a five-slot volume. Their calibrations have nothing to do with each other, and the metric reads
rank order, so averaging the probabilities would let whichever head happens to be more confident decide
the blend. `Vote.ranks` is the ranking every ensemble number this project has ever reported was built
on, reused here rather than re-spelled — a second ranking would report a difference that is partly a
difference in how the two sides were ranked.

WHY 1/7 AND NOT A FITTED WEIGHT. There is nothing to fit it on. The raptor arm is `--fold -1`, so every
study of our holdout is inside its training set and no offline pairing with our members exists; tuning
the weight on the board would cost one slot per value. One member entering a six-member roster takes one
seat of seven, which is the weight every seating entry in this project's history has used, and it is
named in advance rather than chosen after a number.

THE JOIN IS BY STUDY, NOT BY POSITION. Both frames are built from the same `test.csv` and today come out
in the same order; a positional blend would silently pair every study with another study's scores the
day one producer sorted. This is the row-wise twin of the label-order join in `RaptorMember.ordered`,
and it fails the same way: quietly, near chance, with every other check green.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.findings import STUDY, TARGETS
from kneemri.inference.vote import Vote

logger = logging.getLogger(__name__)

SEATS = 7
"""Six of ours plus theirs. The weight is one seat's share, not a tuned coefficient."""


@dataclass(frozen=True)
class RaptorSeat:
    """One weighted rank blend of our roster's table with the raptor member's."""

    weight: float = 1.0 / SEATS

    def blended(self, ours: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame:
        """The two tables joined by study and averaged in rank space, `weight` on the raptor side."""
        if not 0.0 <= self.weight <= 1.0:
            message = f"a seat's share is a fraction; {self.weight} is not one"
            raise ValueError(message)
        aligned = theirs.set_index(STUDY).reindex(ours[STUDY])
        missing = int(aligned[list(TARGETS)].isna().any(axis=1).sum())
        if missing:
            message = f"{missing} of {len(ours)} studies are absent from the raptor table"
            raise ValueError(message)

        mine = Vote.ranks(ours[list(TARGETS)].to_numpy(dtype=np.float64))
        yours = Vote.ranks(aligned[list(TARGETS)].to_numpy(dtype=np.float64))
        combined = (1.0 - self.weight) * mine + self.weight * yours

        frame = pd.DataFrame(combined, columns=list(TARGETS))
        frame.insert(0, STUDY, ours[STUDY].to_numpy())
        logger.info("seated the raptor member at %.4f over %d studies", self.weight, len(frame))
        return frame
