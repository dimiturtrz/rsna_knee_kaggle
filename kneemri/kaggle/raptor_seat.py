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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.findings import STUDY, TARGETS
from kneemri.inference.vote import Vote

logger = logging.getLogger(__name__)

OUR_SEATS = 6
"""The champion roster's own member count, which is what a foreign seat is a share OF."""

SEATS = OUR_SEATS + 1
"""Six of ours plus theirs. The weight is one seat's share, not a tuned coefficient."""


@dataclass(frozen=True)
class RaptorSeat:
    """One weighted rank blend of our roster's table with one or more foreign members'."""

    weight: float = 1.0 / SEATS

    def blended(self, ours: pd.DataFrame, theirs: pd.DataFrame) -> pd.DataFrame:
        """The two tables joined by study and averaged in rank space, `weight` on the raptor side."""
        if not 0.0 <= self.weight <= 1.0:
            message = f"a seat's share is a fraction; {self.weight} is not one"
            raise ValueError(message)
        return self.combined(ours, [theirs], [self.weight])

    @staticmethod
    def tilted(ours: pd.DataFrame, theirs: pd.DataFrame, weights: Mapping[str, float]) -> pd.DataFrame:
        """Each finding blended at its OWN foreign weight: column `f` is `RaptorSeat(weights[f]).blended(...)[f]`."""
        if set(weights) != set(TARGETS):
            message = f"weights name {sorted(weights)}, the board scores {sorted(TARGETS)}"
            raise ValueError(message)
        blends = {weight: RaptorSeat(weight=weight).blended(ours, theirs) for weight in set(weights.values())}
        frame = ours[[STUDY]].reset_index(drop=True)
        for finding in TARGETS:
            frame[finding] = blends[weights[finding]][finding]
        return frame

    @staticmethod
    def fleet(ours: pd.DataFrame, theirs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Several foreign members seated at once, each taking ONE seat of the widened roster.

        The share is `1 / (OUR_SEATS + len(theirs))` per foreign member rather than a fitted vector, for
        the same reason the single seat is 1/7: their arms are `--fold -1`, so every study of our holdout
        sits inside their training set and no offline pairing exists to fit a weight on. Fitting one on
        the board costs a slot per value; equal seats cost none and are named before the number.

        It is the count that changes, not the rule — three foreign members are three seats, not one seat
        split three ways, because the roster they join is six members wide either way.
        """
        share = 1.0 / (OUR_SEATS + len(theirs))
        logger.info("seating %d foreign members at %.4f each", len(theirs), share)
        return RaptorSeat.combined(ours, list(theirs.values()), [share] * len(theirs))

    @staticmethod
    def combined(ours: pd.DataFrame, theirs: Sequence[pd.DataFrame], weights: Sequence[float]) -> pd.DataFrame:
        """Our table and every foreign table joined BY STUDY and averaged in rank space.

        Written once for both the single seat and the fleet: a second copy of the join is exactly the
        kind of silent, near-chance pairing bug the module docstring names.
        """
        share = sum(weights)
        if not 0.0 <= share <= 1.0:
            message = f"the foreign seats take {share} of the blend, which is not a fraction"
            raise ValueError(message)

        combined = (1.0 - share) * Vote.ranks(ours[list(TARGETS)].to_numpy(dtype=np.float64))
        for table, weight in zip(theirs, weights, strict=True):
            aligned = table.set_index(STUDY).reindex(ours[STUDY])
            missing = int(aligned[list(TARGETS)].isna().any(axis=1).sum())
            if missing:
                message = f"{missing} of {len(ours)} studies are absent from a raptor table"
                raise ValueError(message)
            combined += weight * Vote.ranks(aligned[list(TARGETS)].to_numpy(dtype=np.float64))

        frame = pd.DataFrame(combined, columns=list(TARGETS))
        frame.insert(0, STUDY, ours[STUDY].to_numpy())
        logger.info("seated %d foreign members over %d studies", len(theirs), len(frame))
        return frame
