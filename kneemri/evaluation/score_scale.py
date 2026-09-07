"""The scale a ratio of two bounded scores is taken ON, because a ratio is not scale-free.

Every compression number this project owns divides a board delta by a holdout delta in raw AUC. Near
1.0 that is a compressed axis by construction: the same amount of separating power buys less AUC the
higher you already are, so a slope fitted across a record whose top entries are bunched at 0.944 will
FALL at the top whether or not anything saturated. A regime reading reports that fall and cannot tell
the two apart on its own, and on 2026-09-07 the whole 0.953 target turned on which it was.

A monotone re-scaling leaves every ORDERING untouched, which is the property that makes two readings
comparable: the pairs, the resolutions and the inversions are identical under any scale here, and only
the ratios move. So if a fall survives probit it is a fact about transfer, and if it flattens it was
the ruler being read on the wrong scale. Measured that day: the fall was 27% raw and 23% probit, so it
survived — but the CLIMB the same record prices came out at +0.0092 to +0.0126 rather than the +0.0206
an OLS slope through a 0.006-wide tercile had produced, which is a fact about estimators and not about
scales at all. Both readings needed this module to be sayable in one command.

This is a separate module from `board_transfer` because it is a separate SUBJECT: nothing here knows
what a leaderboard is, and a second consumer that wants to price a curve against a bounded metric
takes it without taking the record reader.
"""

import logging
from dataclasses import dataclass
from statistics import NormalDist
from typing import Protocol

logger = logging.getLogger(__name__)

PROBIT_EPS = 1e-6
"""How far off 0 and 1 a score is held before its quantile is taken, since `inv_cdf` diverges at both.

No recorded board or holdout AUC is anywhere near either end, so this clamp has never fired on that
record. It is here because a column-lift sweep DOES saturate a column at exactly 1.0000, and the day
one of those numbers reaches a scale it would be an infinity inside a median rather than an error
anybody sees.
"""


class Scale(Protocol):
    """A monotone re-expression of a bounded score, and the two operations a ratio actually needs."""

    def of(self, value: float) -> float:
        """The score as this scale expresses it."""
        ...

    def back(self, value: float) -> float:
        """The score this scale expresses as `value`, so a requirement can be quoted in AUC again."""
        ...

    def delta(self, better: float, worse: float) -> float:
        """The gap between two scores ON THIS SCALE, which is the quantity a ratio divides."""
        ...

    def advance(self, value: float, climb: float) -> float:
        """The AUC reached by climbing `climb` scale-points from `value`."""
        ...


@dataclass(frozen=True)
class RawScale:
    """AUC as recorded. The scale every transfer constant this project quotes was fitted on."""

    def of(self, value: float) -> float:
        """The score unchanged."""
        return value

    def back(self, value: float) -> float:
        """The score unchanged."""
        return value

    def delta(self, better: float, worse: float) -> float:
        """The plain difference."""
        return better - worse

    def advance(self, value: float, climb: float) -> float:
        """The plain sum."""
        return value + climb


@dataclass(frozen=True)
class ProbitScale:
    """AUC through the normal quantile, on which a bounded rate stops compressing near its edges.

    `NormalDist().inv_cdf` rather than scipy: scipy is not a declared dependency of this project, and
    importing a transitive one to take a quantile is how DEP003 gets suppressed instead of fixed.
    """

    def of(self, value: float) -> float:
        """The score's normal quantile, clamped off the open interval's ends so a 1.0 cannot diverge."""
        return NormalDist().inv_cdf(min(max(value, PROBIT_EPS), 1.0 - PROBIT_EPS))

    def back(self, value: float) -> float:
        """The AUC whose quantile is `value`."""
        return NormalDist().cdf(value)

    def delta(self, better: float, worse: float) -> float:
        """The difference of the two quantiles."""
        return self.of(better) - self.of(worse)

    def advance(self, value: float, climb: float) -> float:
        """The AUC whose quantile sits `climb` above this one's."""
        return self.back(self.of(value) + climb)


SCALES: dict[str, RawScale | ProbitScale] = {"raw": RawScale(), "probit": ProbitScale()}
"""The closed vocabulary a `--scale` flag reads, so an unknown name refuses instead of reading raw."""

RAW = SCALES["raw"]
"""The default every reader gets, named once so it is never constructed in an argument default."""
