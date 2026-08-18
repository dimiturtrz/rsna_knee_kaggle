"""The resampling protocol every corpus measurement quotes its interval under.

A share, a discarded fraction, an asymmetry — three modules measure a different quantity and then report
it the same way: bootstrap the mean of what was measured, quote the 2.5th and 97.5th percentiles. That
last part was written out three times, along with the draw count, the seed and the percentile pair, and
the ARGUMENT for the draw count lived in a fourth module none of the three imports. A constant copied
away from its reason is a constant the next reader is free to lower.

It sits in `core` because it is arithmetic over a sample and knows nothing about knees: `corpus` may not
import `evaluation`, and a statistics primitive parked in either would have been imported upward by the
other.
"""

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

RESAMPLES = 1500
"""MEASURED, after the first version asserted 300 was "stable to about a thousandth" and was wrong.

Across three resample seeds the Lateral Meniscus lower bound moved by 0.0063 at 300 — enough to flip
whether it cleared zero, which is a verdict decided by the random seed. At 1500 the same spread is 0.0015
and the sign is stable. An interval whose own Monte Carlo error is larger than the effect is not an
interval.

The measurement was made on `stack_coverage`'s AUC delta, which is the widest of the quantities that
quote this number, so the three narrower ones inherit a bound rather than an assumption.
"""

RESAMPLE_SEED = 11
"""Fixed, so an interval quoted in an interpretation is the interval anyone else recomputes."""

INTERVAL = (2.5, 97.5)
"""The percentile pair a 95% interval is read at — spelled once so no report quietly widens its own."""


@dataclass(frozen=True)
class Bootstrap:
    """One resampling stream, and the intervals read off it.

    The STREAM is the state, which is why this is an object rather than a function taking a seed. A
    report that quotes several quantities draws them all from one generator; a seed parameter would
    restart it per quantity, giving two measurements in the same report identical draws and intervals
    correlated by construction — and silently moving every interval the repo has already published.
    """

    generator: np.random.Generator

    @classmethod
    def seeded(cls, seed: int = RESAMPLE_SEED) -> "Bootstrap":
        """The protocol's own stream, so a caller states a seed only when it wants a different one."""
        return cls(np.random.default_rng(seed))

    def mean_interval(self, values: Float[np.ndarray, " n"]) -> tuple[float, float]:
        """A bootstrap interval on the mean of `values`, so a share is never a bare point estimate."""
        resampled = [float(self.generator.choice(values, values.size).mean()) for _ in range(RESAMPLES)]
        low, high = np.percentile(resampled, INTERVAL)
        return float(low), float(high)
