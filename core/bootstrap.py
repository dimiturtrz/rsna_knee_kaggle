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

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from jaxtyping import Float
from numpy.typing import NDArray

RESAMPLES = 1500
"""MEASURED, after the first version asserted 300 was "stable to about a thousandth" and was wrong.

Across three resample seeds the Lateral Meniscus lower bound moved by 0.0063 at 300 — enough to flip
whether it cleared zero, which is a verdict decided by the random seed. At 1500 the same spread is 0.0015
and the sign is stable. An interval whose own Monte Carlo error is larger than the effect is not an
interval.

The measurement was made on `stack_coverage`'s (retired, last at e0611cc5) AUC delta, which is the
widest of the quantities that quote this number, so the three narrower ones inherit a bound rather than
an assumption.
"""

RESAMPLE_SEED = 11
"""Fixed, so an interval quoted in an interpretation is the interval anyone else recomputes."""

INTERVAL = (2.5, 97.5)
"""The percentile pair a 95% interval is read at — spelled once so no report quietly widens its own."""

STUDY_COLUMN = "study"
"""The cluster a frame bootstrap draws whole: cells inside one study share a report and a reader."""


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

    @staticmethod
    def percentile_interval(
        values: list[float] | Float[NDArray[np.float64], " draws"], pair: tuple[float, float] = INTERVAL
    ) -> tuple[float, float]:
        """The two-sided bound of a finished list of draws, NaN when every draw was undefined.

        NaN rather than a raise: a label undefined on this split is a fact worth printing beside the ones
        that are defined, and an exception would take the whole report down with it.
        """
        if len(values) == 0:
            return (float("nan"), float("nan"))
        low, high = np.percentile(values, pair)
        return (float(low), float(high))

    def mean_interval(self, values: Float[np.ndarray, " n"]) -> tuple[float, float]:
        """A bootstrap interval on the mean of `values`, so a share is never a bare point estimate."""
        return self.percentile_interval(
            [float(self.generator.choice(values, values.size).mean()) for _ in range(RESAMPLES)]
        )


STUDY_RESAMPLES = 1000
"""Draws for the index bootstrap, kept at what every published interval here was read under.

NOT the measured 1500 above, and the difference is not an oversight. `RESAMPLES` was raised after a
measurement on `stack_coverage`'s (retired, last at e0611cc5) AUC delta; this count has had no such
measurement, and lowering or raising it would move the 0.0062 paired floor, the +/-0.0106 absolute
band and every per-label interval already quoted in the ledger and the interpretations. The number is
frozen until something re-reads those; the case for measuring it is knee-voa6.
"""

STUDY_SEED = 2026
"""Fixed, so an interval quoted in a bead is the interval anyone else recomputes."""


class StudyBootstrap:
    """The 95% interval of anything measured on a resampled corpus, over draws of whole STUDIES.

    The sibling of `Bootstrap`, and separate from it for a reason that is not stylistic. `Bootstrap`
    resamples finished VALUES off one shared stream; this resamples INDICES and hands them back so the
    caller can re-measure from scratch. Two callers need exactly that and cannot be served by the other
    form: a paired interval scores both arms on the drawn studies, and `AlignmentDamage` re-estimates
    its median split inside each draw so the boundary's own error reaches the interval.

    Studies and not labels: the twelve per-label AUCs are twelve readings of ONE corpus, so resampling
    labels would count a single sample twelve times.

    DRAW ORDER IS PART OF THE CONTRACT. One generator per call, `resamples` draws from it in sequence —
    the shape all three hand-written copies had, so every interval this project has published reproduces
    unchanged. That is also why the stream is per-call here while `Bootstrap` shares one: these callers
    each quote a single quantity, and matching the old numbers matters more than correlating draws.
    """

    def __init__(self, resamples: int = STUDY_RESAMPLES, seed: int = STUDY_SEED) -> None:
        self.resamples = resamples
        self.seed = seed

    def draws(self, studies: int) -> Iterator[NDArray[np.int64]]:
        """Index arrays, one per resample, each `studies` long and drawn with replacement."""
        generator = np.random.default_rng(self.seed)
        for _ in range(self.resamples):
            yield generator.integers(0, studies, size=studies)

    def interval(self, studies: int, measure: Callable[[NDArray[np.int64]], float]) -> tuple[float, float]:
        """The interval of a single number — one arm's mean, or one differential."""
        return Bootstrap.percentile_interval(
            [value for value in map(measure, self.draws(studies)) if not np.isnan(value)]
        )

    def per_label(
        self, studies: int, measure: Callable[[NDArray[np.int64]], dict[str, float]]
    ) -> dict[str, tuple[float, float]]:
        """The interval of several numbers measured together on the SAME draws.

        Together and not one label at a time: a per-label delta and its neighbours share the studies that
        produced them, and scoring each label on its own draws would break the pairing that makes the
        paired interval tighter than the unpaired one in the first place.
        """
        collected: dict[str, list[float]] = {}
        for drawn in self.draws(studies):
            for label, value in measure(drawn).items():
                # The key is recorded even when the value is not: a label undefined on EVERY draw still
                # belongs in the result at (nan, nan), because callers index this dict by label and a
                # missing key would raise where the hand-written copies reported "undefined".
                drawn_values = collected.setdefault(label, [])
                if not np.isnan(value):
                    drawn_values.append(value)
        return {label: Bootstrap.percentile_interval(values) for label, values in collected.items()}


@dataclass(frozen=True)
class StudyFrameBootstrap:
    """The interval of a statistic re-measured on a cell FRAME rebuilt from whole resampled studies.

    The frame-shaped sibling of `StudyBootstrap`: the callers hold cells in pandas and re-pool them per
    draw, so the draw is over study NAMES with `choice` rather than over row indices with `integers`. The
    two forms consume the generator differently, which is why they are not one class — merging them would
    move every interval either has published. A `measure` returning None marks a draw on which the
    statistic is undefined; it is skipped, and a run with no defined draw reads (nan, nan).
    """

    resamples: int
    seed: int = STUDY_SEED

    def interval(self, cells: pd.DataFrame, measure: Callable[[pd.DataFrame], float | None]) -> tuple[float, float]:
        """Draw the studies with replacement, rebuild their cells, measure, and read the two-sided bound."""
        studies = cells[STUDY_COLUMN].unique()
        by_study = dict(iter(cells.groupby(STUDY_COLUMN)))
        generator = np.random.default_rng(self.seed)
        drawn = []
        for _ in range(self.resamples):
            picked = generator.choice(studies, size=len(studies), replace=True)
            value = measure(pd.concat([by_study[study] for study in picked]))
            if value is not None:
                drawn.append(value)
        return Bootstrap.percentile_interval(drawn)
