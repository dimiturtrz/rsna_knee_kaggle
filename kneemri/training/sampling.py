"""How an epoch's studies are DRAWN, which every arm on the ledger has so far left uniform.

`Trainer.batches` permutes the training positions and cuts them into batches, so every study is seen
exactly once per epoch and the only thing that varies is order. That is a policy, not a law, and it is
the one part of the recipe no arm has moved.

WHY THERE IS ANYTHING TO MOVE. `residual_split` put ~70% of the pool's errors in a band that more than
half but not all of the members get wrong, against 24.3% that all of them get wrong and ~5.4% a combiner
could recover. The unanimous block is out of reach of any draw — a study every member reads the same way
is not one the family is undecided about — but the middle band is exactly the set of studies the family
still disagrees on, and a draw that spends more of the epoch there is the only lever aimed at it.

WHY THE WEIGHT IS DISAGREEMENT AND NOT LOSS. See `kneemri.evaluation.train_disagreement`, which produces
it: ~87% of the residual's mass is aleatoric, so ranking by the model's own loss ranks label noise to the
top and mines it harder. Spread across separately-trained members is high where the family is undecided
and low where it agrees, including where it agrees on being WRONG.

THE DRAW IS WITH REPLACEMENT, AND THE EPOCH KEEPS ITS LENGTH. A weighted draw of the same size costs the
same compute as the uniform one it replaces, which is what makes the control matched: the two arms differ
in which studies the gradient saw and in nothing else. The cost is that a study can appear twice in one
epoch and another not at all, and over twelve epochs at alpha 0.5 the tails of that are what the arm is
measuring.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import override

import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class Sampling(ABC):
    """The policy that decides which training positions an epoch descends on."""

    @abstractmethod
    def drawn(
        self, positions: Int[NDArray[np.int64], " studies"], generator: np.random.Generator
    ) -> Int[NDArray[np.int64], " studies"]:
        """This epoch's positions, the same number of them as were offered."""

    @abstractmethod
    def tilt(self) -> float:
        """How much of the draw the weight controls; zero is the uniform policy.

        The positive witness the flag would otherwise lack. A spread file that failed to load, or a
        policy that was built and never attached, trains perfectly well and reports the uniform arm's
        behaviour under the sampled arm's name — the exact failure `landed` exists to catch.
        """

    @abstractmethod
    def described(self) -> str:
        """What this policy is, for the run's own record."""


class Uniform(Sampling):
    """Every training study exactly once per epoch — the policy every recorded arm ran under."""

    @override
    def drawn(
        self, positions: Int[NDArray[np.int64], " studies"], generator: np.random.Generator
    ) -> Int[NDArray[np.int64], " studies"]:
        """The positions unchanged; `Trainer.batches` permutes them straight after.

        The generator is untouched here ON PURPOSE. Drawing from it would advance the run's random
        stream and make a uniform arm under this class a different run from every arm already in the
        ledger, which would cost the whole corpus as a control.
        """
        return positions

    @override
    def tilt(self) -> float:
        """Nothing; the draw is uniform."""
        return 0.0

    @override
    def described(self) -> str:
        """The null policy."""
        return "uniform (each study once per epoch)"


class Disagreed(Sampling):
    """Studies drawn in proportion to how much the reference members disagreed about them.

    AT ALPHA ZERO THIS IS NOT THE UNIFORM POLICY, and the difference is the whole reason the arm needs
    its own control rather than the ledger's. The weights go flat, but the draw is still WITH
    REPLACEMENT, so an epoch still holds duplicates and still misses about a third of the pool — a
    bootstrap resample of the training split. `Uniform` sees every study exactly once. Comparing a
    tilted draw against `Uniform` would therefore measure the tilt and the resampling together;
    comparing it against this class at alpha zero measures the tilt alone.
    """

    def __init__(self, spread: Float[NDArray[np.float64], " studies"], alpha: float) -> None:
        self.spread = spread
        self.alpha = alpha

    @classmethod
    def loaded(cls, weights: Path, identifiers: list[str], alpha: float) -> "Disagreed":
        """Read a `train_disagreement` file and align it to THIS corpus's study order.

        Aligning by identifier rather than by position is not defensive coding, it is the only correct
        join: the file records the studies of one fold in that fold's order, and a trainer whose corpus
        was cached differently would otherwise weight study A by study B's disagreement — silently, and
        with a perfectly plausible-looking loss curve.

        A study the file does not name is given the mean spread rather than zero, so an incomplete file
        degrades toward uniform instead of quietly dropping studies out of the epoch.
        """
        held = np.load(weights)
        names = [str(name) for name in held["studies"]]
        recorded = dict(zip(names, held["spread"].tolist(), strict=True))
        mean = float(np.mean(held["spread"]))
        missing = [name for name in identifiers if name not in recorded]
        if missing:
            logger.warning(
                "%d of %d studies are absent from %s; they draw at the mean",
                len(missing),
                len(identifiers),
                weights,
            )
        return cls(np.array([recorded.get(name, mean) for name in identifiers], dtype=np.float64), alpha)

    def weights(self, positions: Int[NDArray[np.int64], " studies"]) -> Float[NDArray[np.float64], " studies"]:
        """The draw probability of each offered position, normalised to sum to one."""
        spread = self.spread[positions]
        scaled = spread / spread.mean() if spread.mean() > 0.0 else np.ones_like(spread)
        mixed = (1.0 - self.alpha) + self.alpha * scaled
        return np.asarray(mixed / mixed.sum(), dtype=np.float64)

    @override
    def drawn(
        self, positions: Int[NDArray[np.int64], " studies"], generator: np.random.Generator
    ) -> Int[NDArray[np.int64], " studies"]:
        """The same number of positions, drawn with replacement against the disagreement weights."""
        return np.asarray(
            generator.choice(positions, size=len(positions), replace=True, p=self.weights(positions)),
            dtype=np.int64,
        )

    @override
    def tilt(self) -> float:
        """The share of the draw the disagreement weight controls."""
        return self.alpha

    @override
    def described(self) -> str:
        """This policy and its tilt."""
        return f"disagreement-weighted at alpha {self.alpha:.2f} over {len(self.spread)} studies"
