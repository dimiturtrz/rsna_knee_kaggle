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

    def ramp(self) -> int:
        """How many epochs the tilt takes to arrive; zero is a tilt applied from the first step.

        Concrete rather than abstract because it is a MODIFIER on the tilt and not a policy of its own,
        so every policy that has no ramp answers the same zero. It is reported separately from `tilt`
        for the reason `tilt` exists at all: a ramp that failed to attach still reports the destination
        tilt, and the static arm and the curriculum arm would then be indistinguishable in the record.
        """
        return 0

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


class Ramped(Disagreed):
    """The same disagreement weights, but reached over the first `epochs` epochs instead of at once.

    THE CELL THIS FILLS. Difficulty has two signals here and two ways to spend one. `disagreed_full`
    tilts toward disagreement from the first step and scores 0.8471 against its own alpha-zero control
    at 0.8507 — the static tilt COSTS. `certainty_ramp` spends six epochs raising label trust rather
    than tilting the draw and scores 0.8584 — the SCHEDULE pays. Zhou et al. (ISBI 2023) report the
    same shape on elbow-fracture radiographs, +0.0245 binary AUC from an easy-to-hard curriculum with
    no human difficulty annotation. What no arm has run is the schedule applied to the disagreement
    signal, which is where the two difficulty measures disagree: certainty is what the labeller felt,
    disagreement is what separately-trained members cannot resolve.

    THE EPOCH COUNTER IS THIS OBJECT'S OWN. `fit_epoch` is handed no epoch index, and threading one
    through it would change the signature every recorded arm ran under. `drawn` is called exactly once
    per epoch by exactly one caller, so counting the calls IS counting the epochs — and the count is
    reported through `tilt`, which `landed` reads, so a policy that was built and never attached still
    cannot pass itself off as this one.
    """

    def __init__(self, spread: Float[NDArray[np.float64], " studies"], alpha: float, epochs: int) -> None:
        super().__init__(spread, 0.0)
        self.target = alpha
        self.epochs = epochs
        self.epoch = 0

    @classmethod
    @override
    def loaded(cls, weights: Path, identifiers: list[str], alpha: float, epochs: int = 0) -> "Ramped":
        """A `Disagreed` read off disk, re-wrapped so its tilt arrives over `epochs` epochs."""
        flat = Disagreed.loaded(weights, identifiers, alpha)
        return cls(flat.spread, alpha, epochs)

    @override
    def drawn(
        self, positions: Int[NDArray[np.int64], " studies"], generator: np.random.Generator
    ) -> Int[NDArray[np.int64], " studies"]:
        """This epoch's draw, at this epoch's alpha, advancing the ramp by one epoch."""
        self.alpha = self.target if self.epoch >= self.epochs else self.target * (self.epoch / self.epochs)
        self.epoch += 1
        return super().drawn(positions, generator)

    @override
    def tilt(self) -> float:
        """The tilt this policy is FOR, not the one it happens to hold at epoch zero.

        `landed` reads this before the first epoch, when the ramp is still at alpha zero and
        indistinguishable from the control. Reporting the destination is what makes the switch
        checkable; the current value belongs in `described`, which is prose.
        """
        return self.target

    @override
    def ramp(self) -> int:
        """The epochs this policy takes to reach its tilt — the one thing `tilt` cannot say."""
        return self.epochs

    @override
    def described(self) -> str:
        """This policy, its destination and how long it takes to get there."""
        return (
            f"disagreement-weighted ramping to alpha {self.target:.2f} over {self.epochs} epochs "
            f"(currently {self.alpha:.2f}) over {len(self.spread)} studies"
        )


class MixedCorpus(Sampling):
    """Oversample non-target corpora so external studies are not drowned by the target domain.

    Scheme B wants roughly half the batch to come from external corpora despite RSNA dominating
    study count. A pure count draw would make most batches almost entirely RSNA; this sampler
    up-weights every non-target position by `factor` and draws with replacement, keeping the
    epoch length unchanged.

    The corpus tag is read from the labels table's `corpus` column rather than guessed from the
    identifier, because MRNet prefixes its IDs, Rijeka uses filename stems, and fastMRI uses file
    names — identifier shape is not a reliable corpus signal.
    """

    def __init__(
        self,
        corpus: NDArray[np.str_],
        target: str,
        factor: float,
    ) -> None:
        self.corpus = np.asarray(corpus)
        self.target = target
        self.factor = factor
        self._weights = np.where(self.corpus == target, 1.0, float(factor)).astype(np.float64)

    @override
    def drawn(
        self, positions: Int[NDArray[np.int64], " studies"], generator: np.random.Generator
    ) -> Int[NDArray[np.int64], " studies"]:
        """The same number of positions, drawn with replacement against corpus weights."""
        weights = self._weights[positions]
        weights = weights / weights.sum()
        return np.asarray(
            generator.choice(positions, size=len(positions), replace=True, p=weights),
            dtype=np.int64,
        )

    @override
    def tilt(self) -> float:
        """How many times heavier a non-target position is than a target one."""
        return self.factor

    @override
    def described(self) -> str:
        """This policy and its factor."""
        return (
            f"mixed-corpus oversampling: {self.target} at 1.0, others at {self.factor:.2f} "
            f"over {len(self.corpus)} studies"
        )
