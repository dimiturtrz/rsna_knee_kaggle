"""How confident test predictions become pseudo-labels.

Split out of `transduction`, which measures what these schemes buy. The 2026-09-03 negative on that
axis named the labelling scheme as its own leading suspect, so the scheme became a strategy that can be
swapped and the suspect became testable. It was then refuted — see that module's docstring for the
forty cells.
"""

from abc import ABC, abstractmethod
from typing import ClassVar, override

import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray


class Labelling(ABC):
    """How the confident test predictions become labels. The suspect the 2026-09-03 negative named."""

    name: ClassVar[str]

    @abstractmethod
    def assigned(
        self,
        predictions: Float[NDArray[np.float64], "held findings"],
        share: float,
        prevalence: Float[NDArray[np.float64], " findings"],
    ) -> tuple[Int[NDArray[np.int64], "held findings"], Float[NDArray[np.float64], "held findings"]]:
        """Labels and a 0/1 mask of which studies count, per finding."""


class FixedTail(Labelling):
    """Exactly `share` of the test set positive per finding, whatever that finding's prevalence is.

    The original scheme, kept so the measured negative stays reproducible. Its defect is that TAIL SIZE
    and IMPLIED PREVALENCE are one knob: at share 0.25 a finding present in 5% of studies is handed
    twenty percentage points of invented positives, and the refit is then fitting that invention.
    """

    name = "fixed"

    @override
    def assigned(
        self,
        predictions: Float[NDArray[np.float64], "held findings"],
        share: float,
        prevalence: Float[NDArray[np.float64], " findings"],
    ) -> tuple[Int[NDArray[np.int64], "held findings"], Float[NDArray[np.float64], "held findings"]]:
        del prevalence
        labels = np.zeros_like(predictions, dtype=np.int64)
        weights = np.zeros_like(predictions, dtype=np.float64)
        tail = max(1, int(len(predictions) * share))
        for column in range(predictions.shape[1]):
            order = np.argsort(predictions[:, column])
            labels[order[-tail:], column] = 1
            weights[order[:tail], column] = 1.0
            weights[order[-tail:], column] = 1.0
        return labels, weights


class PrevalenceTail(Labelling):
    """The positive tail is capped by the finding's own training prevalence; `share` stays confidence.

    Two knobs instead of one. A finding present in 5% of the training rows can never be handed more than
    5% positives here, so a large tail buys more NEGATIVE evidence rather than more invented disease.
    The negative tail is untouched, because a study in the bottom `share` is confidently negative for
    every prevalence a finding can have.
    """

    name = "prevalence"

    @override
    def assigned(
        self,
        predictions: Float[NDArray[np.float64], "held findings"],
        share: float,
        prevalence: Float[NDArray[np.float64], " findings"],
    ) -> tuple[Int[NDArray[np.int64], "held findings"], Float[NDArray[np.float64], "held findings"]]:
        labels = np.zeros_like(predictions, dtype=np.int64)
        weights = np.zeros_like(predictions, dtype=np.float64)
        held = len(predictions)
        tail = max(1, int(held * share))
        for column in range(predictions.shape[1]):
            order = np.argsort(predictions[:, column])
            positives = max(1, min(tail, int(held * prevalence[column])))
            labels[order[-positives:], column] = 1
            weights[order[:tail], column] = 1.0
            weights[order[-positives:], column] = 1.0
        return labels, weights


FIXED_TAIL = FixedTail()
"""The original scheme as a singleton, so the default argument is a value rather than a call."""

LABELLINGS: dict[str, Labelling] = {scheme.name: scheme for scheme in (FIXED_TAIL, PrevalenceTail())}
"""Declaration-only dispatch: the caller names a scheme, never branches on one."""
