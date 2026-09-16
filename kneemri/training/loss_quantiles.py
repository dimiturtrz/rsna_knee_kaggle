"""The per-study losses an epoch collected, and the five quantiles the epoch record carries.

Pure numbers a fit happens to produce: no model, no optimiser, no device. It sat in `Trainer`, which
owns the optimiser, the schedule, the epoch loop and the checkpoint writer, and was the one block
there a test could exercise without building a `StudyModel` first.
"""

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

LOSS_QUANTILES = ("q25", "q50", "q75", "q90", "q95")
"""Which per-study loss quantiles the epoch record carries."""

LEVELS = (0.25, 0.50, 0.75, 0.90, 0.95)
"""The same five as fractions, in the order `LOSS_QUANTILES` names them."""


class EpochLosses:
    """One epoch's per-study losses, accumulated batch by batch and read out once at the end."""

    def __init__(self) -> None:
        self.batches: list[Float[NDArray[np.float64], " b"]] = []

    def collect(self, batch: Float[NDArray[np.float64], " b"]) -> None:
        """Keep one batch's per-study losses; the epoch's shape is read off the whole list."""
        self.batches.append(batch)

    def quantiles(self) -> dict[str, float]:
        """Quantiles of everything collected.

        NaN when no study contributed, so callers always get the same key shape. Computed on CPU after
        the epoch, because the list is small and this keeps the training loop on the device.
        """
        if not self.batches:
            return {name: float("nan") for name in LOSS_QUANTILES}
        values = np.concatenate(self.batches)
        if len(values) == 0:
            return {name: float("nan") for name in LOSS_QUANTILES}
        return {name: float(level) for name, level in zip(LOSS_QUANTILES, np.quantile(values, LEVELS), strict=True)}
