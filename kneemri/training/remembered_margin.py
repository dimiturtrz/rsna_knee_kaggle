"""The AUC-margin surrogate over pairs that span steps, rather than pairs inside one batch."""

from typing import override

import torch
from jaxtyping import Float
from torch import Tensor

from kneemri.training.grades import Predicted
from kneemri.training.objective import (
    AUC_MARGIN,
    PAIR_MEMORY,
    PAIR_POOL,
    MarginObjective,
    Shaping,
)
from kneemri.training.pair_memory import PairMemory


class RememberedMargin(MarginObjective):
    """The AUC-margin surrogate, widened by a remembered pool of past readouts.

    THE MISMATCH THIS EXISTS FOR. `MarginObjective` optimises a per-label spread plus
    shortfall between the positive and negative class means. With a batch of 12 and
    Fracture positive in ~7% of studies, one or both classes are empty in roughly half
    the steps, so the rarest labels get LESS gradient than common ones. That biases the
    panel against the very labels the margin mechanism is meant to help.

    Remembering past readouts breaks that link. The pool spans the last `capacity`
    steps, so a 7% finding is expected in several of them; the gradient still flows only
    through the live step's logits, and the remembered side is a fixed target. The cost
    is the same one `RememberedRanking` pays: remembered logits were produced by weights
    that have since moved, which is what `PAIR_MEMORY` is chosen against.

    IT REPLACES CROSS-ENTROPY, just as `MarginObjective` does. The only change is the
    population over which the class means and spread are computed.
    """

    def __init__(
        self,
        margin: float = AUC_MARGIN,
        shaping: Shaping | None = None,
        *,
        capacity: int = PAIR_MEMORY,
    ) -> None:
        super().__init__(margin, shaping)
        self.memory = PairMemory(capacity)

    @override
    def settings(self) -> dict[str, object]:
        """The margin account with the pool the memory really holds."""
        return {**super().settings(), PAIR_POOL: self.memory.capacity()}

    @override
    def auc_margin(self, batch: Predicted) -> Float[Tensor, ""]:
        """Spread plus shortfall over the live batch and the remembered pool."""
        remembered = self.memory.pooled()
        pooled = batch if remembered is None else self.widened(batch, remembered)
        loss = super().auc_margin(pooled)
        self.memory.remember(batch)
        return loss

    def widened(self, batch: Predicted, remembered: Predicted) -> Predicted:
        """The live batch followed by the pooled memory, as one readout for the margin.

        Margin is a function of logits, scores and weights only; teacher and embedding
        are left at None because this term never reads them.
        """
        return Predicted(
            logits=torch.cat([batch.logits, remembered.logits]),
            scores=torch.cat([batch.scores, remembered.scores]),
            weights=torch.cat([batch.weights, remembered.weights]),
        )
