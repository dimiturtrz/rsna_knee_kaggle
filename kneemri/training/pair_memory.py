"""A few steps' worth of readouts, kept so a ranking pair can span batches."""

from collections import deque

import torch

from kneemri.training.objective import PAIR_MEMORY, Predicted


class PairMemory:
    """The last few steps' readouts, detached and kept so a pair can span batches.

    Small enough to ignore: twelve studies of twelve labels is 144 floats a step, three tensors of
    them, against activations measured in gigabytes. It holds DETACHED copies, so nothing here keeps a
    graph alive past its step.
    """

    def __init__(self, capacity: int = PAIR_MEMORY) -> None:
        self.remembered: deque[Predicted] = deque(maxlen=capacity)

    def capacity(self) -> int:
        """How many steps this memory holds, asked of the deque rather than of a remembered argument.

        The number a run asked for is checked against THIS before the card is spent, so the authority
        has to be the container that would actually drop a readout, not a copy of the constructor's
        argument that agrees with the request by construction.
        """
        return self.remembered.maxlen or 0

    def pooled(self) -> Predicted | None:
        """Everything remembered as one readout, or None when nothing is — the first step of a run."""
        if not self.remembered:
            return None
        return Predicted(
            logits=torch.cat([step.logits for step in self.remembered]),
            scores=torch.cat([step.scores for step in self.remembered]),
            weights=torch.cat([step.weights for step in self.remembered]),
        )

    def remember(self, batch: Predicted) -> None:
        """Keep this step's readout for the next few, dropping the oldest once the ring is full."""
        self.remembered.append(
            Predicted(logits=batch.logits.detach(), scores=batch.scores.detach(), weights=batch.weights.detach())
        )
