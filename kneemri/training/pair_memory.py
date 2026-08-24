"""A few steps' worth of readouts, kept so a ranking pair can span batches."""

from collections import deque

import torch

from kneemri.training.grades import Predicted
from kneemri.training.objective import PAIR_MEMORY


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
        taught = [step.teacher for step in self.remembered if step.teacher is not None]
        return Predicted(
            logits=torch.cat([step.logits for step in self.remembered]),
            scores=torch.cat([step.scores for step in self.remembered]),
            weights=torch.cat([step.weights for step in self.remembered]),
            # ALL OR NOTHING, because a partial pool is worse than no pool: concatenating only the steps
            # that carried a teacher would silently align those rows against the wrong studies. The ring
            # is homogeneous in practice — a run either distils or does not — so this is a guard rather
            # than a case, and the `None` it falls back to is what a distilling objective raises on.
            teacher=torch.cat(taught) if len(taught) == len(self.remembered) else None,
        )

    def remember(self, batch: Predicted) -> None:
        """Keep this step's readout for the next few, dropping the oldest once the ring is full."""
        self.remembered.append(
            Predicted(
                logits=batch.logits.detach(),
                scores=batch.scores.detach(),
                weights=batch.weights.detach(),
                teacher=None if batch.teacher is None else batch.teacher.detach(),
            )
        )
