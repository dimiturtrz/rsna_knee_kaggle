"""What the model said about each training study on its EARLIER visits — the target `EarlyLearning` reads.

Kept beside the trainer the way `TeacherTargets` is, and for the same reason: the term is a function of
one batch, and the history it is judged against is per STUDY, so it has to be looked up by the positions
the labels were looked up by. A per-step ring such as `Remembered` cannot hold it — it forgets a study
long before that study comes round again.

READ, THEN LEARNED. A batch is judged against the average as it stood before this step, and only then does
the step's own reading join it. That order is what makes a study's first visit contribute exactly nothing,
and it is what keeps a sharpness step, which runs the same forward twice, from counting one reading twice.
"""

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from torch import Tensor

from core.findings import TARGETS

EARLY_BETA = 0.7
"""How much of a study's past average survives each new visit — Liu et al.'s published value.

0.7 is what they ship for CIFAR-10 and for Clothing1M, the one real-noise benchmark in the paper, and it is
taken as published rather than tuned. At one visit per epoch it puts about 97% of the target's weight on
the last ten epochs, which is the span our runs peak inside."""


class EarlyTargets:
    """Two running averages per study and finding: the past `p`, and the past `1 - p`."""

    def __init__(self, studies: int, beta: float = EARLY_BETA) -> None:
        self.beta = beta
        self.positive = np.zeros((studies, len(TARGETS)), dtype=np.float32)
        self.negative = np.zeros((studies, len(TARGETS)), dtype=np.float32)

    def rows(self, positions: NDArray[np.int64]) -> Float[np.ndarray, "b o 2"]:
        """The averages for a batch of study positions, indexed the way the labels are."""
        return np.stack([self.positive[positions], self.negative[positions]], axis=-1)

    def learn(self, positions: NDArray[np.int64], logits: Float[Tensor, "b o"]) -> None:
        """Fold this step's readings into each study's average.

        Under mixup, row `i` is credited to `positions[i]`: the blend keeps the dominant half of its draw,
        so that study is always at least half of what the row shows.
        """
        probability = logits.detach().float().sigmoid().cpu().numpy()
        kept = self.beta
        self.positive[positions] = kept * self.positive[positions] + (1.0 - kept) * probability
        self.negative[positions] = kept * self.negative[positions] + (1.0 - kept) * (1.0 - probability)
