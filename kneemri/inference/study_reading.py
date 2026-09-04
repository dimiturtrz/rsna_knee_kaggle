"""The reading loop both foreign families are scored through — one member, one feed, one study a forward.

It lives here, and not beside either member, because it is the same loop for a 2D windowed member and for
a 3D volumetric one: what differs is the rank of the tensor, and neither the batching argument nor the
sigmoid depends on it. Two families reading through two copies of this is how one of them drifts and how a
comparison between them stops meaning anything.

The two contracts are declared INSIDE it because they are its parameter vocabulary and nothing else's: a
`Member` is whatever answers for one study, a `Feed` is whatever can hand it one.
"""

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

logger = logging.getLogger(__name__)

PROGRESS = 50
"""Studies between progress lines — a read of a thousand studies is minutes, and silence reads as a hang."""


@dataclass(frozen=True)
class StudyReadings:
    """One member's probabilities for a fixed set of our studies."""

    device: torch.device
    precision: torch.dtype = torch.float16

    @runtime_checkable
    class Member(Protocol):
        """A published member that answers for ONE study at a time, whatever shape that study arrives in."""

        def to(self, device: torch.device) -> "StudyReadings.Member": ...

        def __call__(self, batch: Float[Tensor, "..."]) -> Float[Tensor, "b n"]: ...

    @runtime_checkable
    class Feed(Protocol):
        """A cache that can hand one study to a `Member`, already on the device and already scaled."""

        def batch(self, row: int, device: torch.device) -> Float[Tensor, "..."]: ...

    def predict(
        self, member: "StudyReadings.Member", feed: "StudyReadings.Feed", rows: list[int]
    ) -> Float[np.ndarray, "studies targets"]:
        """Probabilities for each row of the cache, one study per forward.

        One at a time because window counts DIFFER — a study with a missing slot has fewer bags — and
        padding them to a common count would feed the attention pool blanks it is free to attend to. The
        forward is already a batch of sixty windows, so the study axis buys little and risks a number.
        """
        chunks = []
        with torch.no_grad(), torch.autocast("cuda", dtype=self.precision, enabled=self.device.type == "cuda"):
            for done, row in enumerate(rows):
                logits = member(feed.batch(row, self.device).unsqueeze(0))
                chunks.append(torch.sigmoid(logits.float()).cpu().numpy())
                if done % PROGRESS == 0:
                    logger.info("%d/%d studies", done, len(rows))
        return np.concatenate(chunks).astype(np.float64)
