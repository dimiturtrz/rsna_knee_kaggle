"""The reading loop both foreign families are scored through — one member, one feed, one study a forward.

It lives here, and not beside either member, because it is the same loop for a 2D windowed member and for
a 3D volumetric one: what differs is the rank of the tensor, and neither the batching argument nor the
sigmoid depends on it. Two families reading through two copies of this is how one of them drifts and how a
comparison between them stops meaning anything.

The three contracts are declared INSIDE it because they are its parameter vocabulary and nothing else's:
a `Member` is whatever answers for one study, a `Feed` is whatever can hand it one, and an `Adaptation`
is whatever may tune the member to that study first and must put it back afterwards.
"""

import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from kneemri.inference.norm_adaptation import NoAdaptation

logger = logging.getLogger(__name__)

PROGRESS = 50
"""Studies between progress lines — a read of a thousand studies is minutes, and silence reads as a hang."""


@dataclass(frozen=True)
class StudyReadings:
    """One member's probabilities for a fixed set of our studies."""

    device: torch.device
    precision: torch.dtype = torch.float16
    adaptation: "StudyReadings.Adaptation" = field(default_factory=NoAdaptation)

    @runtime_checkable
    class Member(Protocol):
        """A published member that answers for ONE study at a time, whatever shape that study arrives in."""

        def to(self, device: torch.device) -> "StudyReadings.Member": ...

        def __call__(self, batch: Float[Tensor, "..."]) -> Float[Tensor, "b n"]: ...

    @runtime_checkable
    class Feed(Protocol):
        """A cache that can hand one study to a `Member`, already on the device and already scaled."""

        def batch(self, row: int, device: torch.device) -> Float[Tensor, "..."]: ...

    @runtime_checkable
    class Adaptation(Protocol):
        """Whatever may tune a member to one study before it is read, and must put it back afterwards.

        A CONTEXT MANAGER RATHER THAN A CALL, because the undo is the load-bearing half. An adapt pass
        that returns leaves the member changed for every study after it, which turns an episodic recipe
        into an online one silently — and online is the cell CARVE measures as harmful on a multi-label
        head. Restoring in a `finally` makes the reset unskippable rather than remembered.
        """

        def on(self, member: object, batch: Float[Tensor, "..."]) -> AbstractContextManager[None]: ...

    def predict(
        self, member: "StudyReadings.Member", feed: "StudyReadings.Feed", rows: list[int]
    ) -> Float[np.ndarray, "studies targets"]:
        """Probabilities for each row of the cache, one study per forward.

        One at a time because window counts DIFFER — a study with a missing slot has fewer bags — and
        padding them to a common count would feed the attention pool blanks it is free to attend to. The
        forward is already a batch of sixty windows, so the study axis buys little and risks a number.

        THE ADAPT PASS IS OUTSIDE THE `no_grad`, and that is the whole reason this loop was restructured:
        adaptation needs a gradient and the read must not have one. The default adaptation does nothing,
        so a member that names none is served by exactly the code that produced every number on the
        board.
        """
        chunks = []
        for done, row in enumerate(rows):
            batch = feed.batch(row, self.device).unsqueeze(0)
            with self.adaptation.on(member, batch):
                chunks.append(self._read(member, batch))
            if done % PROGRESS == 0:
                logger.info("%d/%d studies", done, len(rows))
        return np.concatenate(chunks).astype(np.float64)

    def _read(self, member: "StudyReadings.Member", batch: Float[Tensor, "..."]) -> Float[np.ndarray, "1 targets"]:
        """The forward that every published figure came from, unchanged and still ungraded."""
        with torch.no_grad(), torch.autocast("cuda", dtype=self.precision, enabled=self.device.type == "cuda"):
            return torch.sigmoid(member(batch).float()).cpu().numpy()
