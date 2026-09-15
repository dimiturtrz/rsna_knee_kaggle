"""Two studies blended into one example, so a memorised study-to-label pair becomes actively wrong.

WHY THIS AXIS AND NOT ANOTHER EPOCH. `band_dense_long` ran 24 epochs as a single variable and peaked at
epoch 6 (0.8587), sliding to 0.8405 by epoch 23 while its training loss halved — a 0.0182 peak-to-finish
decline against a 0.0062 noise floor, which is memorisation and not a schedule wanting more room. At
MATCHED train loss ConvNeXtV2 reads 0.8371 against 0.7927, a 0.044 generalisation gap at equal fit. A
classifier that memorises at twelve epochs has no use for more slices, pixels, members or epochs, which
is why eleven measured axes all read inside the floor.

WHY THE AUGMENTER WE ALREADY HAVE DID NOT DO THIS. `Augmentation` perturbs how a study LOOKS — gain,
gamma, blur, contrast, noise, plus rigid jitter — and leaves the pair (this study, this label) intact.
The `aug_long` arm turned it on as a single variable and the ledger's own summary is that the regulariser
fired and the overfitting did not care. A convex combination is categorically different: the target moves
with the pixels, so the memorised pair is not merely harder to recognise, it is WRONG.

WHY IT COSTS NO PLUMBING. `objective.py` already scores weighted confidences in [0, 1] rather than binary
labels, so a mixed target needs no soft-label path — it is the same tensor with a different value in it.

WHAT IT DOES NOT REACH, said here rather than discovered later. The label term, the ranking term and the
teacher term all read blended rows; REPORT SUPERVISION does not. `ReportSupervision` is handed the batch's
positions rather than a table of rows, so under `--mixup` with `--report-weight` the report head is scored
against the first study's embedding while the label head is scored against a blend. Both flags are off by
default and no ledger arm sets them together — but an arm that does is confounded, and this is the sentence
that says so.
"""

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor

UNMIXED = 1.0
"""A blend weight of one is the identity: every row keeps its own pixels, its own mask and its own label."""

OFF = 0.0
"""Alpha zero is the null policy. The Beta draw is skipped entirely and the loop is the one every arm in
the ledger ran, which is what lets this land without re-baselining anything."""

WHITE = 255.0
"""The top of the uint8 range the cache stores. Blending happens in float and comes back to uint8 so the
model's own forward is untouched — the same bargain `Augmentation.bagged` already takes."""


@dataclass(frozen=True)
class Blend:
    """One draw: how much of each row survives, and which row it is mixed with.

    A DRAW IS A VALUE, not a branch in the training loop. The null policy returns an identity blend whose
    four methods are the identity, so `fit_epoch` calls the same three lines whether mixup is on or off —
    the alternative is three `if self.mixup` sites in the hottest loop we own, each a place for the flag
    to be honoured in two of them.
    """

    weight: float
    """The surviving share of each row's own content, folded to the dominant half of the Beta draw."""

    order: NDArray[np.int64]
    """Row `i` is mixed with row `order[i]`. A permutation of the batch, so the partner pool is the batch
    itself and no extra read is paid for."""

    def identity(self) -> bool:
        """Is this draw a no-op? True under the null policy, and true when the draw came back degenerate."""
        return self.weight >= UNMIXED

    def images(self, images: UInt8[Tensor, "b n g h w"]) -> UInt8[Tensor, "b n g h w"]:
        """The blended pixels, back in uint8.

        ROUNDED rather than truncated, because a plain cast biases every blended pixel downward by half a
        level — small per pixel and systematic across the corpus, which is the shape of a bug that reads as
        a slightly worse recipe rather than as a defect.
        """
        if self.identity():
            return images
        blended = images.float() * self.weight + images[self.order].float() * (UNMIXED - self.weight)
        return blended.round().clamp(0, WHITE).to(torch.uint8)

    def mask(self, mask: Float[Tensor, "b s"]) -> Float[Tensor, "b s"]:
        """Slot presence under the blend: the ELEMENTWISE MAXIMUM, not the weighted mean.

        A slot carrying pixels from either study carries pixels, so presence is a union. Averaging it
        instead would hand the model a half-present slot that no study in the corpus has, and would make
        the mask's meaning depend on the draw.
        """
        return mask if self.identity() else torch.maximum(mask, mask[self.order])

    def rows(self, rows: Float[NDArray[np.float32], "b o"]) -> Float[NDArray[np.float32], "b o"]:
        """A label or weight table blended with the SAME weight and the SAME order as the pixels.

        This is the whole mechanism. The pixels alone are an augmentation; the pixels and the target
        together are a statement that the memorised answer to this input is now the wrong one.
        """
        if self.identity():
            return rows
        return (rows * self.weight + rows[self.order] * (UNMIXED - self.weight)).astype(np.float32)


@dataclass
class Mixup:
    """The policy: a Beta concentration, drawn once per batch.

    PER BATCH, NOT PER ROW. One weight for the step keeps the blend a single scalar multiply over the
    whole tensor rather than a broadcast against a per-row vector, and it keeps the ranking term's pairs
    comparable — rows in a step are mixed to the same degree, so the pairwise loss is not silently
    comparing a barely-mixed study against a half-mixed one.
    """

    alpha: float = OFF
    """Zero is off. The customary range is 0.1 to 0.4 for a classifier this size: alpha 1.0 draws the
    weight uniformly and blends half the batch near 50/50, which on a corpus this small trades away more
    signal than it buys."""

    strength: float = UNMIXED
    """The progress schedule's current multiplier on `alpha`; 1.0 draws at `alpha` exactly."""

    def enabled(self) -> bool:
        """Is any blending going to happen? The witness asks this, not the trainer's loop."""
        return self.alpha > OFF

    def strengthen(self, strength: float) -> None:
        """Draw at this multiple of `alpha` from now on; zero is the identity blend with no draw."""
        self.strength = strength

    def draw(self, size: int, generator: np.random.Generator) -> Blend:
        """This step's blend, from the epoch's own generator so a seeded run stays reproducible."""
        alpha = self.alpha * self.strength
        if alpha <= OFF:
            return Blend(UNMIXED, np.arange(size, dtype=np.int64))
        drawn = float(generator.beta(alpha, alpha))
        # FOLDED to the dominant half. The blend is symmetric — weight w against a permutation is the same
        # family of examples as 1-w — so the fold costs no diversity and keeps every row recognisably its
        # own study, which is what makes the mixed target a correction rather than a coin flip.
        return Blend(max(drawn, UNMIXED - drawn), generator.permutation(size).astype(np.int64))
