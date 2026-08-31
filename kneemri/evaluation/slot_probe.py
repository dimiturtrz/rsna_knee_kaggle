"""The head as a function of the presence mask alone, with the trunk's work already done.

Split from `attribution_faithfulness`, which is the scoring: this is the thing being scored — the object
captum perturbs. Keeping it apart also keeps the encoder-shaped concern (cached features, batch tiling)
out of the module whose subject is two metrics.
"""

import torch
from jaxtyping import Float
from torch import Tensor

from kneemri.models.slot_head import SlotHead


class SlotProbe:
    """The head as a function of the presence mask alone, with the trunk's work already done.

    A METRIC PERTURBS ITS INPUT AND ASKS THE MODEL AGAIN, hundreds of times per study. Handing it the
    whole `StudyModel` would re-run the encoder for every one of those, and the encoder cannot see the
    mask — it takes pixels. So the trunk runs once, and what the metrics perturb is exactly the thing
    the diagnostic claims to be varying.

    captum evaluates `n` perturbations by stacking them into one batch of `n * b` rows, so the cached
    features are tiled to match. Tiling rather than expanding, because the row order captum stacks in is
    perturbation-major and a broadcast would silently pair a study with another study's features.
    """

    def __init__(self, head: SlotHead, features: Float[Tensor, "b s *o d"]) -> None:
        self.head = head
        self.features = features

    def probabilities(self, mask: Float[Tensor, "n s"]) -> Float[Tensor, "n o"]:
        """Per-finding probability under a presence mask, for as many mask rows as are handed in."""
        return torch.sigmoid(self.head(self.tiled(self.features, len(mask)), mask))

    def weights(self, mask: Float[Tensor, "n s"]) -> Float[Tensor, "n o s"]:
        """What the head says it read — the same attention `attention_readout` tabulates."""
        return self.head.slot_weights(self.tiled(self.features, len(mask)), mask)

    @staticmethod
    def tiled(tensor: Float[Tensor, "b ..."], rows: int) -> Float[Tensor, "n ..."]:
        """`tensor` repeated to `rows` rows, or itself when it already has them."""
        if rows == len(tensor):
            return tensor
        return tensor.repeat(rows // len(tensor), *([1] * (tensor.dim() - 1)))
