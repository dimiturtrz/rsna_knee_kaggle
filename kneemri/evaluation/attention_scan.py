"""One member's attention weights over a holdout — the half of the reading that needs the model.

Split from `attention_readout`, which is the arithmetic: that module is a function of two arrays and
runs anywhere, this one holds a card, a batch size and a model. The same split `attribution` makes.
"""

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor

from core.findings import TARGETS
from kneemri.inference.holdout_predictions import ServedModel
from kneemri.inference.holdout_slab import HoldoutSlab

BATCH = 8
"""Studies per encoder pass — the same batch the holdout scorer uses, for the same memory reason."""


class AttentionScan:
    """A member's attention weights over a holdout — the half of the reading that needs the model.

    It calls `SlotHead.slot_weights`, which recomputes the weights from the same slot features `forward`
    reads. Nothing is hooked, patched or captured, and `forward` is not called at all here — so a member
    on disk computes exactly what it computed before, whether or not anyone ever runs this.
    """

    def __init__(self, device: torch.device, batch: int = BATCH) -> None:
        self.device = device
        self.batch = batch

    def weights(self, served: ServedModel, slab: HoldoutSlab) -> Float[NDArray[np.float32], "b o s"]:
        """Attention per study, finding and slot, averaged over the member's own window starts.

        NO TEST-TIME VIEWS. The scorer averages over rotations and shifts because that is what the
        submission path does and a ruler has to rank members under the policy they ship with. This is
        not a ranking: it asks where a member looks in a study, and averaging that over four rotations of
        the study would blur the one axis the question is about.

        The member's OWN precision, for the reason `HoldoutPredictions.predict` gives: a bf16 member read
        in fp16 is a different function of the same pixels.
        """
        chunks = []
        with torch.no_grad(), served.autocast(self.device):
            for images, mask in slab.batches(self.batch):
                pixels: UInt8[Tensor, "b s g p q"] = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
                flags: Float[Tensor, "b s"] = torch.from_numpy(mask).to(self.device)
                chunks.append(self._batch(served, pixels, flags))
        return np.concatenate(chunks).astype(np.float32)

    def _batch(
        self,
        served: ServedModel,
        pixels: UInt8[Tensor, "b s g p q"],
        flags: Float[Tensor, "b s"],
    ) -> Float[NDArray[np.float32], "b o s"]:
        """One batch's weights, meaned over the window starts this member reads."""
        model, window = served.model, served.window
        total = torch.zeros(pixels.shape[0], len(TARGETS), pixels.shape[1], device=self.device)
        for start in window.starts:
            features = model.slot_features(served.crop(pixels, start), window.size)
            total = total + model.head.slot_weights(features, flags).float()
        return (total / len(window.starts)).cpu().numpy()
