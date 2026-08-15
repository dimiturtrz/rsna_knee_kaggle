"""One member's probabilities for a fixed set of studies — the half of an analysis that needs a GPU.

Separated from `member_curve` because the fitness gate is right that they are two subjects, and because
the same loop was already open-coded in a throwaway slice-count script. An analysis that loads members
and reads them over test-time windows is a recurring shape here, not a one-off.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor
from transformers import AutoModel

from kneemri.backbone import ENCODERS
from kneemri.holdout_slab import HoldoutSlab
from kneemri.members import Member, Window
from kneemri.model import TARGETS, StudyModel
from kneemri.prediction_cache import PredictionCache
from kneemri.test_time import SpatialViews

logger = logging.getLogger(__name__)

BATCH = 8


class HoldoutPredictions:
    """Each checkpoint's probabilities for one fixed set of studies.

    Separate from the arithmetic below because it is the half that needs a GPU, a cache and 90 seconds;
    keeping the curve itself a pure function of arrays is what lets the curve be tested at all.
    """

    def __init__(self, device: torch.device, views: "SpatialViews | None" = None) -> None:
        self.device = device
        # The SAME views the submission path uses, or none. A whole-stack member gets one window, so
        # window averaging - worth a measured +0.0156 to the windowed members - is unavailable to it and
        # rotation is the only test-time average left. The ruler has to apply whatever deployment will,
        # or it ranks members under a policy they will not ship under.
        self.views = views or SpatialViews(degrees=(0.0,), shifts=((0.0, 0.0),))

    def load(self, checkpoint: Path, backbone: torch.nn.Module, dim: int, slots: int) -> StudyModel:
        """One member, rebuilt in the shape its own config records.

        The shape comes from the checkpoint's own config rather than from this method's assumptions: a
        member trained with per-slice max pooling is a different function from one trained with channel
        stacking, and loading either under the wrong rule would silently produce a model whose weights
        no longer mean what they did.

        The backbone is passed in rather than fetched here. Constructing it internally would have meant
        this method could only run with a network and a 90 MB download, which is to say it could not be
        tested at all — the mirror gate said so, and it was right.
        """
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # THE SHARED REBUILD, not this method's own reading of the config. Every switch was read back
        # here except the two that decide the head's WIDTH and whether the encoder carries adapters, so
        # this could not load a focally pooled or a LoRA member at all - and the serve path, which read
        # those two, was the half missing the prior. One rebuild has one set of blind spots.
        model = StudyModel.from_config(saved.get("config", {}), backbone, dim, slots)
        model.load_trained(saved["model"])
        return model.to(self.device).eval()

    def predict(
        self,
        model: StudyModel,
        images: UInt8[np.ndarray, "b n s h w"],
        mask: Float[np.ndarray, "b n"],
        window: "Window",
        precision: torch.dtype = torch.float16,
    ) -> NDArray[np.float64]:
        """Probabilities for every study, averaged over the given test-time windows.

        `width` is how many slices each window covers, and it defaults to the three-slice group only
        because that is what a windowed member reads. A WHOLE-STACK member's slot embedding is the
        elementwise maximum over its entire stack, so it must be given one window of full width — this
        was hardcoded to GROUP, which windowed such a member by three and quietly scored a different
        function. The same defect shipped to the leaderboard through `Ensemble` (knee-y8y); this is the
        analysis path, where it decides which members we choose to submit.
        """
        chunks = []
        # The member's OWN arithmetic, not autocast's cuda default of float16: this is the tool that
        # decides which members to submit, so scoring a bf16 member in fp16 ranks it under a function
        # it was never fitted as. Same defect as Ensemble.predict_member carried until today.
        with torch.no_grad(), torch.autocast("cuda", dtype=precision, enabled=self.device.type == "cuda"):
            for begin in range(0, images.shape[0], BATCH):
                pixels: UInt8[Tensor, "b n s h w"] = torch.from_numpy(
                    np.ascontiguousarray(images[begin : begin + BATCH])
                ).to(self.device)
                flags: Float[Tensor, "b n"] = torch.from_numpy(mask[begin : begin + BATCH]).to(self.device)
                total = torch.zeros(pixels.shape[0], len(TARGETS), device=self.device)
                passes = 0
                for view in self.views.apply(pixels):
                    for start in window.starts:
                        total = total + torch.sigmoid(
                            model(view[:, :, start : start + window.width], flags, window.size).float()
                        )
                        passes += 1
                chunks.append((total / passes).cpu().numpy())
        return np.concatenate(chunks).astype(np.float64)

    def cached(
        self, reading: Member, slab: HoldoutSlab, cache: PredictionCache, window: Window
    ) -> Float[np.ndarray, "studies targets"]:
        """One member's probabilities — from the cache when it holds them, from the encoder when not.

        The encoder, its download, the holdout's pixels and the GPU are ALL reached only on a miss. That
        is the difference between an analysis that can run while the card belongs to something else and
        one that cannot, and every paired interval, oracle and subset search here is arithmetic over what
        this returns.

        `load` takes its backbone as an argument so it can be tested without a network; this method is
        the caller that does the fetching, and it does it inside the miss branch for the same reason.
        """
        entry = cache.key(reading.file, window, reading.precision)
        stored = cache.read(entry)
        if stored is not None:
            return stored
        images, mask = slab.arrays()
        encoder = ENCODERS[reading.variant]
        model = self.load(reading.file, AutoModel.from_pretrained(encoder.repository), encoder.dim, reading.slot_count)
        values = self.predict(model, images, mask, window, reading.precision)
        cache.write(entry, values)
        return values
