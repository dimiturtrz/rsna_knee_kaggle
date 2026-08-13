"""One member's probabilities for a fixed set of studies — the half of an analysis that needs a GPU.

Separated from `member_curve` because the fitness gate is right that they are two subjects, and because
the same loop was already open-coded in a throwaway slice-count script. An analysis that loads members
and reads them over test-time windows is a recurring shape here, not a one-off.
"""

from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor

from kneemri.members import Window
from kneemri.model import TARGETS, StudyDesign, StudyModel
from kneemri.test_time import SpatialViews

GROUP = 3
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

        `slice_pooling` is read from the checkpoint rather than assumed: a member trained with per-slice
        max pooling is a different function from one trained with channel stacking, and loading either
        under the wrong rule would silently produce a model whose weights no longer mean what they did.

        The backbone is passed in rather than fetched here. Constructing it internally would have meant
        this method could only run with a network and a 90 MB download, which is to say it could not be
        tested at all — the mirror gate said so, and it was right.
        """
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = saved.get("config", {})
        design = StudyDesign(
            slice_pooling=str(config.get("slice_pooling", "channels")),
            anatomical_prior=bool(config.get("anatomical_prior", False)),
        )
        model = StudyModel(backbone, dim, slots, design)
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
