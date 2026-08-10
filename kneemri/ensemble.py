"""Predicting a study from many members, and combining them the way the metric actually reads.

The score is the unweighted mean of twelve per-label ROC AUCs, which is invariant under any strictly
increasing map of a label's scores. Two consequences are load-bearing here:

- Calibration is worth nothing, so no thresholding, no temperature, no Platt scaling.
- Members combine by averaging **ranks**, not probabilities. Averaging probabilities lets whichever
  member is most confident dominate; ranks carry the only information the metric reads.

Within one member, test-time windows are averaged as probabilities — an arithmetic mean of risk rather
than the geometric mean of odds that averaging logits gives. The two order studies differently, and
this is the pooling the published members' own out-of-fold predictions were produced under (measured:
disjoint windows, probability pooling, 0.025 max deviation against ours).
"""

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor

from kneemri.members import Member, MemberPackage
from kneemri.model import TARGETS, StudyModel


class Ensemble:
    """A package's members, read over test-time windows and combined by rank."""

    def __init__(self, package: MemberPackage, models: dict[str, StudyModel], device: torch.device) -> None:
        self.package = package
        self.models = models
        self.device = device

    def windows(self, slices: int, group: int, *, overlap: bool) -> list[int]:
        """Where each test-time window begins.

        Overlapping windows cost forward passes and no extra decoding, which is the cheap direction to
        spend — the cache already holds the slices. Disjoint windows are what the published members'
        out-of-fold predictions were computed under.
        """
        if overlap and slices >= group:
            return list(range(slices - group + 1))
        return [index * group for index in range(max(slices // group, 1))]

    def predict_member(
        self,
        member: Member,
        images: UInt8[Tensor, "b n s h w"],
        mask: Float[Tensor, "b n"],
        *,
        overlap: bool = False,
    ) -> NDArray[np.float64]:
        """One member's probabilities for a batch of studies, averaged over its windows."""
        model = self.models[member.identifier]
        model.eval()
        starts = self.windows(images.shape[2], member.group, overlap=overlap)
        total = torch.zeros(images.shape[0], len(TARGETS), device=self.device)
        with torch.no_grad(), torch.autocast("cuda", enabled=self.device.type == "cuda"):
            for start in starts:
                logits = model(images[:, :, start : start + member.group], mask, member.image_size).float()
                total = total + torch.sigmoid(logits)
        return (total / len(starts)).cpu().numpy().astype(np.float64)

    def combine(self, per_member: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Rank-average across members, per label, normalised to [0, 1].

        Ranking is per label because the metric ranks per label; ranking across labels would mix twelve
        unrelated orderings into one and destroy exactly the information being scored.
        """
        if len(per_member) == 1:
            return per_member[0]
        stacked = np.stack(per_member)
        ranks = np.empty_like(stacked)
        studies = stacked.shape[1]
        for index in range(stacked.shape[0]):
            for label in range(stacked.shape[2]):
                order = stacked[index, :, label].argsort().argsort()
                ranks[index, :, label] = order / max(studies - 1, 1)
        return ranks.mean(0)

    def submission(self, identifiers: list[str], predictions: NDArray[np.float64]) -> dict[str, list[object]]:
        """The submission table as plain columns: the study id, then one column per target in order."""
        table: dict[str, list[object]] = {"StudyInstanceUID": list(identifiers)}
        for position, target in enumerate(TARGETS):
            table[target] = predictions[:, position].tolist()
        return table
