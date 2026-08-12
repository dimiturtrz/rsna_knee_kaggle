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

from collections.abc import Iterable

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

    def slices(self) -> int:
        """How deep a stack to decode: the deepest any member was fitted on.

        Derived, never defaulted. A member trained on twenty slices and fed twelve is a train/inference
        mismatch that raises nothing — the window loop reads `images.shape[2]`, so it simply runs on a
        shallower stack than the weights ever saw. The fingerprint cannot catch it either: it covers the
        image size and the group, not the depth of the stack they are drawn from. That is the same class
        of silent mismatch that cost 0.018 when inference ran at 32 slices.

        The MAX rather than each member's own, because a chunk is decoded once and shown to every
        member; a member that wants fewer simply windows over part of it.
        """
        return max(member.slices for member in self.package.members)

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
        if member.whole_stack:
            # A whole-stack member's slot embedding is the elementwise maximum over its ENTIRE stack.
            # Windowing it by `group` would hand it maxima over three slices at a time — a different
            # function of the same pixels, and one the holdout cannot expose, because holdout scoring
            # reads the same window training did. Measured consequence before this existed: two members
            # at 0.8481 holdout scored 0.882 on the leaderboard while five members at 0.8269 scored
            # 0.888.
            #
            # The MIDDLE `member.slices` of whatever was decoded, which is exactly what
            # `Trainer.predict` does — the decoded depth is the deepest any member wants, so a member
            # fitted on twelve must not be handed sixteen.
            width = min(member.slices, images.shape[2])
            start = (images.shape[2] - width) // 2
            with torch.no_grad(), torch.autocast("cuda", enabled=self.device.type == "cuda"):
                logits = model(images[:, :, start : start + width], mask, member.read_size).float()
            return torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        starts = self.windows(images.shape[2], member.group, overlap=overlap)
        total = torch.zeros(images.shape[0], len(TARGETS), device=self.device)
        with torch.no_grad(), torch.autocast("cuda", enabled=self.device.type == "cuda"):
            for start in starts:
                logits = model(images[:, :, start : start + member.group], mask, member.read_size).float()
                total = total + torch.sigmoid(logits)
        return (total / len(starts)).cpu().numpy().astype(np.float64)

    def predict_chunks(
        self, chunks: Iterable[tuple[UInt8[Tensor, "b n s h w"], Float[Tensor, "b n"]]], *, overlap: bool = True
    ) -> NDArray[np.float64]:
        """Every member over every decoded chunk, rank-combined into one table.

        The chunk loop is outside the member loop on purpose: a chunk is decoded once and then shown to
        every member, so the read cost is paid per STUDY rather than per (study, member). That is the
        whole reason members are grouped by the pixels they need.
        """
        per_member: dict[str, list[NDArray[np.float64]]] = {member.identifier: [] for member in self.package.members}
        for images, mask in chunks:
            for member in self.package.members:
                per_member[member.identifier].append(self.predict_member(member, images, mask, overlap=overlap))
        return self.combine([np.concatenate(per_member[member.identifier]) for member in self.package.members])

    @staticmethod
    def combine(per_member: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Rank-average across members, per label, normalised to [0, 1].

        Ranking is per label because the metric ranks per label; ranking across labels would mix twelve
        unrelated orderings into one and destroy exactly the information being scored.

        Static because it reads no member state — it is arithmetic over arrays that happens to describe
        how this ensemble votes. Analysis code needs exactly this rule without owning a package or a set
        of loaded models, and the alternative to saying so is calling it with a `None` self, which this
        repository has already done once with `Trainer.area_under_curve` and had to unpick.
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

    @staticmethod
    def submission(identifiers: list[str], predictions: NDArray[np.float64]) -> dict[str, list[object]]:
        """The submission table as plain columns: the study id, then one column per target in order.

        Static for the same reason `combine` is: it reads no ensemble state, only the target order. A
        test or an analysis that has predictions and identifiers should not have to construct a package
        and a dict of loaded models to lay them out as columns.
        """
        table: dict[str, list[object]] = {"StudyInstanceUID": list(identifiers)}
        for position, target in enumerate(TARGETS):
            table[target] = predictions[:, position].tolist()
        return table
