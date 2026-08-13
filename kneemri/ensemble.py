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

from collections.abc import Callable, Iterable

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor

from kneemri.backbone import ENCODERS
from kneemri.members import Member, MemberPackage
from kneemri.model import TARGETS, StudyModel
from kneemri.slots import SCHEMES, SlotScheme
from kneemri.test_time import SpatialViews

IMAGE_BUDGET = 864
"""Images per forward for a 384-wide encoder — the largest that has ever completed on a Kaggle T4.

Not a tuned number: it is 48 studies x 6 slots x a 3-slice window, which is what every scored kernel
before 2026-08-12 sent, including the 0.894 entry — whose members are all `small`. Deriving the study
count from it keeps that peak fixed however deep a member reads, instead of letting depth multiply it.

IT DOES NOT TRANSFER TO A WIDER ENCODER. Measured on this card at 336 px, no-grad, autocast, as
activation per image beyond the weights:

    small  (384)   0.0099 GB      864 images ->  8.6 GB     fits a T4's ~15 GB
    base   (768)   0.0187 GB      864 images -> 16.5 GB     does NOT
    large  (1024)  0.0235 GB      864 images -> 21   GB     does NOT

A flat 864 would therefore have OOM'd the first entry carrying `large_deep_rank24` — which is in our
pool at 0.8450 — in exactly the way fusion v2 did, and for exactly the same reason: one constant
standing in for a value that travels per member.
"""

BUDGET_WIDTH = 384
"""The encoder width the budget was measured at.

Activation scales with hidden size, so the budget is divided by how much wider a member is: base gets
432 images, large 324. Both land near 8-9 GB by the table above — the headroom `small` has always run
with, rather than a limit discovered by a failed submission.
"""


class Ensemble:
    """A package's members, read over test-time windows and combined by rank."""

    def __init__(
        self,
        package: MemberPackage,
        models: dict[str, StudyModel],
        device: torch.device,
        views: SpatialViews | None = None,
    ) -> None:
        self.package = package
        self.models = models
        self.device = device
        # None means the identity view alone, which is what every entry so far has run. Spatial views
        # are the one test-time axis a whole-stack member has not already spent (knee-eeo).
        self.views = views

    def groups(self) -> list[list[Member]]:
        """Members needing identical pixels, gathered so ONE decode serves each group.

        `pixel_group` has existed since the first package and inference never asked: the whole test set
        was decoded once, at whatever the default happened to be, and every member was fed it. That was
        harmless only while every member wanted the same pixels.

        It stopped being harmless the moment resolution paid off. Our best recipe reads a 448 px cache
        at 392, so it records img=448, while every other member records 336 — and `decoded_size`
        correctly refuses to serve both from one decode. Without grouping, the two things that actually
        work here (a better recipe, and ensembling) could not be used together.

        The cost is honest and worth naming: N groups is N decodes of the test set, which is the
        dominant cost of a submission. Grouping keeps it at one decode PER GROUP rather than per member.
        """
        gathered: dict[str, list[Member]] = {}
        for member in self.package.members:
            gathered.setdefault(member.pixel_group, []).append(member)
        return list(gathered.values())

    def slices(self, members: list[Member] | None = None) -> int:
        """How deep a stack to decode: the deepest any member was fitted on.

        Derived, never defaulted. A member trained on twenty slices and fed twelve is a train/inference
        mismatch that raises nothing — the window loop reads `images.shape[2]`, so it simply runs on a
        shallower stack than the weights ever saw. The fingerprint cannot catch it either: it covers the
        image size and the group, not the depth of the stack they are drawn from. That is the same class
        of silent mismatch that cost 0.018 when inference ran at 32 slices.

        The MAX rather than each member's own, because a chunk is decoded once and shown to every
        member; a member that wants fewer simply windows over part of it.
        """
        return max(member.slices for member in (members or self.package.members))

    def decoded_size(self, members: list[Member] | None = None) -> int:
        """The resolution the test set must be DECODED at — the members', never a default.

        `run()` already derives the slice depth this way, after a member fitted on a deeper stack was
        silently scored on twelve. Resolution and the slot list were left behind on that lesson: the
        decode used `SlotReader()`'s default 336 whatever `img` said, and `pixel_group` — which exists
        precisely to say which members can share one decode — was computed by `verify_members` and
        consulted nowhere at inference.

        It matters as soon as a member is fitted on another cache. A member trained by reading a 448 px
        cache at 392 would be handed 336 px pixels here and UPSAMPLED to 392, where training
        DOWNSAMPLED from 448. Same tensor shape, different image, nothing raised.

        A single value rather than a maximum, because unlike depth this cannot be windowed down: two
        members wanting different resolutions need two decodes, so disagreement is refused loudly here
        rather than silently resolved in favour of one of them.
        """
        sizes = {member.image_size for member in (members or self.package.members)}
        if len(sizes) != 1:
            message = f"members disagree on decode resolution {sorted(sizes)} — that is two decodes, not one"
            raise ValueError(message)
        return sizes.pop()

    def decoded_scheme(self, members: list[Member] | None = None) -> SlotScheme:
        """The slot scheme the test set must be DECODED under — the members', never a default.

        The sibling of `decoded_size`, and the last of the four fields that travelled in a checkpoint
        and were read by a default somewhere. `Submission.decode` built `SlotPipeline()` with no
        scheme, so a nine-slot member would have been routed and bagged under the canonical six: the
        head's slot embedding expects nine positions, would receive six, and the three it never saw are
        precisely the ones the extended scheme exists to add.

        Refused rather than resolved when members disagree, for the same reason as resolution — a
        different slot list is a different decode, not a subset of one.
        """
        lists = {tuple(member.config["slots"]) for member in (members or self.package.members)}
        if len(lists) != 1:
            message = f"members disagree on the slot list ({len(lists)} distinct) — that is two decodes, not one"
            raise ValueError(message)
        names = lists.pop()
        known = {slot.name: slot for scheme in SCHEMES.values() for slot in scheme}
        missing = [name for name in names if name not in known]
        if missing:
            message = f"members name slots this build does not define: {missing}"
            raise ValueError(message)
        return SlotScheme(tuple(known[name] for name in names))

    def predict_member(
        self,
        member: Member,
        images: UInt8[Tensor, "b n s h w"],
        mask: Float[Tensor, "b n"],
        *,
        overlap: bool = False,
    ) -> NDArray[np.float64]:
        """One member's probabilities for a batch of studies, averaged over its windows.

        The window comes from the MEMBER, which is the only thing that knows how it was fitted. This
        method used to decide inline, `member_curve` decided separately, and `HoldoutPredictions` took
        loose arguments — three independent decisions about one fact, and each got it wrong at least
        once.
        """
        model = self.models[member.identifier]
        model.eval()
        window = member.window(images.shape[2], overlap=overlap)
        views = self.views or SpatialViews(degrees=(0.0,), shifts=((0.0, 0.0),))
        total = torch.zeros(images.shape[0], len(TARGETS), device=self.device)
        passes = 0
        # THE FORWARD IS SPLIT BY IMAGES, NOT BY STUDIES, and this cost a submission on 2026-08-12.
        # A chunk is 48 studies; what reaches the encoder is studies x slots x window width. A windowed
        # member sends 48 x 6 x 3 = 864 images and a whole-stack member sends 48 x 6 x 12 = 3,456 — four
        # times the peak activation for an identical chunk. Locally that is invisible: this card has
        # 32 GB and training runs under a --vram-gb cap with checkpointing. On the kernel's 16 GB T4 it
        # is an OOM, the run produced no submission, and the deleted fallback scored the sample table at
        # 0.500. Compute was never the problem — reading whole is CHEAPER than ten overlapping windows
        # (12 images against 30), which is what rules out a timeout and points at memory.
        #
        # The budget is per ENCODER WIDTH, not flat: a 1024-wide member costs 2.4x a 384-wide one per
        # image, so one number for both is the same mistake one number for every depth was.
        budget = IMAGE_BUDGET * BUDGET_WIDTH // ENCODERS[member.variant].dim
        rows = max(1, budget // max(images.shape[1] * window.width, 1))
        # THE DTYPE COMES FROM THE MEMBER. Written dtype-less, this autocast defaults to float16, so
        # every bf16-trained member was served in arithmetic it was never fitted under — and fp16's
        # exponent range is float32's minus eight bits, which a bf16 model can overflow to inf on the
        # very T4 the submission runs on. The witness check found it; nothing else could have.
        with torch.no_grad(), torch.autocast("cuda", dtype=member.precision, enabled=self.device.type == "cuda"):
            for view in views.apply(images):
                for start in window.starts:
                    read = view[:, :, start : start + window.width]
                    logits = torch.cat(
                        [
                            model(read[begin : begin + rows], mask[begin : begin + rows], window.size).float()
                            for begin in range(0, read.shape[0], rows)
                        ]
                    )
                    total = total + torch.sigmoid(logits)
                    passes += 1
        return (total / passes).cpu().numpy().astype(np.float64)

    def predict_groups(
        self,
        chunks_for: "Callable[[list[Member], int], Iterable[tuple[UInt8[Tensor, 'b n s h w'], Float[Tensor, 'b n']]]]",
        *,
        overlap: bool = True,
    ) -> NDArray[np.float64]:
        """Predict every member, decoding once PER PIXEL GROUP, and rank-combine across all of them.

        The decode is injected rather than performed here: this class knows which members can share
        pixels and how their outputs combine, and nothing about DICOM directories. It lives here rather
        than in `Submission` because the loop is six calls into this object and two into the caller's —
        the feature-envy gate said so, and it was right.

        Combining across ALL groups at once matters: rank-averaging within a group and then averaging
        the group results would weight a group of one as heavily as a group of five.
        """
        per_member: list[NDArray[np.float64]] = []
        for members in self.groups():
            per_member.extend(self.predict_members(members, chunks_for(members, self.slices(members)), overlap=overlap))
        return self.combine(per_member)

    def predict_members(
        self,
        members: list[Member],
        chunks: Iterable[tuple[UInt8[Tensor, "b n s h w"], Float[Tensor, "b n"]]],
        *,
        overlap: bool = True,
    ) -> list[NDArray[np.float64]]:
        """One array per member, over every chunk — UNCOMBINED, so groups can be combined together.

        Returned uncombined because rank-averaging must happen ACROSS every member of every group at
        once: combining per group and then averaging the results would weight a group of one as heavily
        as a group of five.
        """
        per_member: dict[str, list[NDArray[np.float64]]] = {member.identifier: [] for member in members}
        for images, mask in chunks:
            for member in members:
                per_member[member.identifier].append(self.predict_member(member, images, mask, overlap=overlap))
        return [np.concatenate(per_member[member.identifier]) for member in members]

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
