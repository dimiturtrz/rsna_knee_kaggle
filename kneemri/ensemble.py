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

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor

from kneemri.backbone import ENCODERS
from kneemri.crop_centre import Crop
from kneemri.members import Member, MemberPackage, Window
from kneemri.model import TARGETS, StudyModel
from kneemri.slot_reader import Band
from kneemri.slots import SCHEMES, SlotScheme
from kneemri.test_time import SpatialViews

logger = logging.getLogger(__name__)

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

NATIVE_BF16 = (8, 0)
"""Compute capability where bfloat16 stops being emulated. A T4 is (7, 5) and reports it as supported."""

BUDGET_TOKENS = 576
"""Patch tokens per image the budget was measured at: 336 px on a patch-14 grid is 24 x 24.

THE BUDGET COUNTED IMAGES AND ENCODER WIDTH AND NOT RESOLUTION, and on 2026-08-14 that cost two
submissions. `res392` reads 392 px, which is 28 x 28 = 784 tokens against this 576 — a third more
activation per image, before the attention matrices that grow with the SQUARE of the token count. Both
entries that came back unscored carried it; every entry that has ever scored (0.882, 0.888, 0.894) was
336 px only. A transformer's cost is per TOKEN, so a budget in images is only valid at one resolution.
"""

BUDGET_WIDTH = 384
"""The encoder width the budget was measured at.

Activation scales with hidden size, so the budget is divided by how much wider a member is: base gets
432 images, large 324. Both land near 8-9 GB by the table above — the headroom `small` has always run
with, rather than a limit discovered by a failed submission.
"""


@dataclass(frozen=True)
class Read:
    """What ONE decode of the corpus has to produce: how deep, under which scheme, at which sizes.

    One object because a decode is not free to vary these separately — the scheme decides which series
    are routed, the depth how many slices are sampled, and the sizes are simply the grids the same
    cropped stack is resized onto. Passing them one by one is the shape `Window` and `Split` were each
    created to fix in this package.

    Sizes is a tuple rather than a number because that is the whole point: resolution is the one axis a
    single read CAN serve many of, and pretending otherwise cost a full decode per resolution.

    `crop_mm` is here because it is the one reading rule that changes the PIXELS and was reaching the
    reader as a module constant regardless of what a member recorded. A member fitted on an 80 mm crop
    served under 130 mm is a different function, and nothing downstream would notice: every fingerprint
    still passes, because each is checked against its own reading.

    `band` is here for the same reason and was the last recorded reading rule the decode ignored: it
    decides WHICH slices exist to be sampled, so a member fitted on a wider band and served under a
    narrower one sees a stack it was never fitted on, and no fingerprint covers it either.
    """

    slices: int
    scheme: SlotScheme
    sizes: tuple[int, ...]
    crop: Crop = Crop()
    band: Band = Band()


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

    def reads(self) -> list[list[Member]]:
        """Members that can share ONE read of the corpus, whatever resolutions they then want.

        Grouping by pixels alone charged a full decode per RESOLUTION: our best member reads a 448 px
        cache and the rest read 336, so an entry carrying both decoded every DICOM twice — and decode is
        the I/O-bound half on a host with four CPUs. Measured in our own dry run: three groups, three
        decodes, for the same studies.

        What a decode actually cannot share is the slot SCHEME (which series are routed) and the slice
        DEPTH (how many are sampled). Resolution is not in that list: `SlotReader.cropped` stops before
        the resize, so one read serves every size exactly — `Study.bags` returns them together and each
        is byte-identical to a native read at that size.

        So the unit of decoding is (scheme, slices, crop), and pixel groups remain the unit of
        PREDICTION, because a member is still fed the cache size it was fitted from.

        THE TIE RULE IS IN THAT LIST TOO, and it is the reason a fused entry may need two decodes: the
        published members were fitted under a LAST-maximal router and ours under first (knee-evn), and a
        different router puts a different SERIES in a slot — which is different pixels, not a different
        view of the same ones.

        THE CROP JOINS THAT LIST for the same reason the scheme is in it: it changes which pixels exist,
        not merely how they are sampled, so two members that crop differently cannot share one read.
        `Member.pixel_group` has always included it; this is the decode side finally agreeing.
        """
        gathered: dict[tuple[str, int, Crop, str, Band], list[Member]] = {}
        for member in self.package.members:
            key = (member.scheme, member.slices, member.crop, member.tie, member.band)
            gathered.setdefault(key, []).append(member)
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

    def decoded_scheme(self, members: list[Member] | None = None) -> SlotScheme:
        """The slot scheme the test set must be DECODED under — the members', never a default.

        The sibling of `slices`, and the last of the four fields that travelled in a checkpoint
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
        ties = {member.tie for member in (members or self.package.members)}
        if len(ties) != 1:
            message = f"members disagree on the tie rule ({sorted(ties)}) — that is two decodes, not one"
            raise ValueError(message)
        return SlotScheme(tuple(known[name] for name in names), tie=ties.pop())

    def arithmetic(self, member: Member) -> torch.dtype:
        """The dtype this DEVICE can actually run, which is not always the one the member was fitted in.

        SUPPORTED IS NOT NATIVE, and that distinction is the whole guard. A T4 is Turing, sm_75:
        `is_bf16_supported()` answers True because torch EMULATES bfloat16 there, so the autocast call
        does not raise — it just runs a different, slower path through kernels that were never the ones
        the member was fitted under. Native bf16 starts at sm_80.

        So the question asked here is `capability >= (8, 0)`, not "is it supported". Emulated bf16 is
        exactly the kind of works-until-it-does-not that costs a submission rather than a warning, and
        fp16 is what every entry that has ever SCORED here ran under.

        AND IT IS NOT THE COMPROMISE THIS DOCSTRING ONCE CLAIMED. fp16 has 5 exponent bits against
        bf16's 8, but it has MORE mantissa (10 against 7) — so the only thing at risk is range, and the
        range was measured rather than feared: on real studies these members peak at 127.6 of fp16's
        65504 ceiling, 2026 for the v3 one, a margin of 32x at worst. The weights are float32 in the
        checkpoint either way (autocast never stored bf16), 0.0004% of them sit below fp16's smallest
        subnormal, and loss scaling is a TRAINING device for gradient underflow that inference has no
        use for.

        So this serves fp16 for the speed — a T4 has fp16 tensor cores and none for bf16, which is
        65 TFLOPS against 8.1 — and gives up nothing that was measured to matter. `predict_member`
        carries the belt: a non-finite logit redoes that pass in fp32.
        """
        native = self.device.type == "cuda" and torch.cuda.get_device_capability() >= NATIVE_BF16
        if member.precision is torch.bfloat16 and self.device.type == "cuda" and not native:
            logger.warning("%s was fitted in bf16 and this device cannot run it — serving fp16", member.identifier)
            return torch.float16
        return member.precision

    def forwarded(
        self,
        model: StudyModel,
        read: UInt8[Tensor, "b n s h w"],
        mask: Float[Tensor, "b n"],
        window: Window,
        rows: int,
    ) -> Float[Tensor, "b t"]:
        """The chunked forward, halving the rows per pass rather than dying when memory runs out.

        A budget is a PREDICTION about a card we cannot measure from here, and the cost of predicting
        high is total: the kernel writes no submission (deliberately — a fallback scored 0.500 twice),
        Kaggle fails the entry, and the run is gone. The cost of predicting low is a slower pass. So an
        out-of-memory is caught and retried at half the rows until it fits, and only a batch of one that
        still cannot run is a real failure.

        This is the belt to the budget's braces: the budget being wrong about resolution is exactly what
        this session discovered, and no arithmetic here can be trusted more than the card's own answer.
        """
        while True:
            try:
                return torch.cat(
                    [
                        model(read[begin : begin + rows], mask[begin : begin + rows], window.size).float()
                        for begin in range(0, read.shape[0], rows)
                    ]
                )
            except torch.OutOfMemoryError:
                if rows <= 1:
                    raise
                torch.cuda.empty_cache()
                rows = max(1, rows // 2)
                logger.info("out of memory — retrying at %d rows per forward", rows)

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
        encoder = ENCODERS[member.variant]
        tokens = (window.size // encoder.patch) ** 2
        # NEVER ABOVE THE CEILING, only below it. 864 is the largest forward that has ever completed on
        # a T4; the width and token terms can justify sending FEWER images, never more, and a small read
        # size would otherwise scale the budget up into a number nothing has ever survived.
        budget = min(IMAGE_BUDGET, IMAGE_BUDGET * BUDGET_WIDTH * BUDGET_TOKENS // (encoder.dim * max(tokens, 1)))
        rows = max(1, budget // max(images.shape[1] * window.width, 1))
        # THE DTYPE COMES FROM THE MEMBER. Written dtype-less, this autocast defaults to float16, so
        # every bf16-trained member was served in arithmetic it was never fitted under — and fp16's
        # exponent range is float32's minus eight bits, which a bf16 model can overflow to inf on the
        # very T4 the submission runs on. The witness check found it; nothing else could have.
        precision = self.arithmetic(member)
        with torch.no_grad(), torch.autocast("cuda", dtype=precision, enabled=self.device.type == "cuda"):
            for view in views.apply(images):
                for start in window.starts:
                    read = view[:, :, start : start + window.width]
                    logits = self.forwarded(model, read, mask, window, rows)
                    # THE ONLY FAILURE MODE fp16 HAS THAT fp32 DOES NOT: an activation past 65504 comes
                    # back inf, and a rank-average over inf is a wrong answer that looks like a number.
                    # Measured on real studies, these members peak at 128 of 65504 (3% for the v3 one),
                    # so this should never fire — and if a hidden study is stranger than any we hold, it
                    # costs that chunk a slower pass instead of costing the entry its meaning.
                    if not torch.isfinite(logits).all():
                        logger.warning("%s produced non-finite logits — redoing this pass in fp32", member.identifier)
                        with torch.autocast("cuda", enabled=False):
                            logits = self.forwarded(model, read, mask, window, rows)
                    total = total + torch.sigmoid(logits)
                    passes += 1
        return (total / passes).cpu().numpy().astype(np.float64)

    def predict_groups(
        self,
        chunks_for: "Callable[[Read], Iterable[dict[int, tuple[UInt8[Tensor, 'b n s h w'], Float[Tensor, 'b n']]]]]",
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
        for members in self.reads():
            read = Read(
                slices=self.slices(members),
                scheme=self.decoded_scheme(members),
                sizes=tuple(sorted({member.image_size for member in members})),
                # One value, because `reads` groups by it — a group spanning two crops would be a group
                # whose members want different pixels, which is the thing that grouping exists to stop.
                crop=members[0].crop,
                band=members[0].band,
            )
            per_member.extend(self.predict_members(members, chunks_for(read), overlap=overlap))
        return self.combine(per_member)

    def predict_members(
        self,
        members: list[Member],
        chunks: Iterable[dict[int, tuple[UInt8[Tensor, "b n s h w"], Float[Tensor, "b n"]]]],
        *,
        overlap: bool = True,
    ) -> list[NDArray[np.float64]]:
        """One array per member, over every chunk — UNCOMBINED, so groups can be combined together.

        A chunk arrives as {decoded size: (images, mask)} from ONE read, and each member is handed the
        size it was fitted from. That is what lets members of different resolutions share a decode
        without any of them being fed pixels it never saw.

        Returned uncombined because rank-averaging must happen ACROSS every member of every group at
        once: combining per group and then averaging the results would weight a group of one as heavily
        as a group of five.
        """
        per_member: dict[str, list[NDArray[np.float64]]] = {member.identifier: [] for member in members}
        failed: set[str] = set()
        for sized in chunks:
            for member in members:
                if member.identifier in failed:
                    continue
                images, mask = sized[member.image_size]
                try:
                    per_member[member.identifier].append(self.predict_member(member, images, mask, overlap=overlap))
                # ONE MEMBER MUST NOT COST THE ENTRY. Three submissions died because any single failure
                # anywhere took the whole run down, and a run that fails writes no submission at all. An
                # ensemble is the one structure that does not need every part: dropping a member costs a
                # little accuracy, and dropping the entry costs a slot and a day.
                except Exception:
                    logger.exception("member %s failed mid-prediction — dropping it from this entry", member.identifier)
                    failed.add(member.identifier)
        survived = [member for member in members if member.identifier not in failed and per_member[member.identifier]]
        if not survived:
            message = f"every member of this group failed: {sorted(failed)}"
            raise RuntimeError(message)
        return [np.concatenate(per_member[member.identifier]) for member in survived]

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
