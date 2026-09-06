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
import time
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch
from jaxtyping import Float, UInt8
from numpy.typing import NDArray
from torch import Tensor

from core.data.corpus.slot_reader import Reading
from core.data.corpus.slots import SCHEMES, SlotScheme
from core.findings import TARGETS
from kneemri.inference.budget import SECONDS_PER_HOUR, Budget
from kneemri.inference.members import RESIZED_FROM, Member, MemberPackage, Window
from kneemri.inference.test_time import SpatialViews
from kneemri.inference.vote import RankAverage, Vote
from kneemri.models.backbone import ENCODERS
from kneemri.models.model import StudyModel

logger = logging.getLogger(__name__)

SizedBag = dict[int, tuple[UInt8[Tensor, "b n s h w"], Float[Tensor, "b n"]]]
"""One decoded chunk at every size ONE read group serves — the resize is the only size-dependent step, so
one decode answers 448 and 336 together. Named because the traversal passes a LIST of these, one per group."""

Decision = TypeVar("Decision", bound=Hashable)
"""A field the whole decode has to share — the slot list, the tie rule — not one member's own setting."""

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


CHANCE = 0.5
"""ROC AUC of a member that knows nothing — what a member's holdout has to be counted above.

The competition score is a mean of AUCs, and an AUC of 0.5 is the value a random ordering earns, so
`holdout - CHANCE` is what a member contributes over not being in the entry at all. Using the holdout
itself instead would call a 0.50 member half as useful as a 1.00 one, when it is useless.
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

    `reading` is here because it is what changes the PIXELS and was reaching the reader as module
    constants regardless of what a member recorded. A member fitted on an 80 mm crop served under 130 mm
    is a different function, and nothing downstream would notice: every fingerprint still passes,
    because each is checked against its own reading. The band decides WHICH slices exist to be sampled,
    so a member fitted on a wider band and served under a narrower one is the same defect one axis over.

    It is the WHOLE `Reading`, not its two fields spelled out again, because `Reading.reader` is the one
    place that turns a geometry into a decoder — carrying the fields separately meant `Submission`
    assembled its own `SlotReader` from them, so a third reading axis would have been applied at cache
    time and silently dropped at serve time. That is the shape `Member.reading` and `StudyCache.of`
    already exist to close, arriving here from the third direction.
    """

    slices: int
    scheme: SlotScheme
    sizes: tuple[int, ...]
    reading: Reading = Reading()


@dataclass(frozen=True)
class SharedRead:
    """Serving an already-fitted member under ANOTHER read group's depth and band.

    THE ENTRY'S DECODE BILL IS THE NUMBER OF READ GROUPS, NOT THE NUMBER OF SEATS. `Ensemble.reads`
    groups members by their reading, and a group is one decode of every study; sizes collapse inside a
    group because `SlotReader.cropped` stops before the resize, but the slice count and the band do not
    — they decide WHICH files are opened. `raptorweight_v1`'s six seats span three groups (20 slices at
    the 0.1-0.9 band, 12 at that band, 12 at the historic one), and its Kaggle log decodes the same
    three visible studies three times over: `decode 4s`, then `2s`, then `2s`, projecting 7.6 h of a
    9 h limit. One group would decode 12 slices per study instead of 44.

    So the question this asks is not "how much accuracy does a shallower read cost" but "how much does
    it cost to serve every seat out of ONE decode". It lives beside `Serving` because the same object
    has to answer it in BOTH halves: the kernel serves through it, and `holdout_backfill` prices it on
    each member's own holdout. Two copies of a regroup would be two definitions of what shipped.

    The band is the reader's own `[low, high]` recorded form, so the value handed here is the value a
    manifest carries and nothing has to translate it.

    A re-grouped member carries the `RESIZED_FROM` mark, which is what `TrainerRecord.instead_of`
    refuses on. The mark's name says "size" and its RULE is "this is not the reading the trainer
    fitted", which is what a re-grouped member is; without it the trainer's own array would be handed
    back and an offline price would read zero.
    """

    slices: int
    band: tuple[float, float]

    def regrouped(self, fitted: Member) -> Member:
        """The same weights, read at this group's depth and band."""
        return replace(
            fitted,
            config={**fitted.config, "slices": self.slices, "band": list(self.band), RESIZED_FROM: fitted.read_size},
        )

    def beside(self, checkpoint: Path) -> Path:
        """Where a priced record goes: beside the checkpoint, named for the group that produced it."""
        low, high = self.band
        return checkpoint.with_name(f"{checkpoint.stem}_n{self.slices}b{low:g}-{high:g}{checkpoint.suffix}")


@dataclass(frozen=True)
class Serving:
    """How the members are RUN, as opposed to what any of them recorded about itself.

    Four knobs that no member can answer and that every one of them is served under: the test-time
    views, the wall-clock ceiling, a forced arithmetic, and how the members' tables become one. They
    arrived as three more positional arguments on an `__init__` that already had three, which is the
    long-arg-list smell exactly — and a caller passing the wrong one of three same-shaped optionals is
    a defect that runs.
    """

    views: SpatialViews | None = None
    """None means the identity view alone, which is what every entry so far has run. Spatial views are
    the one test-time axis a whole-stack member has not already spent (knee-eeo)."""
    budget: Budget | None = None
    """None means no wall-clock ceiling, which is right for a dry run and a witness — only a Kaggle
    batch session is killed for running long, and only the kernel knows it is inside one."""
    precision: torch.dtype | None = None
    """None means each member is served in the arithmetic it was FITTED in, which is what a submission
    wants. A dtype forces one across every member, and the only caller that should want that is one
    MEASURING what a precision costs: the loadcheck wrapped `predict_member` in an outer autocast to
    compare three, but the inner autocast that `arithmetic` decides carries an explicit dtype and
    overrides it — so all three arms ran the same arithmetic and reported max|diff| of exactly 0.0,
    twelve times, which read as "fp16 is free". A knob that is not connected reports agreement."""
    read_size: int | None = None
    """None means each member is read at the size it was FITTED at, which is what a submission wants.
    A size forces one across every member, for an entry whose variable IS the resolution. What it buys
    is FORWARD time and never decode time: `Member.resized` overrides `train_size`, never `img`, so the
    corpus is still decoded once at the cache size the members share (`reads` groups without resolution
    precisely because one read serves every size). Priced offline first — six champion twins go 0.8706
    native, 0.8687 at 280 px, 0.8637 at 224."""
    vote: Vote | None = None
    """None means `RankAverage`, the rule every number this project has published came out of. It sits
    here because it is the same KIND of knob as the other three — something no member can answer,
    decided once for the whole run — and because `member_curve` has been able to vary the vote offline
    since the family existed while the SUBMISSION path could not vary it at all. An axis measurable
    only offline is an axis whose board transfer cannot be asked about."""

    group: SharedRead | None = None
    """None means each member is read at the depth and band it was FITTED at, and the entry pays one
    decode per distinct reading. A group forces ONE reading across every member, which is the only knob
    here that moves DECODE time rather than forward time — `reads` collapses to a single group and the
    corpus is opened once. Priced offline on the champion's five fold-0 twins: 0.8706 native against
    0.8684 at 12 slices / band 0.1-0.9, a cost of 0.0022 under the 0.0045 board resolution."""

    def read(self, member: Member) -> Member:
        """The member as this entry serves it: its own reading, unless the entry forces one.

        The re-group runs FIRST and the size second, because they act on different halves of the read:
        depth and band decide WHICH files are decoded, and a size is applied to pixels already decoded
        (knee-d2fc). Reversed, the size override would be overwritten by the regroup's config copy.
        """
        served = member if self.group is None else self.group.regrouped(member)
        return served if self.read_size is None else served.resized(self.read_size)


class Ensemble:
    """A package's members, read over test-time windows and combined by rank."""

    def __init__(
        self,
        package: MemberPackage,
        models: dict[str, StudyModel],
        device: torch.device,
        serving: Serving | None = None,
    ) -> None:
        self.package = package
        self.models = models
        self.device = device
        self.serving = serving or Serving()
        # The views this run is CURRENTLY serving, which the frozen `Serving` cannot be: they are the
        # first thing given up under a deadline, and giving them up is an edit to what the rest of the
        # run does. `Serving` stays the entry's declaration; this is the run's state.
        self.views = self.serving.views
        # Seconds each member has actually cost, which is what a give-up order under a deadline has to
        # divide by. A member's cost is not readable from its config: depth, resolution and encoder
        # width all multiply, and the run only learns the product by paying it.
        self.spent: dict[str, float] = {}

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

        THE KEY ITSELF IS `Member.decode_key`, not a tuple spelled out here. Written inline, it was
        reachable only through a mounted package and an `Ensemble`, so answering "which of our
        checkpoints share a decode?" meant building a throwaway package and dry-running it. On the
        member, an analysis tool groups raw checkpoints by the SAME property the kernel keys on, and
        cannot answer a decode count the serve path would contradict.
        """
        gathered: dict[tuple[str, int, str, Reading], list[Member]] = {}
        for member in self.served():
            gathered.setdefault(member.decode_key, []).append(member)
        return list(gathered.values())

    def served(self) -> list[Member]:
        """Every member as this entry RUNS it, with the reading `Serving` forces applied up front.

        The forced reading has to land HERE rather than only in `predict_member`, because a re-group
        moves the DECODE: `reads` keys on the members' own depth and band, so a group applied after the
        grouping would decode three times and then predict from one reading — paying the bill the group
        exists to remove. A size alone was safe either way (`reads` ignores resolution), which is why
        the late application was correct until the group joined it.
        """
        return [self.serving.read(member) for member in self.package.members]

    def _asked(self, members: list[Member] | None) -> list[Member]:
        """The subset a caller named, or the whole package when it named none — and NOT when it named zero.

        `members or self.package.members` reads the same until the subset is empty, and then it answers a
        question nobody asked: the decode fields of every member in the package, handed to a group that
        holds none of them. Both callers here say "derived, never defaulted" in their own docstrings, and
        a falsy default is exactly a default. `None` is the absence of an argument; `[]` is a subset that
        came out empty, which is a caller's bug and should surface as one — `max()` refuses an empty
        sequence and `agreed` refuses a set of zero distinct values, so the loud failure already exists
        and only the fallback was hiding it.
        """
        return self.served() if members is None else members

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
        return max(member.slices for member in self._asked(members))

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
        decoding = self._asked(members)
        names = self.agreed(decoding, "slot list", lambda member: tuple(member.config["slots"]))
        known = {slot.name: slot for scheme in SCHEMES.values() for slot in scheme}
        missing = [name for name in names if name not in known]
        if missing:
            message = f"members name slots this build does not define: {missing}"
            raise ValueError(message)
        tie = self.agreed(decoding, "tie rule", lambda member: member.tie)
        return SlotScheme(tuple(known[name] for name in names), tie=tie)

    def agreed(self, members: list[Member], what: str, decide: Callable[[Member], Decision]) -> Decision:
        """The one value every member decodes under, or a refusal — never a majority, never the first.

        A decode is shared: one stack is read, routed and bagged, then shown to every member. So a field
        that differs between members is not a disagreement to resolve, it is two decodes asked for at
        once, and picking either silently serves the other member's weights an input they were never
        fitted on. Refusing here is cheap; the same mismatch found by a score is a submission.
        """
        distinct = {decide(member) for member in members}
        if len(distinct) != 1:
            message = f"members disagree on the {what} ({len(distinct)} distinct) — that is two decodes, not one"
            raise ValueError(message)
        return distinct.pop()

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

        A FORCED precision goes through the same guard rather than around it. The point of forcing one
        is to measure what an arithmetic costs, and a bf16 arm that silently ran emulated on a T4 would
        be measuring the emulation — so the downgrade still applies and still says so.
        """
        asked = self.serving.precision or member.precision
        native = self.device.type == "cuda" and torch.cuda.get_device_capability() >= NATIVE_BF16
        if asked is torch.bfloat16 and self.device.type == "cuda" and not native:
            logger.warning("%s asked for bf16 and this device cannot run it — serving fp16", member.identifier)
            return torch.float16
        return asked

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
        member = self.serving.read(member)
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
        # fp32 IS THE DISABLED AUTOCAST, not an autocast to float32 — that combination raises, and the
        # non-finite fallback below has always spelled it the same way. Only a forced precision can
        # reach here as float32 (no member is fitted in it), so this costs a comparison to make the
        # reference arm of a precision measurement runnable at all.
        casting = self.device.type == "cuda" and precision is not torch.float32
        with torch.no_grad(), torch.autocast("cuda", dtype=precision, enabled=casting):
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

    def read_of(self, members: list[Member]) -> Read:
        """The one reading a group shares, asked of the group rather than taken from its first member.

        ASKED, not taken from member[0]. `reads` groups by the reading, so agreement holds by construction
        today — which is exactly the guarantee a later edit to the grouping key drops silently, serving the
        rest of the group under member[0]'s pixels. Every other field here is refused on disagreement.
        """
        return Read(
            slices=self.slices(members),
            scheme=self.decoded_scheme(members),
            sizes=tuple(sorted({member.image_size for member in members})),
            reading=self.agreed(members, "reading", lambda member: member.reading),
        )

    def budget(self) -> Budget | None:
        """The run budget, and NONE for a multi-group entry until the projection is repriced.

        The traversal is chunk-outer: every group advances together over one pass, so "group k of three" is
        no longer a fraction of the run and `afford` would project against a model of progress that no
        longer exists. Repricing the net inside the change that moved the traversal makes a bug in either
        show up as the other, so it is off for the multi-group case and repriced separately (knee-fma1r).
        A SINGLE-group entry is unaffected. This is a development state, not a shipping one —
        `submit_check` refuses a multi-group entry while this returns None.
        """
        return self.serving.budget if len(self.reads()) == 1 else None

    def predict_groups(
        self,
        chunks_for: Callable[[list[Read]], Iterable[list[SizedBag]]],
        *,
        overlap: bool = True,
    ) -> NDArray[np.float64]:
        """Predict every member over ONE pass of the corpus, and rank-combine across all groups.

        CHUNK-OUTER, GROUPS-INNER, and that is the whole point. Group-outer streamed the corpus once per
        read group, so the champion's three groups opened the same DICOM files in three traversals
        minutes apart — 204.9 opens a study against 105.6 distinct, 48.5% of them a second decode of a
        file already read. A memo cannot span traversals (the corpus's planes do not fit in memory) but
        spans a STUDY trivially: one chunk decoded for every read at once, one `FileMemo` per study.

        The decode is injected rather than performed here: this class knows which members can share pixels
        and how their outputs combine, and nothing about DICOM directories — the loop is six calls into
        this object and two into the caller's, which is what the feature-envy gate said, correctly.

        Combining across ALL groups at once matters: rank-averaging within a group and then averaging the
        group results would weight a group of one as heavily as a group of five.
        """
        groups = self.reads()
        budget = self.budget()
        if budget:
            budget.begin(len(groups))
        per_member: dict[str, list[NDArray[np.float64]]] = {
            member.identifier: [] for members in groups for member in members
        }
        failed: set[str] = set()
        for chunk in chunks_for([self.read_of(members) for members in groups]):
            for members, sized in zip(groups, chunk, strict=True):
                self.absorb(members, sized, per_member, failed, overlap=overlap)
            self.afford([member for members in groups for member in members], failed)
        combined = [array for members in groups for array in self.survived(members, per_member, failed)]
        if not combined:
            message = f"every member of this group failed: {sorted(failed)}"
            raise RuntimeError(message)
        return (self.serving.vote or RankAverage()).combine(combined)

    def predict_members(
        self,
        members: list[Member],
        chunks: Iterable[SizedBag],
        *,
        overlap: bool = True,
    ) -> list[NDArray[np.float64]]:
        """One array per member, over every chunk — UNCOMBINED, so groups can be combined together.

        A chunk arrives as {decoded size: (images, mask)} from ONE read, and each member is handed the
        size it was fitted from — which lets members of different resolutions share a decode without any
        of them being fed pixels it never saw. Uncombined, for the reason `predict_groups` gives.
        """
        per_member: dict[str, list[NDArray[np.float64]]] = {member.identifier: [] for member in members}
        failed: set[str] = set()
        for sized in chunks:
            self.absorb(members, sized, per_member, failed, overlap=overlap)
            self.afford(members, failed)
        arrays = self.survived(members, per_member, failed)
        if not arrays:
            message = f"every member of this group failed: {sorted(failed)}"
            raise RuntimeError(message)
        return arrays

    def absorb(
        self,
        members: list[Member],
        sized: SizedBag,
        per_member: dict[str, list[NDArray[np.float64]]],
        failed: set[str],
        *,
        overlap: bool = True,
    ) -> None:
        """ONE chunk through one group's members, appending each survivor's predictions.

        The unit of work both traversals are built from: `predict_members` loops it over the chunks of a
        single read, and `predict_groups` loops the groups inside one chunk. A member already in `failed`
        is skipped rather than retried — a member that raised once raises again, and the retries were
        measured costing more than the member was worth.
        """
        for member in members:
            if member.identifier in failed:
                continue
            predicted = self.attempt(member, sized, overlap=overlap)
            if predicted is None:
                failed.add(member.identifier)
            else:
                per_member[member.identifier].append(predicted)

    @staticmethod
    def survived(
        members: list[Member], per_member: dict[str, list[NDArray[np.float64]]], failed: set[str]
    ) -> list[NDArray[np.float64]]:
        """One concatenated array per member that both survived AND predicted something.

        Both conditions, because they are not the same: a member can be absent from `failed` and still
        hold nothing, when the run gave it up on the first chunk it was ever offered.
        """
        return [
            np.concatenate(per_member[member.identifier])
            for member in members
            if member.identifier not in failed and per_member[member.identifier]
        ]

    def attempt(self, member: Member, sized: SizedBag, *, overlap: bool) -> NDArray[np.float64] | None:
        """One member's pass over one chunk, timed — or None, because ONE MEMBER MUST NOT COST THE ENTRY.

        Three submissions died because any single failure anywhere took the whole run down, and a run
        that fails writes no submission at all. An ensemble is the one structure that does not need
        every part: dropping a member costs a little accuracy, dropping the entry costs a slot and a day.

        The seconds are banked only on the way out through success, since a member that raised part-way
        did not pay for a prediction and must not be charged for one — `afford` divides by this.
        """
        images, mask = sized[member.image_size]
        started = time.time()
        try:
            predicted = self.predict_member(member, images, mask, overlap=overlap)
        except Exception:
            logger.exception("member %s failed mid-prediction — dropping it from this entry", member.identifier)
            return None
        self.spent[member.identifier] = self.spent.get(member.identifier, 0.0) + time.time() - started
        return predicted

    def afford(self, members: list[Member], failed: set[str]) -> None:
        """Give up the weakest surviving member when the run is projected past its ceiling.

        DROPPING IS THE ONLY LEVER THAT WORKS MID-RUN. The decode is already shared and already
        overlapped with the forward; what remains is one forward per member per chunk, so members are
        the term the run can still change while it is running. Reused rather than reinvented: this is
        the same `failed` set a member that raises goes into, so a member given up here leaves by the
        path that is already tested — its partial predictions discarded, the entry written from the
        rest.

        BY SCORE PER SECOND, LOWEST FIRST — the ratio, not the score. What is being bought is TIME, and
        the members do not cost the same: a whole-stack 336 px member reads four times the depth of a
        224 px one, so it can cost several times as much per study. Ordering on holdout alone gives up
        the cheap weak members first, which is the order that saves the least wall clock per member lost
        — a run can shed three of them and still hit the ceiling, having thrown away three votes for
        nothing. Value is holdout above CHANCE because that is what a member contributes over having no
        member at all, and this pool's holdouts sit in a narrow band, so between two near-equal members
        the ratio gives up whichever is slower. Cost is MEASURED here, not predicted from the config:
        the thing that killed the runs is exactly the gap between what a member should cost and what it
        does.

        That is still not the member worth least to a RANK average, where a weak member that fails
        differently can be worth more than a strong one that agrees — this project's own ledger says
        diversity beats quality. A diversity order would need the members' predictions on shared
        studies, which is exactly the thing the run has no time left to compute.

        ONE PER CHUNK, not a recomputed set: the projection improves as the run goes, so giving up one
        member and re-measuring converges on the smallest set that finishes, instead of over-cutting
        from a single early estimate made when the pool spawn and the first decode are still in it.
        """
        budget = self.budget()
        if not budget or not budget.overrunning():
            return
        # VIEWS GO FIRST, AND THEY GO WHOLE. `kneemri-views` came back COMPLETE with a blank public
        # score on 2026-08-17: two members under rot3, the only entry tripling the forward, and the one
        # thing this mitigation could not reach. Dropping a member of two saves at most a factor of two
        # and turns the entry into a different entry; dropping rot3 saves exactly the factor the views
        # cost — here three — and leaves the pair that scored 0.919 with the identity view alone.
        # Ordering follows worth per second the same as members do: TTA has never been measured to buy
        # anything on this pool (knee-eeo priced overlap at 0.0000 for whole-stack members), so its
        # numerator is the smallest one on offer.
        if self.views is not None:
            logger.warning(
                "projecting %.1fh against a %.1fh ceiling — giving up test-time views (%s), "
                "which no member's holdout was measured with",
                budget.projected() / SECONDS_PER_HOUR,
                budget.seconds / SECONDS_PER_HOUR,
                self.views,
            )
            self.views = None
            return
        surviving = [member for member in members if member.identifier not in failed]
        if len(surviving) <= 1:
            return
        weakest = min(surviving, key=self.worth)
        failed.add(weakest.identifier)
        logger.warning(
            "projecting %.1fh against a %.1fh ceiling — giving up %s (holdout %.4f, %.1fs spent, "
            "worth %.2e per second), %d members left",
            budget.projected() / SECONDS_PER_HOUR,
            budget.seconds / SECONDS_PER_HOUR,
            weakest.identifier,
            weakest.holdout,
            # .get, because this line runs INSIDE the mitigation for runs that die: a member the loop
            # has not timed yet is exactly the case `worth` is built to handle, and a KeyError raised
            # while logging the give-up would take down the entry the give-up exists to save.
            self.spent.get(weakest.identifier, 0.0),
            self.worth(weakest),
            len(surviving) - 1,
        )

    def worth(self, member: Member) -> float:
        """Holdout above chance per second already spent on this member — what a second of it buys.

        Unmeasured members share the pool's mean cost rather than counting as free: a member the loop
        has not reached yet would otherwise look infinitely cheap and never be given up, which inverts
        the order on exactly the first chunk where the decision matters most.
        """
        spent = self.spent.get(member.identifier) or self.mean_cost()
        return (member.holdout - CHANCE) / spent if spent > 0 else float("inf")

    def mean_cost(self) -> float:
        """Seconds a member has cost on average so far, or zero before any member has been timed."""
        timed = [seconds for seconds in self.spent.values() if seconds > 0]
        return sum(timed) / len(timed) if timed else 0.0

    @staticmethod
    def submission(identifiers: list[str], predictions: NDArray[np.float64]) -> dict[str, list[object]]:
        """The submission table as plain columns: the study id, then one column per target in order.

        Static because it reads no ensemble state, only the target order. A
        test or an analysis that has predictions and identifiers should not have to construct a package
        and a dict of loaded models to lay them out as columns.
        """
        table: dict[str, list[object]] = {"StudyInstanceUID": list(identifiers)}
        for position, target in enumerate(TARGETS):
            table[target] = predictions[:, position].tolist()
        return table
