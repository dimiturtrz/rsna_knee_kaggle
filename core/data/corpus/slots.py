"""Routing a study's series into the six acquisition slots the head attends over.

A knee is read in three planes because the structures run in different directions — cruciates obliquely
and best sagittally, collateral ligaments and the meniscal body coronally, patellar cartilage and the
retinacula axially. Crossing plane with the two acquisition axes gives six slots chosen so each of the
twelve findings has at least one sequence that shows it well.

Measured here on 150 random series: the shipped `Fat_Suppression` column matches recovered fat
suppression on 99.3% of them, but the shipped `Fluid_Sensitive` column matches recovered fluid
sensitivity on only 78.7% — and the non-fat-suppressed pool is 40 T1 against 32 PD/T2. Routing on the
shipped flags alone would therefore put fluid-weighted series in the T1 slots for a fifth of the corpus.
"""

from dataclasses import dataclass, replace
from enum import StrEnum

from core.data.corpus.acquisition import Acquisition


class Plane(StrEnum):
    """The three acquisition planes, spelled once.

    A StrEnum rather than a bare string because these are matched by VALUE in two places that fail
    quietly when the spelling is wrong: routing a series into a slot, and the anatomy table, where a
    mistyped plane produces a flat prior row for that finding rather than an error. The values are the
    ones `train_series.csv` carries in `Anatomical_Plane`, so comparison against raw ingest data still
    works without converting it.
    """

    SAGITTAL = "Sagittal"
    CORONAL = "Coronal"
    AXIAL = "Axial"


@dataclass(frozen=True)
class Slot:
    """One slot's predicate: a plane, a weighting axis, and a fat-suppression state."""

    name: str
    plane: str
    fluid_sensitive: bool
    fat_suppressed: bool
    crop_mm: float | None = None
    """The physical extent this slot is read at, or None for the reader's own.

    A slot is normally a SEQUENCE selector and this is empty. It is set only by a multi-scale scheme,
    where the same series is read twice — wide for context and tight for detail — and the two land in
    different slots so the head's existing slot attention pools over SCALE without a new mechanism.
    """

    def accepts(self, acquisition: Acquisition, plane: str) -> bool:
        """Whether this acquisition satisfies the slot's predicate exactly."""
        return (
            plane == self.plane
            and acquisition.fat_suppressed == self.fat_suppressed
            and acquisition.fluid_sensitive == self.fluid_sensitive
        )


SLOTS: tuple[Slot, ...] = (
    Slot("SAG_FLUID_FS", Plane.SAGITTAL, fluid_sensitive=True, fat_suppressed=True),
    Slot("COR_FLUID_FS", Plane.CORONAL, fluid_sensitive=True, fat_suppressed=True),
    Slot("AX_FLUID_FS", Plane.AXIAL, fluid_sensitive=True, fat_suppressed=True),
    Slot("SAG_FLUID_NOFS", Plane.SAGITTAL, fluid_sensitive=True, fat_suppressed=False),
    Slot("COR_T1", Plane.CORONAL, fluid_sensitive=False, fat_suppressed=False),
    Slot("SAG_T1", Plane.SAGITTAL, fluid_sensitive=False, fat_suppressed=False),
)
"""The order is load-bearing: it is the slot axis of every cached array and of the head's slot embedding."""

EXTENDED_SLOTS: tuple[Slot, ...] = (
    *SLOTS,
    Slot("COR_FLUID_NOFS", Plane.CORONAL, fluid_sensitive=True, fat_suppressed=False),
    Slot("AX_T1", Plane.AXIAL, fluid_sensitive=False, fat_suppressed=False),
    Slot("AX_FLUID_NOFS", Plane.AXIAL, fluid_sensitive=True, fat_suppressed=False),
)
"""The canonical six plus the three combinations they cannot express.

MEASURED over all 4,407 studies: 2,421 of 24,371 acquired series (9.9%) match no canonical slot, and
these three account for 2,335 of them — 96.4%, and 83,297 slices of imaging the model never sees.
None are localisers: coronal PD without fat suppression is a standard meniscus sequence and axial T1
reads patellar cartilage.

APPENDED, never inserted. The order is the slot axis of every cached array and of the head's slot
embedding, so the first six positions must keep meaning what they meant — that is what lets a
canonical member and an extended one share every other piece of code.
"""

FIRST_MAXIMAL = "first"
"""Keep the first of equally dense candidates, in the order they were probed. What every cache holds."""

LAST_MAXIMAL = "last"
"""Keep the LAST of equally dense candidates — what a sort by slice count ascending then `.iloc[-1]`
gives. The alternative worth testing: their rule was recorded here as a stable `sort_values`, which is
also a FIRST-maximal rule, so if the two pipelines diverge on contested slots it is because their rule
is not the one we assumed. For TESTING against the published members' recorded predictions, not for
switching on."""

TIES: tuple[str, ...] = (FIRST_MAXIMAL, LAST_MAXIMAL)
"""Selectable tie rules, by the name a cache and a manifest record.

A registry for the same reason `SCHEMES` and `CENTRES` are ones: it is the vocabulary a CLI offers, so
it is also the vocabulary a test can WALK. Spelled as a literal tuple at the argparse call site, the
tie was the one reading axis no test could enumerate, and a name accepted there but dropped from the
cache key is a decode that lands on the arrays every member on disk was fitted on."""

CANONICAL_SCHEME = "canonical"
"""The scheme whose caches keep their historic names, and the one the published members use.

THE ONE HOME for this name. It keys `SCHEMES`, it is what `Routing` compares against to decide whether
a cache key carries a suffix, and it is what a member claims when its slot list is the published six —
three jobs that must agree. They had three separate copies of the literal, so renaming the vocabulary
here would have left caches written under one name and looked for under another."""

DETAIL_MM = 80.0
"""The tight extent of the multi-scale scheme.

DERIVED, not chosen: the field of view is a median 160 mm and the canonical crop is 130 mm, so we spend
two thirds of every image outside the joint at 0.387 mm/px — while a meniscal tear is 1-2 mm and DINOv2
patches at 14 px, making a tear a fraction of ONE token. 80 mm at the same grid is 0.238 mm/px, 1.6x
finer for identical compute, and still wider than the 70-80 mm the joint itself spans (knee-1nn).
"""

MULTISCALE_SCHEME = "multiscale"
"""The scheme name that selects the two-extent slot axis, and the suffix its caches carry."""

MULTISCALE_SLOTS: tuple[Slot, ...] = (
    *SLOTS,
    *(replace(slot, name=f"{slot.name}@{DETAIL_MM:g}", crop_mm=DETAIL_MM) for slot in SLOTS),
)
"""The canonical six read wide, then the SAME six read tight — twelve slots over six sequences.

WHY THIS SHAPE. Both directions have already paid separately and a single scale must trade them against
each other: res392 (finer) is our best single member at 0.8523 and whole_stack (more context) is our
biggest arm at +0.0147. Two scales need not trade. The twelve findings do not want the same field of
view — measured against radiologist annotations, the model BEATS its own labels on the diffuse findings
and loses badly on the small ones (knee-dtk), a split by structure size.

APPENDED, never interleaved, for the same reason EXTENDED_SLOTS is: the first six positions keep meaning
what they meant, so a canonical member and a multi-scale one share every other piece of code.

ONE DECODE SERVES BOTH, because 80 mm is a subset of 130 mm: the tight crop is taken from the stack the
wide one already produced, so scale costs encoder passes and not I/O.
"""

RIJEKA_SCHEME = "rijeka"
"""The slot axis of the kneeMRI (Rijeka) corpus — one sequence, and therefore one slot.

WHY A FOREIGN CORPUS NEEDS A SCHEME NAME AT ALL, and it is a safety requirement rather than tidiness.
`StudyCache.key` is the only thing separating one decoded corpus from another on disk, and
`Routing.suffix()` is EMPTY for the canonical scheme — so a foreign corpus decoded at 336 px / 12 slices
under the canonical name writes to `images_336px_12sl.npy`, the path every member on disk was fitted on,
and destroys 71.6 GB with an array differing only in its FIRST axis. Naming the scheme makes that
collision unrepresentable, which is the accident `StudyCache.key`'s own docstring records escaping three
times on our own axes (slot count, physical crop, slice band). A second corpus is the fourth chance.
"""

RIJEKA_SLOTS: tuple[Slot, ...] = (SLOTS[0],)
"""SAG_FLUID_FS alone — the canonical first slot standing by itself, not a lookalike beside it.

Štajduhar et al. 2017 (Comput Methods Programs Biomed 140, 151-164) acquired that whole corpus on one
Siemens Avanto 1.5T, sagittally, by a proton-density-weighted fat-suppression technique. That is
`SLOTS[0]`'s predicate exactly, so an exam of theirs occupies the same position in the slot axis that it
would occupy in ours, and a pretrained encoder meets the same sequence in the same channel afterwards.

Nothing ROUTES into it: that corpus ships one volume per exam with no series to choose between, so
`SlotScheme.assign` is never asked. The scheme is here to name the axis and the cache, not to select.
"""

SCHEMES: dict[str, tuple[Slot, ...]] = {
    CANONICAL_SCHEME: SLOTS,
    "extended": EXTENDED_SLOTS,
    MULTISCALE_SCHEME: MULTISCALE_SLOTS,
    RIJEKA_SCHEME: RIJEKA_SLOTS,
}
"""Selectable slot schemes. The canonical one is not merely a default: the 20 published members we
grafted are six-slot by construction, so it has to remain expressible for the graft to keep loading."""


class SlotScheme:
    """Assigns each study's series to slots, leaving a slot empty when nothing matches its predicate."""

    def __init__(self, slots: tuple[Slot, ...] = SLOTS, tie: str = FIRST_MAXIMAL) -> None:
        self.slots = slots
        # WHICH of several equally dense candidates wins. Default is what every cache and member on
        # disk was built with; the alternatives exist to be TESTED against the published members'
        # recorded predictions, not to be switched on.
        self.tie = tie

    def assign(self, acquisitions: list[Acquisition], planes: dict[str, str]) -> dict[str, Acquisition]:
        """One acquisition per slot, ties broken toward the stack with the most slices.

        A denser stack samples the joint more finely and leaves the band sampler margin to work with.

        A slot with no matching series stays EMPTY and no substitute is admitted from a neighbouring
        predicate. Relaxing the weighting to fill a T1 slot would draw from the same pool
        `SAG_FLUID_NOFS` selects from, putting one series in two slots and leaving the T1 slot holding
        fluid-weighted images; the presence mask would then assert a sequence that was never acquired,
        and the per-diagnosis softmax would split its attention across two identical slots. The mask
        exists to say a slot is absent, which is what an absent slot is.
        """
        chosen: dict[str, Acquisition] = {}
        for slot in self.slots:
            candidates = [
                acquisition
                for acquisition in acquisitions
                if slot.accepts(acquisition, planes.get(acquisition.series, ""))
            ]
            if candidates:
                chosen[slot.name] = self.won(candidates)
        return chosen

    def won(self, candidates: list[Acquisition]) -> Acquisition:
        """Which of a slot's candidates fills it — the densest stack, and a rule for equal ones.

        THE TIE RULE IS A STRATEGY BECAUSE IT IS AN OPEN QUESTION, not because anything wants to vary
        it. Our residual against the published members correlates +0.664 with the number of contested
        slots a study has, and +0.496 controlling for series count — so the rule may be where their
        pipeline and ours part company. What was refuted on 2026-08-14 is that the two see candidates
        in a different ORDER; what was never checked is whether their rule is this one at all, because
        we hold their checkpoints and not their router.

        `first` is what every cache and every member on disk was built with and stays the default.
        """
        densest = max(candidate.slice_count for candidate in candidates)
        tied = [candidate for candidate in candidates if candidate.slice_count == densest]
        if self.tie == LAST_MAXIMAL:
            return tied[-1]
        return tied[0]

    def unrouted(self, acquisitions: list[Acquisition], planes: dict[str, str]) -> list[Acquisition]:
        """The series no slot accepts, which the model therefore never sees.

        Six slots cover three planes x two weighting axes only partly: there is no AX_T1, no
        AX_FLUID_NOFS and no COR_FLUID_NOFS. A study acquiring any of those loses them entirely, and
        `assign` is silent about it because a slot with no candidate simply stays empty.

        This is distinct from a slot LOSING a contest (`contested`): a contested slot still gets an
        image, just possibly a different one than another pipeline would pick. An unrouted series
        contributes nothing at all.
        """
        return [
            acquisition
            for acquisition in acquisitions
            if not any(slot.accepts(acquisition, planes.get(acquisition.series, "")) for slot in self.slots)
        ]

    def contested(self, acquisitions: list[Acquisition], planes: dict[str, str]) -> dict[str, int]:
        """Per slot, whether its winner was a TIE on slice count — 1 if contested, 0 if clear.

        `assign` breaks a tie with `max`, which returns the FIRST maximal element in list order. The
        published members were routed by a pandas `sort_values` that takes the first in FRAME order.
        Two different series in a slot means the head reads a different image, and nothing downstream
        recovers it — which is the leading explanation for our data layer scoring 0.888 against the
        public forks' ~0.899 on identical weights, and for the 0.025 out-of-fold residual.

        This counts how often the ambiguity ARISES. If ties are rare and the residual persists, the
        cause is elsewhere and the hypothesis dies cleanly, which is most of what it is worth.
        """
        contested = {}
        for slot in self.slots:
            counts = [
                acquisition.slice_count
                for acquisition in acquisitions
                if slot.accepts(acquisition, planes.get(acquisition.series, ""))
            ]
            contested[slot.name] = int(bool(counts) and counts.count(max(counts)) > 1)
        return contested

    def presence(self, chosen: dict[str, Acquisition]) -> list[float]:
        """The mask the head's softmax renormalises over: 1.0 where a slot was filled, 0.0 where not."""
        return [1.0 if slot.name in chosen else 0.0 for slot in self.slots]


@dataclass(frozen=True)
class Routing:
    """How a study's series become slots: which slots exist, and which candidate wins a tie.

    One object because the argument-count gate refused the second one separately, and because they are
    the same subject — `SlotScheme` already holds both. The scheme decides which series are ROUTED at
    all and the tie decides WHICH of the eligible ones fills a slot; either changes the pixels a member
    sees, so neither can be resolved silently when members disagree.

    Both defaults are what every cache and every member of ours on disk was built with.
    """

    scheme: str = CANONICAL_SCHEME
    tie: str = FIRST_MAXIMAL

    def suffix(self) -> str:
        """What this routing adds to a cache's name — empty for the historic one.

        Empty by design, for the same reason `Crop.suffix` is: those arrays are memory-mapped by running
        trainers and every member on disk was fitted on them.
        """
        named = (
            self.scheme if self.scheme != CANONICAL_SCHEME else "",
            self.tie if self.tie != FIRST_MAXIMAL else "",
        )
        return "".join(f"_{part}" for part in named if part)

    def built(self) -> "SlotScheme":
        """The scheme this routing names, carrying its tie rule."""
        return SlotScheme(SCHEMES[self.scheme], tie=self.tie)
