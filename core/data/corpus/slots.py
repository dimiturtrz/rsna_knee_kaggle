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

COR_FLUID_NOFS = Slot("COR_FLUID_NOFS", Plane.CORONAL, fluid_sensitive=True, fat_suppressed=False)
"""Coronal fluid-weighted without fat suppression — a standard meniscus sequence, and NOT a T1 slot.

Named rather than spelled inline because two axes now need the same predicate: it is the largest of the
three combinations the canonical six cannot express, and it is half of `FASTMRI_SLOTS`. The alternative
was `EXTENDED_SLOTS[6]`, which is a literal index into a tuple whose whole documented property is that
things get APPENDED to it.
"""

EXTENDED_SLOTS: tuple[Slot, ...] = (
    *SLOTS,
    COR_FLUID_NOFS,
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

MRNET_SCHEME = "mrnet"
"""The slot axis of the MRNet (Stanford) corpus — three planes, and therefore three slots.

Same safety argument as `RIJEKA_SCHEME`: a foreign corpus decoded under the canonical name writes onto
the arrays every member on disk was fitted on. The scheme makes the collision unrepresentable.
"""

MRNET_SLOTS: tuple[Slot, ...] = (SLOTS[0], SLOTS[2], SLOTS[4])
"""SAG_FLUID_FS, AX_FLUID_FS and COR_T1 — three of the canonical six, in canonical order.

Bien et al. 2018 (PLOS Medicine 15(11)) state the corpus's most common sequences: sagittal T2 with fat
saturation, coronal T1, axial PD with fat saturation. Those are exactly `SLOTS[0]`, `SLOTS[4]` and
`SLOTS[2]`'s predicates, so an MRNet exam occupies three of the positions it would occupy in our own
corpus and a trunk meets each sequence in the channel it already knows.

WHY THIS CORPUS IS WORTH MORE THAN RIJEKA'S ONE SLOT. A Rijeka exam fills one of six and leaves five
zero, so "a foreign corpus reads flat" and "a degenerate input reads flat" are the same measurement.
Three filled slots separates them.

THE SEQUENCES ARE MODAL, NOT GUARANTEED. The published description says most common, and the corpus
ships no per-exam sequence metadata to check it against — so a minority of exams sit in a slot whose
weighting predicate they do not satisfy. That is a known, stated impurity of this axis, not an
assumption being passed off as a fact.

Canonical ORDER is load-bearing: `MergedCache.positions` matches by slot NAME, and `_copy` assigns a
source block into `[0, 2, 4]` in this tuple's order.
"""

FASTMRI_SCHEME = "fastmri"
"""The slot axis of the fastMRI knee corpus — coronal proton density, fat-suppressed or not.

Same safety argument as `RIJEKA_SCHEME`: `Routing.suffix()` is empty for the canonical scheme, so a
foreign corpus decoded under it writes onto the arrays every member on disk was fitted on. The scheme
name makes that collision unrepresentable.
"""

FASTMRI_SLOTS: tuple[Slot, ...] = (SLOTS[1], COR_FLUID_NOFS)
"""COR_FLUID_FS and COR_FLUID_NOFS — the two acquisitions that corpus actually contains.

READ OFF THE FILES, not off a paper. Every exam carries an `acquisition` attribute reading either
`CORPDFS_FBK` or `CORPD_FBK`, split 131/119 over the first 250 volumes on disk. That is coronal proton
density with and without fat suppression, which is these two predicates exactly — so this is the one
foreign corpus whose sequence assignment is a recorded fact per exam rather than a modal claim about the
corpus, the stated impurity `MRNET_SLOTS` has to carry.

EXACTLY ONE SLOT IS FILLED PER EXAM, and that is a feature. A study is acquired fat-suppressed or not,
never both, so the presence mask over this corpus is genuinely per-study rather than all-ones. Rijeka
fills one of six and leaves five zero, which makes "a foreign corpus reads flat" and "a degenerate input
reads flat" the same measurement; here the mask varies within the corpus itself.

WHAT A CANONICAL MERGE COSTS. `MergedCache.positions` matches by slot NAME, and `COR_FLUID_NOFS` is not
among the canonical six — so merging this into a `merged` cache carries the fat-suppressed half and drops
the other. That is a real cost, stated rather than hidden, and it is not a reason to decode only half the
corpus: the decode is the expensive step, it runs once, and an extended-scheme merge would carry both.
"""

KMAR_SCHEME = "kmar"
"""The slot axis of the KMAR-50K corpus — five of the canonical six, ROUTED per volume.

Same safety argument as `RIJEKA_SCHEME`: `Routing.suffix()` is empty for the canonical scheme, so a
foreign corpus decoded under it writes onto the arrays every member on disk was fitted on. The scheme
name makes that collision unrepresentable. This is the FOURTH foreign corpus and the fourth chance at
that error.

IT IS THE ONLY CORPUS ON THIS MACHINE WHOSE LICENCE CLEARS. CC BY 4.0 (Mendeley, DOI
10.17632/xw7mrg7ntg.6). Rijeka is CC BY-NC-ND, MRNet's Stanford agreement bars adaptations, fastMRI bars
distributing derived files, and SKM-TEA's Stanford RUA is non-commercial and no-derivative-works. So a
member fitted here is the first foreign-data member this project could actually SHIP — which is the
whole reason the corpus is worth a decode.
"""

KMAR_SLOTS: tuple[Slot, ...] = (SLOTS[0], SLOTS[1], SLOTS[2], SLOTS[4], SLOTS[5])
"""SAG_FLUID_FS, COR_FLUID_FS, AX_FLUID_FS, COR_T1 and SAG_T1 — canonical five, in canonical order.

READ OFF THE COHORT TABLES, not off a paper, and not modally. Every volume carries a
`SeriesDescription` and ALL 1,339 paired volumes on disk join to one (measured 2026-09-12, 1339/1339) —
`pd_tse_fs_{tra,sag,cor}`, `t1_tse_{sag,cor}` and their `q`/`_p2`/`_256` variants. Proton density and
T2 are fluid-sensitive, `_fs` is fat suppression, and the plane word is in both the description and the
filename and agrees. So each volume's slot is a RECORDED FACT, the stated impurity `MRNET_SLOTS` has to
carry and this one does not.

Canonical ORDER is load-bearing: `MergedCache.positions` matches by slot NAME, so an ascending
subsequence of `SLOTS` is what lets a merge copy block i into the canonical position of the same name.

MOST STUDIES FILL ONE SLOT, and that is the honest caveat rather than a footnote. Grouped by subject:
918 of 1,101 fill exactly one, 152 fill two, 30 fill three, 1 fills four. So this corpus has Rijeka's
shape per study — five zero slots on the typical row — and a flat reading is ambiguous between "the
corpus does not transfer" and "one filled slot is a degenerate input", which belongs in an arm's
pre-registration rather than its post-mortem.

WHAT IS STILL DIFFERENT FROM RIJEKA, and it is what makes the ambiguity narrower here. Rijeka fills
`SLOTS[0]` and only ever `SLOTS[0]`, so its presence mask is a constant and carries no information at
all. Here the FILLED slot varies across the corpus — 434 axial fluid-FS, 359 sagittal fluid-FS, 296
coronal fluid-FS, 109 sagittal T1, 103 coronal T1 — so the mask varies within the corpus itself, the
property `FASTMRI_SLOTS` names as a feature.

THE THIRTEEN `COR_FLUID_NOFS` AND TWO `SAG_FLUID_NOFS` VOLUMES ARE DROPPED, along with four whose
description is a free-text scanner string and two reading `pd_tse_fs_con`. Nineteen of 1,339 is 1.4%,
they are named here so the denominator is never the published one by accident, and carrying them would
need `EXTENDED_SLOTS` — which no canonical merge could then copy.

THERE IS NO PATHOLOGY SUPERVISION HERE. The metadata JSON is `[]` and neither cohort table carries a
finding, so this scheme names an axis and a cache; what supervises an arm over it is a separate
question (`kmar_corpus` carries the argument for why the SSL-as-init answer is closed).
"""

MERGED_SCHEME = "merged"
"""Our corpus and a foreign one in ONE array — the canonical slot axis under a name of its own.

THE SLOTS ARE THE CANONICAL SIX, unchanged, so a run over a merged cache has the same head width, the
same slot attention and the same members as a run over ours: the arm is a normal supervised run, and its
control is that same recipe on our corpus alone. Only the study axis grows.

THE NAME EXISTS FOR THE COLLISION, exactly as `RIJEKA_SCHEME` does. A merged cache is canonical in every
respect the shape can express — same slots, same slices, same size — so under the canonical scheme it
would write `images_336px_12sl.npy`, the path every member on disk was fitted on, and differ from it only
in the length of axis 0. That is the one failure `StudyCache.key` records escaping three times. The
scheme name is what makes it unrepresentable here.
"""

TRIPLE_SCHEME = "merged3"
"""Our corpus and BOTH foreign ones in one array — the same canonical six slots again.

A SECOND MERGED NAME rather than a bigger `merged`, and the distinction is the whole point. Every
`merged_*` member on disk was fitted on the 5,143-study canonical+Rijeka cache, and that cache is a
published number here (`merged_span_cf` 0.8618, the second-best solo the project owns). Growing it in
place to 6,393 studies would leave those runs unreproducible with every gate still green — the exact
shape of the failure `MERGED_SCHEME`'s own docstring records. The two caches coexist instead, and the
2,000-study arm is a control for the 6,393-study one.

IT IS BUILT BY MERGING A MERGE, which `build_merged` allows only because the destination is named
separately from both sources. `MergedCache` concatenates a base and an extra along the study axis and
cares about neither's provenance, so canonical+Rijeka is an ordinary base — but a run whose destination
was ALSO its base would grow the same file on every invocation, and that is what stays refused.
"""

FASTMRI_MERGED_SCHEME = "mergedf"
"""Our corpus and fastMRI in one array — the canonical six slots. TRAINING ONLY; NOT SHIPPABLE.

**A MEMBER FITTED HERE MAY NOT BE SUBMITTED.** This docstring said the opposite until 2026-09-10 and it
was wrong on both halves of its claim, which was written from prose and never checked against the
agreement. The primary text was read that evening, an hour before a slot would have been spent on an
entry it licensed. It says `"non-exclusive, royalty-free license to use the fastMRI Dataset for internal
research or educational purposes only"` and, explicitly, not to `"DISTRIBUTE ... any subsequent variables
or data files derived from the fastMRI Dataset to any person or entity"` — distribution is permitted only
`"for uses in academic publications and presentations with citations"`. So there IS a no-derivative-
distribution clause, and algorithm development is NOT written in by name; it is covered only inasmuch as
it is internal research. Fitting a member here is internal research and stays legal. Uploading that
member's weights to a Kaggle dataset and running them in a competition kernel is distributing a
derivative to a person or entity, and it is not an academic publication.

WHAT THAT COSTS, STATED PLAINLY: NO merged member of any name can be shipped. Rijeka's bar is its ND
clause ALONE — the NC half was retired on 2026-09-13 when the host wrote that "readily accessible
datasets with non-commercial use restrictions are generally acceptable... since receiving prize money
does not constitute commercial exploitation" (discussion 733965); see
`research/2026-09-13_host_rulings_forum_sweep.md`, which holds every host ruling in one place. MRNet's
Stanford agreement bars creating an adaptation, so `merged` and `merged3` were already blocked; this
name was the exemption, and the exemption does not exist. `merged_span_cf` 0.8618 and every other
`merged_*` number in this repo is an OFFLINE result about what more data does, and only that. The merge
remains worth building for exactly that reason — the site axis is the largest measured lever the project
has — but the board is not where its answer is read.

WHY A THIRD MERGED NAME. `merged` is canonical+Rijeka and `merged3` adds MRNet. It cannot be built
into either existing name for the reason `TRIPLE_SCHEME` already records — every `merged_*` member on disk
was fitted on the 5,143-study cache, and growing it in place makes those runs unreproducible with every
gate still green.

IT IS A SECOND SITE, NOT A FOURTH. `merged3_corpus` measured the site axis at +0.0043 for the second site
and -0.0024 for the third, which closes fastMRI-as-a-fourth-site before a card is spent on it. This is the
corner that PAID: 4,407 + 974 = 5,381 studies over two sites, the same operation the positive point was
measured on.

WHAT IT COSTS, STATED BEFORE THE RUN. `MergedCache.positions` matches slots by NAME, so of
`FASTMRI_SLOTS` only `SLOTS[1]` is carried and `COR_FLUID_NOFS` is dropped — a fastMRI row fills one of
six slots and leaves five zero, and its mask says so. That is the same shape the Rijeka merge has, and
that merge produced `merged_span_cf` at 0.8618, so the pattern is not disqualifying on its own
measurement. It does mean a flat read here is ambiguous between "the corpus does not transfer" and "one
filled slot is too little input", which belongs in the arm's pre-registration rather than in its
post-mortem.
"""

KMAR_MERGED_SCHEME = "mergedk"
"""Our corpus and KMAR-50K in one array — the canonical six slots. THE ONLY MERGE THAT IS SHIPPABLE.

**THIS NAME EXISTS BECAUSE THE FIRST BUILD USED `merged3` AND THAT WAS WRONG.** On 2026-09-12 the
canonical+KMAR cache was written as `images_224px_12sl_merged3.npy`, which destroyed nothing — the
image size is in the filename and `merged3` on disk is 336 px — but it left ONE scheme name meaning
two different corpora depending on a size argument nobody would think to check. The arrays were
renamed rather than rebuilt (a rename is free; a rebuild is ~19 GB and `/d` holds 23 GB), and the
labels table moved with them.

LICENCE — AND THIS IS THE POINT OF THE WHOLE LANE. `FASTMRI_MERGED_SCHEME` records that NO merged
member of any name could be shipped: Rijeka is CC BY-NC-ND, MRNet's Stanford agreement bars an
adaptation, fastMRI bars distributing a derivative. **KMAR-50K is CC BY 4.0** (Mendeley DOI
10.17632/xw7mrg7ntg.6) — attribution and nothing else. A member fitted here may be packaged, mounted
and submitted. It is the only corpus on this machine of which that is true.

IT IS A SECOND SITE, WHICH IS THE CORNER THAT PAID. `merged3_corpus` measured the site axis at
**+0.0043 for the second site and -0.0024 for the third**, and that closure is what refuses fastMRI
as a fourth. 4,407 + 1,088 = 5,495 studies over two sites is the same operation the POSITIVE point
was measured on, not the one that closed.

AND ITS LABELS ARE A DIFFERENT MECHANISM FROM EVERY EARLIER MERGE. Rijeka supervises ACL alone and
MRNet three findings; KMAR ships no findings at all, and its rows are supervised by
`kneemri.inference.pseudo_labels` — our own champions' pooled belief over all twelve columns, weighted
by how much those champions agree (mean 0.6548 over the 1,088). So a flat read here is NOT the
site-count flat: the site axis added SPARSE human labels, this adds DENSE distilled ones, and
`experiments/README.md:7411` states in its own words that external data as LABELLED augmentation is
"a different mechanism and untested here".
"""

ROI_SCHEME = "roi"
"""The canonical six plus each SAGITTAL slot again, cropped to the box a tissue locator drew. OFFLINE ONLY.

The three views are APPENDED, never swapped in: `crop80_tissue` replaced the whole frame with a tight
one and read about -0.008, and the multi-scale scheme that kept it measured on a max-pooling head, which
the ledger has since downgraded to unread. What is new here is the CENTRE — learned from cartilage and
meniscus rather than taken from the image middle or an intensity mass.

`crop_mm` stays None because the crop is not a decode-time extent: the box is drawn per study by
`kneemri.embeddings.roi_cache` over the canonical cache's own pixels, so this name is FOREIGN to
`build_cache`. The locator was fitted on SKM-TEA, so a member fitted here is not shipped until the
licence ruling (knee-vgz5b.7)."""

ROI_SLOTS: tuple[Slot, ...] = (
    *SLOTS,
    *(replace(slot, name=f"{slot.name}@{ROI_SCHEME}") for slot in SLOTS if slot.plane is Plane.SAGITTAL),
)

ROI_NATIVE_SCHEME = "roinative"
"""The `roi` slots with each box re-cut from the DICOM decoded at twice the cache grid, not upsampled from it.

Same boxes, same frame, same slot order — only the pixels inside a box change. 73% of RSNA sagittal
series are acquired finer than the cache's 0.387 mm/px (PixelSpacing p50 0.312), so an 87 mm box holds
about 1.24x the linear detail the `roi` crop threw away. OFFLINE ONLY, for the same licence reason."""

SCHEMES: dict[str, tuple[Slot, ...]] = {
    CANONICAL_SCHEME: SLOTS,
    "extended": EXTENDED_SLOTS,
    MULTISCALE_SCHEME: MULTISCALE_SLOTS,
    RIJEKA_SCHEME: RIJEKA_SLOTS,
    MRNET_SCHEME: MRNET_SLOTS,
    FASTMRI_SCHEME: FASTMRI_SLOTS,
    KMAR_SCHEME: KMAR_SLOTS,
    MERGED_SCHEME: SLOTS,
    TRIPLE_SCHEME: SLOTS,
    FASTMRI_MERGED_SCHEME: SLOTS,
    KMAR_MERGED_SCHEME: SLOTS,
    ROI_SCHEME: ROI_SLOTS,
    ROI_NATIVE_SCHEME: ROI_SLOTS,
}
"""Selectable slot schemes. The canonical one is not merely a default: the 20 published members we
grafted are six-slot by construction, so it has to remain expressible for the graft to keep loading."""

FOREIGN_SCHEMES = (
    RIJEKA_SCHEME,
    MRNET_SCHEME,
    FASTMRI_SCHEME,
    KMAR_SCHEME,
    MERGED_SCHEME,
    TRIPLE_SCHEME,
    FASTMRI_MERGED_SCHEME,
    KMAR_MERGED_SCHEME,
    ROI_SCHEME,
    ROI_NATIVE_SCHEME,
)
"""Schemes that name a corpus THIS repo's own decoder cannot produce, each owned by its own builder.

WHY THIS TUPLE EXISTS. `--scheme` selects a FILENAME, not a corpus: `Routing.suffix()` appends it and
nothing else consults it, so `build_cache --scheme merged` decodes our own 4407 studies and writes them
to `images_336px_20sl_0.1-0.9band_merged.npy` — a file named for 5143 studies from two corpora. Shape
cannot catch it, because `SCHEMES[MERGED_SCHEME]` IS `SLOTS`: our corpus under the merged name has the
identical six-slot shape the real merged cache has, and differs only in the length of axis 0. That is
59.7 GB and an hour, spent on 2026-08-27 (knee-48br), and it is the same failure `RIJEKA_SCHEME`'s own
docstring already records escaping three times in the other direction.

Each of these belongs to a builder that reads a different source: `build_rijeka` for the kneeMRI archive,
`build_merged` for the concatenation. `build_cache` refuses all three rather than assuming."""


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
