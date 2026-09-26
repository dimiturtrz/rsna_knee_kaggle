"""The fastMRI knee corpus with its fastMRI+ annotations, read into the arrays this trainer loads.

A THIRD CORPUS, and the first one whose labels are graded on the same axis ours are. See
`rijeka_corpus` for the three downstream mechanisms that already admit a foreign corpus without a line
of change (fold assignment on the identifier, `Targets.pair` weighting an absent column 0.0, and
`train.py` never reading the reindexed NaN rows), and `slots.FASTMRI_SCHEME` for why the scheme name is
a safety requirement rather than a label.

WHY THIS CORPUS AND NOT MORE ROWS OF ANY CORPUS. The only external corpus that ever moved our number is
`merged_corpus`, +0.0020 overall — but its per-finding breakdown moved **Lateral Meniscus +0.0196 on a
corpus that never labels the meniscus**. The gain was ENCODER-side and came from labels differing in
KIND, not from study count. Our own dose curve prices count directly (`half_data` 0.8344,
`quarter_data` 0.7907, concave), and 1,172 studies against our 4,407 is a quarter-doubling worth roughly
+0.002. So this corpus is worth building for what its labels ARE, and it should not be sold as volume.

A ROW IS A KNEE, AND IT WAS A FILE UNTIL 2026-09-16 (knee-0i7v5). The corpus ships 1,172 routed h5 over
556 `patient_id`, 547 of which carry BOTH acquisitions — so a cache built per file held 1,172 rows with
exactly one of two slots filled, where 547 two-slot studies existed. `FastmriStudy` is that grouping, and
it is also what any fold over this corpus must be taken on: by file, the same knee lands on both sides of
every split. EVERY RATE IN THE PARAGRAPHS BELOW IS STILL PER FILE — they are rates over what the READERS
read, which is files, and re-denominating them onto knees is a measurement nobody has taken.

EVERY RATE BELOW IS OVER 1,173 READ STUDIES. That number was 1,172 in the first draft of this file,
because `knee_file_list.csv` ships WITHOUT A HEADER and a default `read_csv` silently promotes
`file1000000` — a real study — into the column name. One row, and it is the difference between a rate
and a rate that is wrong; the reader takes `header=None` for exactly this reason.

WHAT IS DIFFERENT IN KIND HERE, stated so it can be checked rather than assumed:

  * THE PLANE. Rijeka is sagittal PD-FS; this is CORONAL, both fat-suppressed and not. The collateral
    ligaments and the meniscal body are coronal structures. A trunk that has only ever been supervised
    on the ACL sagittally meets MCL and meniscus supervision in the plane they are read in.
  * THE GRADING. fastMRI+ separates `ACL High Grade Sprain` from `ACL Low Grade sprain` and
    `Cartilage - Full Thickness` from `Partial Thickness`. Our twelve criteria are themselves graded
    thresholds — high-grade, greater than 50 percent, moderate or large — and `findings.SPECIFICITY`
    says borderline reads negative. So their low grade is not missing data on our scale: it is a
    radiologist stating the finding sits BELOW our threshold, which is a confident negative.
  * THE GEOMETRY IS STATED, NOT DERIVED. `RIJEKA_SPACING` had to be argued from field-of-view priors
    because that corpus records no spacing. Every file here carries an ISMRMRD header giving the recon
    field of view outright: 140 mm on 116 of 120 sampled files, 150 mm on the other 4, over a 320 px
    matrix. Spacing is READ per exam.

WHAT IT MOSTLY SUPERVISES IS ABSENCE, and that is the honest headline. fastMRI+ names findings we
cannot locate to a compartment — a `Meniscus Tear` is not sided, `Cartilage - Full Thickness` names no
compartment — so a POSITIVE of that kind cannot fill either of our two meniscus columns or any of our
three OA columns. It still tells us something the corpus can carry: a study the reader annotated and
did NOT box for meniscal tearing is a meniscus-negative in both compartments. Measured over the 1,172
studies in `knee_file_list.csv`: 56.5% carry a meniscus tear and the remaining 43.5% are supervised
negative on Medial and Lateral Meniscus both; 10.4% carry full-thickness cartilage loss and the other
89.6% are supervised negative on all three OA columns.

Only ACL and MCL get supervision on BOTH sides, because only they are graded against a threshold we
share. ACL: 8.6% high grade (positive), 13.0% low grade (an assessed negative), the rest silent. MCL:
0.3% high grade, 10.3% low-to-moderate.

THE HAZARD THIS CREATES, named because it is the thing most likely to make the arm read negative. Nine
of our twelve columns receive negatives from this corpus and almost no positives, so merging it shifts
the prior toward absence on exactly the findings it cannot locate. The control for any arm built on it
is that same recipe on our corpus alone, and a per-finding breakdown is not optional — an overall flat
made of "meniscus better, OA worse" is a different result from an overall flat made of nothing moving.

THE DENOMINATOR IS `knee_file_list.csv`, 1,173 studies, NOT the 974 files that carry at least one box.
The 199 difference were read and found clean, so scoring rates over the annotated files alone would
report 68.1% meniscus tear where the truth over read studies is 56.5%. A study absent from the file
list was never read at all and receives no label row of any kind.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from jaxtyping import Float

from core.data.corpus.fastmri_annotations import FastmriAnnotations
from core.data.corpus.foreign_build import ForeignBuild, SlotVolume
from core.data.corpus.study_cache import StudyCache
from core.data.corpus.study_metadata import StudyMetadata
from core.findings import STUDY, TARGETS

logger = logging.getLogger(__name__)

VOLUME_SUFFIX = ".h5"
RECONSTRUCTION = "reconstruction_esc"
"""The emulated-single-coil reconstruction, which is what fastMRI+ was annotated on.

`reconstruction_rss` sits beside it in every file and is the root-sum-of-squares of the multi-coil data.
The two differ, and the annotation protocol converted the singlecoil reconstruction to DICOM for reading
(`ExampleScripts/fastmri_to_dicom.py`), so the boxes belong to this array. Reading the other one would
put the labels on images they were not drawn against.
"""

HEADER = "ismrmrd_header"
ACQUISITION = "acquisition"

PATIENT = "patient_id"
"""The h5 ATTRIBUTE that says which knee an exam belongs to — not a header field.

`<patientID>` does not appear in the ISMRMRD header of any of the 1,172 routed files; the identity lives
beside `acquisition` in `h5py.File(...).attrs` as a 64-hex hash. A reader that looks for it in the XML
finds nothing on every file and reads that as "the corpus has no patient identity", which is how this
corpus spent its first build being 1,172 half-knees.
"""

FAT_SUPPRESSED_ACQUISITION = "CORPDFS_FBK"
"""Coronal proton-density with fat suppression — the corpus's own name for it, matched exactly.

Matched by equality rather than by an `FS` substring: `CORPD_FBK` contains no `FS`, but a substring rule
is the kind that survives review and then quietly admits a future acquisition string nobody checked.
"""

FASTMRI_CORPUS = "fastmri"
"""What this corpus calls itself in the `corpus` column."""

FASTMRI_PLANE = "Coronal"
"""Every routed exam here is coronal — both acquisitions this scheme knows say so in their own name, and
that plane is the reason the corpus is worth building (see this module's docstring)."""

FASTMRI_SEQUENCE = "PD"
"""Proton density, fat-suppressed or not. Read off the acquisition string rather than assumed: an exam on
neither known string is skipped by `_slot` and never reaches a metadata row."""

FAT_SUPPRESSED_SLOT = 0
PLAIN_SLOT = 1
"""Which cache slot each acquisition fills. Named because `metadata` reads fat suppression BACK off the
slot, so the two would drift silently if the mapping lived only as a literal inside `_slot`."""

PLAIN_ACQUISITION = "CORPD_FBK"
"""Coronal proton-density without fat suppression.

Fluid-sensitive all the same, which is why it lands in `COR_FLUID_NOFS` rather than a T1 slot — see that
slot's own note in `slots.EXTENDED_SLOTS`, where coronal PD without fat suppression is called a standard
meniscus sequence. Getting this wrong would put half this corpus in a channel meant for T1 anatomy.
"""

FOV_PATTERN = re.compile(
    r"<reconSpace>.*?<matrixSize>.*?<x>(\d+)</x>.*?<fieldOfView_mm>.*?<x>([\d.]+)</x>",
    re.DOTALL,
)
"""In-plane matrix size and field of view out of the ISMRMRD header, IN THAT ORDER.

The order is not a style choice and getting it backwards is a silent no-match: ISMRMRD writes
`matrixSize` before `fieldOfView_mm` inside each space, so a pattern that asks for the field of view
first matches nothing at all. It read that way here first and refused every file in the corpus.

`reconSpace` rather than `encodedSpace`: the encoded space is the acquisition grid (640 x 372 here) and
the reconstruction is 320 x 320, so the encoded field of view would give a spacing for pixels that are
not the ones in `reconstruction_esc`. `reconSpace` follows `encodedSpace` in the document, so anchoring
on it cannot capture the wrong one.
"""

POSITIVE_LEVEL = 0.82
"""What a boxed finding that meets our criterion is worth on the level scale.

`Supervision.MEASURED_AGREEMENT` fixes what a level MEANS by what an RSNA radiologist would call: 0.08 ->
4.1% present, 0.28 -> 13.6%, 0.68 -> 37.8%, 0.82 -> 68.8%, 0.94 -> 95.8%. So the question to ask of a
fastMRI+ box is not how certain it looks but how often a second reader would agree.

0.82 rather than Rijeka's 0.94 for a confirmed injury, and the gap is the protocol. Rijeka's verdicts are
DOUBLE-BLIND reads by diagnostic radiologists. These are, in the corpus README's own words, labelled "by
a single radiologist and without the benefit of looking at other views and angles of the same subject",
and "should not be considered clinical ground truth ... but rather an indication of where a pathology
could be present". One reader, one view, indicative by declaration: 68.8% agreement is the level that
describes that, and 95.8% would be a claim the corpus explicitly disowns.
"""

NEGATIVE_LEVEL = 0.28
"""What an assessed absence is worth — the reader read this study and did not box this finding.

A COLLAPSE THAT WAS CONSIDERED AND REJECTED AS UNMEASURABLE. Two absences arrive here and they are not
obviously equal: an explicit low grade (`ACL Low Grade sprain` against our high-grade criterion) is a
structure the reader assessed and graded below our threshold, while silence is silence under a protocol
that disclaims exhaustiveness. The first is plainly the stronger negative. Nothing here measures how
much stronger, and inventing a second constant to express it would be a magic number wearing an
argument, so both take this value and the distinction is recorded rather than fitted.

0.28 and not 0.08: 0.08 is the level Rijeka's double-blind read of an intact ligament earns, and this
corpus disclaims exactly the exhaustiveness that would justify it.
"""

READER_CONFIDENCE = 0.6
"""The weight every row of this corpus carries — A STATED PRIOR, NOT A MEASUREMENT.

Rijeka rows carry 1.0 because each is a double-blind expert read. These are single-reader, single-view,
and self-declared indicative, which is two independent degradations of that protocol; 0.6 says the rows
count for meaningfully less than an expert consensus and meaningfully more than a text extraction.

NO PUBLISHED INTER-READER AGREEMENT FOR fastMRI+ IS KNOWN TO THIS REPO, and until one is, the number is
a judgement. It is exposed as `--reader-confidence` for that reason, and the sensitivity of any arm to
it is untested. Do not quote it as though it were fitted.
"""


@dataclass(frozen=True)
class Reading:
    """How one of our twelve findings reads the fastMRI+ vocabulary.

    Two tuples rather than three, and the missing third is the point. A vocabulary term that is evidence
    the finding sits BELOW our threshold — `ACL Low Grade sprain`, `Cartilage - Partial Thickness` — needs
    no listing at all, because the outcome it produces is exactly the outcome silence produces. Giving it
    a field of its own would have been a field nothing reads.
    """

    positive: tuple[str, ...] = ()
    """Terms that satisfy our criterion for this finding outright."""

    coarse: tuple[str, ...] = ()
    """Terms that prove a finding is PRESENT but cannot locate it to this column.

    These block the negative instead of asserting a positive. `Meniscus Tear` carries no side, so a study
    with one is unknown on Medial Meniscus and unknown on Lateral Meniscus — not positive on both, which
    would assert two lesions where one was read, and not negative on both, which would be false.
    """

    compartmental: bool = False
    """Whether this column is one half of a `core.findings.COMPARTMENT_PAIRS` pair.

    A pair is two columns describing ONE finding that differ only in where it sits, so a coarse term that
    is boxed in BOTH compartments of a coronal exam has located itself in both without anyone having to
    say which compartment is which. That is the whole trick, and it is why this needs no laterality: the
    rule is symmetric under swapping the sides, so it survives not knowing whether the knee is a left or a
    right one — which nothing in fastMRI states (no laterality in the ISMRMRD header, no side column in
    the file list, and a knee coil that is the same coil either way).

    It buys the meniscus pair 190 studies and therefore 380 positive cells, in two columns that held
    ZERO. It is deliberately NOT claimed for the cartilage pair, where the same rule fires on 15 studies —
    the n=4 MCL number this module already records as unusable, one decimal place up. And `PF OA` is not
    compartmental because it is in no pair: the patellofemoral joint has no contralateral half to straddle.
    """

    def verdict(self, seen: frozenset[str], straddling: frozenset[str] = frozenset()) -> float | None:
        """This finding's level for a study whose boxed terms are `seen`, or None to leave it unlabelled.

        None is the mechanism, not a gap. `Targets.pair` gives an absent column weight 0.0 and score 0.5,
        so a finding this corpus cannot speak to contributes no gradient at all — which is the whole
        reason a one-column corpus like Rijeka's is usable beside a twelve-column one.

        `straddling` is the subset of `seen` whose boxes fall on BOTH sides of `COMPARTMENT_TROUGH`. A
        compartmental column reads that as a positive, because a finding boxed in both compartments is
        present in this one whichever compartment this one turns out to be. A term seen on one side only
        keeps blocking the negative exactly as before — it proves the finding is somewhere and cannot say
        where, which is the case `coarse` was written for.
        """
        if any(term in seen for term in self.positive):
            return POSITIVE_LEVEL
        if self.compartmental and any(term in straddling for term in self.coarse):
            return POSITIVE_LEVEL
        if any(term in seen for term in self.coarse):
            return None
        return NEGATIVE_LEVEL


MENISCUS_TERMS = ("Meniscus Tear", "Displaced Meniscal Tissue")
"""Meniscal pathology of any side. Displaced tissue is a fragment of a tear, so it carries the same block."""

CARTILAGE_LOSS = ("Cartilage - Full Thickness loss/defect",)
"""Full-thickness loss meets our greater-than-50-percent bar; partial does not and is deliberately absent.

Our OA criteria also require a moderate or large AREA, roughly 1 cm or greater. fastMRI+ boxes carry a
width and height and the spacing is known, so that threshold is computable — and it is not computed here.
A size cut chosen without a measurement to set it against is a fitted constant, and this corpus is being
added to supply labels rather than to have thresholds tuned on it.
"""

EFFUSION_TERMS = ("Joint Effusion ",)
"""The one blocked term that asserts rather than blocks, because effusion has neither a side nor a
compartment: a box labelled it IS the finding, in the only column no localisation is owed to.

The bar it has to clear is SIZE, and the boxes clear it by a distance no threshold has to adjudicate.
Across the 142 studies carrying one, the median box is 36.7 mm on a side and the lower QUARTILE is
31 mm — measured at 0.4375 mm/px, the 140 mm FOV on a 320 px matrix the ISMRMRD header reports. A cut
anywhere in the moderate range keeps the whole distribution, so there is no fitted constant here to
choose; that is exactly what `CARTILAGE_LOSS` cannot say about its own 9.6 mm median.
"""

BONE_INJURY = ("Bone-Fracture/Contusion/dislocation",)
"""One fastMRI+ term spanning two of our columns, so it can only ever block their negatives.

Our `Contusion` is marrow oedema from impact without a discrete fracture line and our `Fracture` is the
cortical break — the distinction this term merges. Splitting it would be guessing which of the two a
radiologist meant.
"""

READINGS: dict[str, Reading] = {
    "ACL": Reading(positive=("Ligament - ACL High Grade Sprain",)),
    "MCL": Reading(positive=("Ligament - MCL High Grade sprain",)),
    "Medial Meniscus": Reading(coarse=MENISCUS_TERMS, compartmental=True),
    "Lateral Meniscus": Reading(coarse=MENISCUS_TERMS, compartmental=True),
    "Medial OA": Reading(coarse=CARTILAGE_LOSS),
    "Lateral OA": Reading(coarse=CARTILAGE_LOSS),
    "PF OA": Reading(coarse=CARTILAGE_LOSS),
    "Effusion": Reading(positive=EFFUSION_TERMS),
    "Baker's": Reading(coarse=("Periarticular cysts",)),
    "Contusion": Reading(coarse=(*BONE_INJURY, "Bone- Subchondral edema")),
    "Fracture": Reading(coarse=BONE_INJURY),
}
"""Our twelve findings against the fastMRI+ vocabulary. Eleven appear; `Synovitis` does not.

SYNOVITIS IS ABSENT ON PURPOSE. The fastMRI+ vocabulary has no synovial term at all, so its readers were
never asked the question. Writing a negative there would teach the model that every fastMRI knee has a
quiet synovium on the authority of a reader who was not looking, and writing 0.5 would teach it that they
are all exactly average. Omitting the key leaves the column weightless, which is what not-asked means.

`Joint Effusion ` keeps its trailing space: that is the literal string in `Annotations/knee.csv`, and
stripping it here while the CSV is read raw would silently match nothing. Its own term is POSITIVE, not
coarse — `EFFUSION_TERMS`' docstring argues that case with the box-size measurement that justifies it
(median 36.7 mm, lower quartile 31 mm over the 142), and it is the one blocked term that asserts rather
than blocks. This paragraph said COARSE until 2026-09-08 while the map one line up read
`Reading(positive=EFFUSION_TERMS)`; the code and the constant agreed with each other and only the prose
dissented. It cost nothing that runs, and it would have cost a census its second positive column — 142
studies, the largest positive class this corpus has (D177 §4).

Terms not named anywhere above are ignored by construction: `Bone - Lesion`, `Soft Tissue Lesion`,
`Muscle Strain`, `Joint Bodies`, `artifact`, the PCL and LCL sprains, and the patellar retinaculum. None
of them maps onto any of our twelve, and the two grades our criteria explicitly call negative — ACL low
grade and partial-thickness cartilage — are absent for the reason `Reading` records.
"""


@dataclass(frozen=True)
class FastmriVolume:
    """One exam: where its pixels are, which slot its acquisition fills, and the geometry it was recon'd on.

    The matrix and the field of view are kept rather than only their quotient, because they are what the
    header actually STATES and the spacing is what we compute from them. A metadata row that recorded the
    quotient alone could not say whether an exam was one of the four in a hundred acquired at 150 mm.
    """

    identifier: str
    path: Path
    slot: int
    matrix_px: int
    field_of_view_mm: float
    patient: str = ""

    @property
    def spacing(self) -> float:
        """Millimetres per pixel in plane — the stated field of view over the stated matrix."""
        return self.field_of_view_mm / self.matrix_px

    def pixels(self) -> Float[np.ndarray, "depth rows columns"]:
        """The `RECONSTRUCTION` array as float32 — the one the boxes were drawn against."""
        with h5py.File(self.path, "r") as handle:
            return np.asarray(handle[RECONSTRUCTION][:], dtype=np.float32)


@dataclass(frozen=True)
class FastmriStudy:
    """One KNEE: every routed exam sharing a `patient_id`, at most one per slot.

    THE CORPUS IS 556 KNEES, NOT 1,172 EXAMS, and the difference is the whole reason this class exists.
    Measured over the routed files: 556 distinct `patient_id`, of which **547 carry BOTH `CORPD_FBK` and
    `CORPDFS_FBK`** — files per patient {1: 8, 2: 513, 3: 5, 4: 28, 5: 1, 6: 1}, and no patient spans
    `singlecoil_train` and `singlecoil_val`. Built one row per FILE, every cache row has exactly one of
    two slots filled and the corpus reads as 1,172 half-studies where 547 two-slot studies exist — the
    same shape our own corpus has, thrown away at the last step.

    Two consequences beyond the input. A split or holdout taken over fastMRI BY FILE puts the same knee
    on both sides — grouping is the fix and it is this key. And the 36 patients carrying a REPEATED
    acquisition are two reads of one knee: the second is kept in `spare` and counted rather than dropped
    silently, because an exam nobody can see is an exam nobody can price.
    """

    patient: str
    volumes: tuple[FastmriVolume, ...]
    spare: tuple[FastmriVolume, ...] = ()

    @property
    def identifier(self) -> str:
        """What the cache index and the labels table call this study — the knee, not one of its files."""
        return self.patient

    @staticmethod
    def grouped(volumes: list[FastmriVolume]) -> list["FastmriStudy"]:
        """Routed exams folded into knees, in first-appearance order, one volume per slot.

        The KEPT volume for a slot is the first in the order `volumes` produced, which is filename order —
        deterministic, and the same choice on every rebuild. A patient with no `patient_id` at all would
        collapse the whole corpus into one study, so an empty key keeps the file's own identifier.
        """
        held: dict[str, dict[int, FastmriVolume]] = {}
        spare: dict[str, list[FastmriVolume]] = {}
        for volume in volumes:
            key = volume.patient or volume.identifier
            slots = held.setdefault(key, {})
            if volume.slot in slots:
                spare.setdefault(key, []).append(volume)
                continue
            slots[volume.slot] = volume
        studies = [
            FastmriStudy(key, tuple(slots[slot] for slot in sorted(slots)), tuple(spare.get(key, ())))
            for key, slots in held.items()
        ]
        paired = sum(1 for study in studies if len(study.volumes) > 1)
        repeated = sum(len(study.spare) for study in studies)
        logger.info(
            f"{len(volumes)} exams -> {len(studies)} knees, {paired} filling both slots, {repeated} repeat reads"
        )
        return studies

    def files(self) -> tuple[str, ...]:
        """The identifiers of the exams this knee was built from, kept and spare alike.

        The fastMRI+ annotations are per FILE and stay that way, so this is the join between a box table
        that reads exams and a cache row that holds a knee.
        """
        return tuple(volume.identifier for volume in (*self.volumes, *self.spare))

    def fat_suppressed(self) -> bool | None:
        """Whether one word describes this knee's fat suppression, or None when both answers are true.

        A paired knee is acquired BOTH ways, and `StudyMetadata` has no set spelling for a bool — so the
        honest cell is absent, and the per-slot truth lives where it is now representable: the presence
        mask, whose two slots are fat-suppressed and plain by construction.
        """
        answers = {volume.slot == FAT_SUPPRESSED_SLOT for volume in self.volumes}
        return answers.pop() if len(answers) == 1 else None

    def geometry(self) -> FastmriVolume:
        """The exam whose stated geometry describes this row — the lowest filled slot.

        Every volume is resized with its OWN spacing before it reaches the cache, so this cell is
        descriptive rather than load-bearing; naming which exam it came from is what keeps it readable.
        """
        return self.volumes[0]


class FastmriCorpus:
    """Reads the fastMRI knee volumes into a `StudyCache`-shaped cache, with a labels table beside it.

    The files are read in place. They are HDF5 rather than DICOM, so this does not go through
    `StudyCache.build` for the same reason `RijekaCorpus` does not — but it shares the part that decides
    whether transfer means anything: the physical crop, the band sampling and the percentile intensity
    scaling are all `SlotReader`'s own, so both corpora reach the encoder in one geometry and one
    intensity space rather than merely in one array shape.
    """

    def __init__(self, root: Path, annotations: Path, cache: StudyCache, confidence: float = READER_CONFIDENCE) -> None:
        self.root = root
        self.annotations = annotations
        self.cache = cache
        self.confidence = confidence

    def volumes(self) -> list[FastmriVolume]:
        """Every readable exam under the root, in filename order, with its slot and spacing resolved.

        THE JOIN IS AGAINST WHAT IS ON DISK, not against the published file list. The corpus arrives as
        two archives and the larger one extracts for hours, so a build started early sees a subset — and
        a subset is a legitimate thing to decode, but only if the count it reports is the count it read.
        An exam whose acquisition string is neither known one is skipped and counted, never guessed into
        a slot.
        """
        found = sorted(self.root.rglob(f"*{VOLUME_SUFFIX}"))
        volumes = []
        skipped = 0
        for path in found:
            routed = self._routed(path)
            if routed is None:
                skipped += 1
                continue
            slot, patient = routed
            matrix, field_of_view = self._geometry(path)
            volumes.append(FastmriVolume(path.stem, path, slot, matrix, field_of_view, patient))
        logger.info(f"{len(volumes)} of {len(found)} exams routed, {skipped} on an unknown acquisition")
        return volumes

    def studies(self) -> list[FastmriStudy]:
        """The routed exams as KNEES — what a cache row, a labels row and a fold assignment are each about."""
        return FastmriStudy.grouped(self.volumes())

    def _routed(self, path: Path) -> tuple[int, str] | None:
        """This exam's slot and the knee it belongs to, or None if its acquisition is not one we route.

        Both come out of ONE open. They are two attributes of the same file and the geometry read below
        already costs a second one; asking three times for what one handle holds is three times the
        filesystem for the same corpus.
        """
        with h5py.File(path, "r") as handle:
            acquisition = str(handle.attrs.get(ACQUISITION, ""))
            patient = str(handle.attrs.get(PATIENT, ""))
        if acquisition == FAT_SUPPRESSED_ACQUISITION:
            return FAT_SUPPRESSED_SLOT, patient
        if acquisition == PLAIN_ACQUISITION:
            return PLAIN_SLOT, patient
        return None

    def _geometry(self, path: Path) -> tuple[int, float]:
        """The recon matrix and field of view, read from this exam's own ISMRMRD header.

        Read rather than assumed, which is the one place this corpus is on firmer ground than Rijeka:
        140 mm over 320 px on 116 of 120 sampled files and 150 mm on the other 4, so a single constant
        would be wrong on a measurable minority by 7%.
        """
        with h5py.File(path, "r") as handle:
            header = bytes(handle[HEADER][()]).decode()
        found = FOV_PATTERN.search(header)
        if found is None:
            raise ValueError(f"{path.name} carries no reconSpace field of view")
        return int(found.group(1)), float(found.group(2))

    def build(self) -> None:
        """Decode every KNEE through `ForeignBuild`, each exam into the slot its acquisition routed to.

        UNLIKE RIJEKA'S, THIS MASK IS NOT ALL ONES, AND UNLIKE THIS BUILD'S OWN FIRST VERSION IT IS NOT
        ONE SLOT EITHER. An EXAM is acquired with fat suppression or without, never both — but a KNEE is
        usually scanned both ways in the same sitting, and 547 of the 556 patients here carry both
        acquisitions. A row per file left every mask one-hot and the corpus reading as 1,172 half-studies;
        a row per knee fills both slots on 547 of them, which is the same shape our own corpus has. The
        nine that stay one-hot are the ones genuinely acquired one way, which is what the mask is for: a
        Rijeka exam fills one of six slots and leaves five zero, so "a foreign corpus reads flat" and "an
        empty input reads flat" are the same measurement there and are two different ones here.
        """
        studies = self.studies()
        ForeignBuild(self.cache).write(
            [study.identifier for study in studies],
            (
                [SlotVolume(volume.slot, volume.pixels(), volume.spacing) for volume in study.volumes]
                for study in studies
            ),
        )

    def metadata(self) -> list[StudyMetadata]:
        """What each exam was acquired with — the corpus whose geometry is STATED rather than derived.

        Fat suppression is read back off the routed slot, which is read off the acquisition string, which
        is matched by equality. So a filled `fat_suppressed` cell here traces to a word the corpus wrote
        about itself, and an exam on an unrecognised acquisition has no row at all rather than a guessed
        one — `volumes` has already dropped it. A knee scanned BOTH ways leaves the cell absent rather
        than picking one of its two true answers; `FastmriStudy.fat_suppressed` argues that case.

        Both acquisitions are `fluid_sensitive`: coronal PD without fat suppression is a standard meniscus
        sequence, which is the same reading `PLAIN_ACQUISITION` records and the reason it does not land in
        a T1 slot. The patient axes stay absent — the corpus is de-identified and publishes no age, sex or
        scanner — and so does `field_strength_t`, which the ISMRMRD header does carry and this reader does
        not parse. That last one is an omission with a known fix, not an unanswerable axis.
        """
        return [
            StudyMetadata(
                identifier=study.identifier,
                corpus=FASTMRI_CORPUS,
                plane=FASTMRI_PLANE,
                sequence=FASTMRI_SEQUENCE,
                fat_suppressed=study.fat_suppressed(),
                fluid_sensitive=True,
                pixel_spacing_mm=study.geometry().spacing,
                matrix_px=study.geometry().matrix_px,
                field_of_view_mm=study.geometry().field_of_view_mm,
            )
            for study in self.studies()
        ]

    def labels(self, destination: Path) -> None:
        """The labels table: eleven of our twelve findings, most of them carrying absence.

        A KNEE on disk with no READ exam gets NO ROW. It was never read, so every column would be a
        guess, and `Targets.pair` already handles an absent study the way it handles an absent column.

        THE UNION OVER A KNEE'S EXAMS, AND IT IS A DECISION RATHER THAN AN ARITHMETIC. fastMRI+ readers
        read each FILE independently, so a knee's two exams are two reads of one joint — and they
        disagree: the count this method logs is how often one exam of a pair is boxed while the other is
        clean. A box is positive evidence a finding is there, and silence under a protocol that disclaims
        exhaustiveness is weak evidence it is not, so the union is the reading that treats each as what it
        is. The intersection would let the weaker read veto the stronger one; it is the other arm of this
        experiment and this method does not hide that it was not taken.
        """
        annotations = FastmriAnnotations.read(self.annotations)
        seen, straddling = annotations.terms(), annotations.straddling()
        knees = {study.identifier: [name for name in study.files() if name in seen] for study in self.studies()}
        read = {knee: files for knee, files in knees.items() if files}
        logger.info(f"{self._disagreeing(seen, read)} knees where one exam was boxed and another read clean")
        rows = [
            self._row(knee, self._union(seen, files), self._union(straddling, files)) for knee, files in read.items()
        ]
        ForeignBuild.write_table(pd.DataFrame(rows), destination, TARGETS)

    @staticmethod
    def _disagreeing(seen: dict[str, frozenset[str]], read: dict[str, list[str]]) -> int:
        return sum(1 for files in read.values() if len({bool(seen[name]) for name in files}) > 1)

    @staticmethod
    def _union(table: dict[str, frozenset[str]], files: list[str]) -> frozenset[str]:
        return frozenset().union(*(table.get(name, frozenset()) for name in files))

    def _row(self, knee: str, terms: frozenset[str], located: frozenset[str]) -> dict[str, str | float]:
        row: dict[str, str | float] = {STUDY: knee}
        for target, reading in ((target, READINGS.get(target)) for target in TARGETS):
            level = None if reading is None else reading.verdict(terms, located)
            if level is not None:
                row[target] = level
                row[f"{target}__conf"] = self.confidence
        return row
