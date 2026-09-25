"""The MRNet (Stanford) corpus, read into the arrays this project's trainer already loads.

A THIRD CORPUS on the same three mechanisms that already admit a foreign one — `ReportFolds.assign`
folding an unknown identifier on itself, `Targets.pair` weighting an absent column at 0.0, and
`train.py` never reading the NaN rows it reindexes. `rijeka_corpus` carries that audit; this module
does not repeat it, and is small for the same reason.

WHAT IT ADDS OVER RIJEKA, which is the whole argument for paying for it. A Rijeka exam is one sequence
and fills ONE slot of six, so a flat reading there has two explanations that cannot be told apart:
foreign data does not transfer, or a knee with five zero slots is a degenerate input. MRNet ships
sagittal, coronal and axial per exam and fills THREE, so it separates them. See `slots.MRNET_SLOTS`
for which slot each plane lands in and on whose authority.

IT SUPERVISES ACL AND NOTHING ELSE, and the omissions are deliberate:

  * `meniscus` does NOT distinguish medial from lateral, while our table carries `Medial Meniscus` and
    `Lateral Meniscus` as separate findings. Writing one label into both would teach a medial tear as
    lateral-positive — the same class of error `rijeka_corpus.labels` refuses when it declines to write
    0.5 into eleven columns nobody read.
  * `abnormal` has no column of ours to land in. It is a study-level flag, not one of the twelve.

1,370 exams were published; 1,250 are downloadable (1,130 train + 120 validation) and the 120-exam
test split is withheld. The join below is over what the archive actually holds, never over the paper's
figure — the denominator is the thing this project has got wrong most often.
"""

import io
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from core.data.corpus.foreign_build import ForeignBuild, SlotVolume
from core.data.corpus.mrnet_labels import SPLITS, ArchiveLabels, MRNetLabels
from core.data.corpus.mrnet_source import VOLUME_SUFFIX, MRNetSource
from core.data.corpus.study_cache import StudyCache
from core.data.corpus.study_metadata import SET_JOIN, StudyMetadata
from core.findings import STUDY

logger = logging.getLogger(__name__)

PLANES: tuple[str, ...] = ("sagittal", "axial", "coronal")
"""The archive's directory names, in the order of `slots.MRNET_SLOTS`. The two are one fact and must
stay aligned: position i of this tuple is written into slot i of the cache."""

PREFIX = "mrnet-"
"""Prefixed onto the archive's case number, and not decoration.

The cases are zero-padded numerals, so a merged labels table round-tripped through pandas reads `0000`
back as the integer 0 and stops joining to the index. The prefix also makes `MergedCache._disjoint`'s
guarantee hold by construction rather than by luck.
"""

ACL = "ACL"
CONFIDENCE = f"{ACL}__conf"

VOLUME_DEPTH = 2
"""How many path components a volume is identified by AT LEAST: `<plane>/<case>.npy`. Anything shallower
is a label table or an archive stub, never pixels.

It was 3, for the `<split>/<plane>/<case>.npy` the Stanford download ships. The Redivis export of the
same corpus FLATTENS the split away — see `MRNetLabels.split_of` for how it is recovered — so the split component
is now optional and the plane is read from the tail either way.
"""

NESTED_DEPTH = 3
"""The depth at which `parts[-3]` could be a split rather than a wrapper directory. Both layouts put a
directory there; only one of them puts a SPLIT there, which is why the name is checked and not the depth
alone."""

MRNET_SPACING = 0.625
"""Millimetres per pixel in plane — DERIVED, because the corpus records no spacing.

Bien et al. resampled every exam to a 256 px grid, and a knee is acquired at a median 160 mm field of
view (`slots.DETAIL_MM` carries our own measurement of that). 160/256 gives 0.625.

IT IS WEAKER EVIDENCE THAN `RIJEKA_SPACING`, and saying so is the point: Rijeka ships ACL bounding
boxes whose median size checks the derivation against a physical length, and MRNet ships no such
quantity at all. What can be said is that the value is load-bearing rather than cosmetic — the
canonical read crops a constant 130 mm, which is 208 px of 256 here, while 0.5 mm/px would want 260 and
`SlotReader.window` would decline to crop at all, handing the encoder a different mm/pixel than every
member was fitted at. The two branches are far apart, so this is a value to re-examine if the corpus
ever reads worse than a random-crop control rather than one to accept quietly.
"""

MRNET_CORPUS = "mrnet"
"""What this corpus calls itself in the `corpus` column, matching the name `MergedCache.labels` is given
for it. The prefix on the identifiers is `mrnet-`, and the two are deliberately the same word."""

MRNET_MATRIX = 256
"""The in-plane grid every exam was resampled onto before release — stated by Bien et al., not derived,
and the denominator `MRNET_SPACING` divides the field of view by."""


MRNET_LEVEL: tuple[float, ...] = (0.08, 0.82)
"""What a negative and a positive ACL row are worth on our level scale.

The labels were extracted by hand from the clinical reports, which is our OWN corpus's provenance
rather than Rijeka's double-blind image read — so they belong on the same scale our reader emits onto,
read through `Supervision.MEASURED_AGREEMENT`:

  negative -> 0.08.  The report was read for the ACL and states no tear. That is the confident-absence
                     level, agreed with 95.9% of the time when read as an absence claim.
  positive -> 0.82.  A stated tear, agreed with 68.8% of the time. NOT 0.94: this label is binary and
                     lumps partial with complete, and 0.94 is what our reader reserves for an unhedged
                     severe finding. Taking the middle confident-yes level is what a label carrying no
                     severity can honestly claim.
"""

REPORT_CONFIDENCE = 0.95
"""The weight a populated cell pulls with — our own report-derived cells carry exactly this, and these
labels come from the same kind of text by the same kind of extraction. Rijeka's 1.0 is not transferable:
that corpus is an expert reading the pixels."""


@dataclass(frozen=True)
class MRNetExam:
    """One exam: its identifier, where each plane's pixels are, and what the report said about the ACL."""

    identifier: str
    members: dict[str, str]
    acl: int

    def level(self) -> float:
        """This verdict on our level scale — see `MRNET_LEVEL` for why each value is what it is."""
        return MRNET_LEVEL[self.acl]

    def present(self, slots: int) -> list[float]:
        """The presence flag per slot: 1.0 where this exam's plane ships, 0.0 where it does not.

        A complete download fills all three. A truncated one does not, and the mask is what lets the
        head's softmax renormalise over what is actually there instead of attending to black pixels the
        mask calls acquired.
        """
        return [1.0 if PLANES[position] in self.members else 0.0 for position in range(slots)]


class MRNetCorpus:
    """Reads the MRNet archive into a `StudyCache`-shaped cache and a labels table beside it."""

    def __init__(
        self,
        archive: Path,
        cache: StudyCache,
        spacing: float = MRNET_SPACING,
        tables: MRNetLabels | None = None,
    ) -> None:
        self.archive = archive
        self.source = MRNetSource.at(archive)
        self.cache = cache
        self.spacing = spacing
        self.tables = tables or ArchiveLabels(archive)

    def volumes(self) -> list[MRNetExam]:
        """The exams whose PIXELS ship, in split then case order, joined against the archive's members.

        THE JOIN IS THE DENOMINATOR. The label tables name every case in their split; the archive is
        what holds the arrays. Trusting the table's length would report rates over exams that are not
        here — the paper's 1,370 against the 1,250 that download.
        """
        names = self.source.names()
        labels = {split: self.tables.table(split) for split in SPLITS}
        members = self._members(names)
        exams = [
            MRNetExam(identifier=f"{PREFIX}{case}", members=members[(split, case)], acl=int(verdict))
            for split in SPLITS
            for case, verdict in labels[split].items()
            if (split, case) in members
        ]
        published = sum(len(table) for table in labels.values())
        logger.info(f"{len(exams)} of {published} labelled exams ship here: {self._share(published, len(exams))}")
        # The truncation is the CACHE's rule and not this module's: `first` is in the cache's name, so a
        # caller that sliced its own list would be free to write 60 exams under the name that promises
        # 1,250. Asking `enrolled` is what makes the file say what it holds.
        kept = set(self.cache.enrolled([exam.identifier for exam in exams]))
        return [exam for exam in exams if exam.identifier in kept]

    @staticmethod
    def _share(published: int, present: int) -> str:
        """The join as a percentage of what the label tables named, so no caller computes it to be honest."""
        return f"{100.0 * present / max(published, 1):.1f}% of the labelled set"

    @staticmethod
    def _members(names: list[str]) -> dict[tuple[str, str], dict[str, str]]:
        """Every volume in the archive, keyed by (split, case) and then by plane.

        Matched on the tail of the path rather than on a fixed root, because the top-level directory
        name differs between the Stanford download and every mirror of it — and because one export has
        no split directory at all, which is the case `_split_at` covers.
        """
        found: dict[tuple[str, str], dict[str, str]] = {}
        for name in names:
            path = Path(name)
            case = MRNetLabels.case_of(path.stem)
            if path.suffix != VOLUME_SUFFIX or len(path.parts) < VOLUME_DEPTH or case is None:
                continue
            plane = path.parts[-2]
            if plane in PLANES:
                found.setdefault((MRNetCorpus._split_at(path, case), case), {})[plane] = name
        return found

    @staticmethod
    def _split_at(path: Path, case: str) -> str:
        """The split a volume belongs to: the directory that says so, or the case number that implies it.

        The stated one WINS where it exists. The two agree on every case MRNet publishes, so this only
        ever matters if a mirror re-splits the corpus — and then the mirror's own word is the better
        evidence than an arithmetic rule fitted to the original.
        """
        stated = path.parts[-3] if len(path.parts) >= NESTED_DEPTH else ""
        return stated if stated in SPLITS else MRNetLabels.split_of(case)

    def build(self) -> None:
        """Decode every exam through `ForeignBuild`, each plane into the slot `PLANES` puts it in.

        A plane the download is missing writes nothing, so its slot stays black AND unmarked — the mask is
        what lets the head's softmax renormalise over what actually shipped.
        """
        exams = self.volumes()
        ForeignBuild(self.cache).write([exam.identifier for exam in exams], (self._planes(exam) for exam in exams))

    def _planes(self, exam: MRNetExam) -> Iterator[SlotVolume]:
        for position, plane in enumerate(PLANES[: self.cache.slots]):
            if plane in exam.members:
                yield SlotVolume(position, self._pixels(self.source, exam.members[plane]), self.spacing)

    @staticmethod
    def _pixels(source: MRNetSource, member: str) -> NDArray[np.float32]:
        """One plane's stack as float32. Loaded from bytes, so no pickle is ever unpacked."""
        return np.asarray(np.load(io.BytesIO(source.read(member)), allow_pickle=False), dtype=np.float32)

    def metadata(self) -> list[StudyMetadata]:
        """What each exam was acquired with, as far as MRNet says — which is the geometry and the planes.

        THREE AXES ARE LEFT ABSENT ON PURPOSE AND THE OMISSIONS CARRY THE SAME WEIGHT AS THE FILLED CELLS.
        MRNet publishes no sequence per plane — the release is stated to mix T1, T2 and PD across the
        corpus and the per-exam mapping is not distributed — so `sequence`, `fat_suppressed` and
        `fluid_sensitive` are unanswerable here. Filling them from the plane would be the failure
        `study_metadata` names: a guessed cell reads exactly like a header read, and Rijeka's cells beside
        them ARE header-grade.

        The spacing is `MRNET_SPACING`, which that constant's own note calls weaker evidence than Rijeka's
        — derived from a stated 256 px grid and a field-of-view prior with nothing physical to check it
        against. It is recorded because it is the number this corpus's whole geometry already runs on.
        """
        return [
            StudyMetadata(
                identifier=exam.identifier,
                corpus=MRNET_CORPUS,
                plane=SET_JOIN.join(plane.capitalize() for plane in PLANES if plane in exam.members),
                pixel_spacing_mm=self.spacing,
                matrix_px=MRNET_MATRIX,
            )
            for exam in self.volumes()
        ]

    def labels(self, destination: Path) -> None:
        """The labels table, carrying the ONE finding this corpus can honestly supervise.

        Absence is the mechanism, not an omission — see this module's own docstring for why `meniscus`
        and `abnormal` are left out rather than mapped onto columns they do not mean.
        """
        self.write_labels([(exam.identifier, exam.level()) for exam in self.volumes()], destination)

    @staticmethod
    def write_labels(rows: list[tuple[str, float]], destination: Path) -> None:
        """Write the ACL table, one row per exam: one supervised column and one weight.

        Shared with `mrnet_cached.CachedMRNet` rather than duplicated there, because the two producers
        differ ONLY in where they learn which exams are present — and a table whose columns drifted
        between them would join cleanly and mean something different, which is the failure this project
        has paid for most.
        """
        table = pd.DataFrame(
            {
                STUDY: [identifier for identifier, _ in rows],
                ACL: [level for _, level in rows],
                CONFIDENCE: REPORT_CONFIDENCE,
            }
        )
        ForeignBuild.write_table(table, destination, [ACL])
