"""The kneeMRI (Rijeka) corpus, read into the arrays this project's trainer already loads.

A SECOND CORPUS, not a second pipeline. Everything downstream of the cache is unchanged and none of it
was touched to make this work — three separate mechanisms already admit a foreign corpus, and the audit
that found them is the reason this module is small:

  * `ReportFolds.assign` hashes `reports.get(identifier, identifier)`, so a study absent from `train.csv`
    folds on its own identifier instead of failing.
  * `Targets.pair` gives a target column ABSENT from the labels table weight 0.0 and score 0.5, so a
    table carrying only `ACL` supervises ACL and contributes exactly nothing on the other eleven.
  * `train.py` reindexes `train.csv` onto our identifiers and never reads the resulting NaN rows unless
    annotations are switched on, which they are not by default.

So a pretraining arm needs a cache and a labels table, which is what this writes.

WHY IT DOES NOT GO THROUGH `StudyCache.build`. That path decodes DICOM directories — `decode_one` calls
`Study.from_directory` — and this corpus ships one pickled volume per exam. What it shares instead is
the part that matters for transfer: the same physical crop, the same band sampling and the same
percentile intensity normalisation, all of them `SlotReader`'s own, so the two corpora arrive at the
encoder in one geometry and one intensity space rather than merely in one array shape.

THE CORPUS IS 736 VOLUMES AND THE PUBLISHED FIGURE IS 917. The Kaggle mirror ships `vol01`..`vol08` and
omits `vol09`/`vol10`, so every rate taken from it has 736 underneath, never 917. The dropped 181 are
not label-biased — present 74.3/19.6/6.1% against the published 75.2/18.8/6.0% — which is what makes the
subset usable rather than merely available.
"""

import io
import logging
import pickle
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from core.data.corpus.foreign_build import ForeignBuild, SlotVolume
from core.data.corpus.study_cache import StudyCache
from core.data.corpus.study_metadata import StudyMetadata
from core.findings import STUDY

logger = logging.getLogger(__name__)

METADATA = "metadata.csv"
VOLUME_SUFFIX = ".pck"

RIJEKA_SPACING = 0.5
"""Millimetres per pixel in plane — DERIVED, because `metadata.csv` records no spacing.

The volumes are 320 px across and the corpus does not state its field of view. Ours does: a median
160 mm (see `slots.DETAIL_MM`), and a knee is a knee on a 1.5T scanner either side of the join. 160/320
gives 0.5.

IT IS CHECKED AGAINST THE ONE PHYSICAL LENGTH THE METADATA DOES CARRY, rather than left as an
assumption. The ACL bounding boxes median 90 x 91 px, which is 45 x 45 mm at this spacing, against an
annotation protocol that calls them boxes "enveloping the wider cruciate ligament area" around a
ligament 30-35 mm long. 0.4 mm/px would make those boxes 36 mm and 0.6 would make them 54 mm; 0.5 puts
the check in the middle of its plausible range, and no value that fits the boxes moves the crop enough
to matter.

It matters because the canonical read takes a constant 130 mm and resamples it to 336 px. A wrong
spacing lands this corpus at a different mm/pixel than the corpus it exists to pretrain for — the one
property `SlotReader`'s entire geometry is built to hold constant.
"""

DIAGNOSIS = "aclDiagnosis"
VOLUME_FILE = "volumeFilename"

ACL = "ACL"
CONFIDENCE = f"{ACL}__conf"

READER_LEVEL: tuple[float, ...] = (0.08, 0.82, 0.94)
"""What each of the corpus's three ligament verdicts is worth on OUR level scale — healthy, partial,
ruptured — and the mapping is read off measured agreement rather than off the levels' face value.

`Supervision.MEASURED_AGREEMENT` records what each level is actually worth: 0.08 -> the radiologist
calls the finding present 4.1% of the time, 0.28 -> 13.6%, 0.68 -> 37.8%, 0.82 -> 68.8%, 0.94 -> 95.8%.
Those rates are the meaning of a level, so the question to ask of a Rijeka verdict is not "how sure does
it sound" but "if an RSNA radiologist read this same knee, how often would they call the ACL abnormal".

  healthy   -> 0.08.  A double-blind expert read of an intact ligament from the images themselves. This
                      is the most confident ABSENCE in the label set and 4.1% is its home.
  partial   -> 0.82.  The ligament is abnormal and an expert confirmed it, but a partial tear is the
                      classic disagreement case between readers, which is exactly what 68.8% describes.
  ruptured  -> 0.94.  A complete rupture read directly off the pixels is the most certain call anywhere
                      in this vocabulary, and 95.8% is the level that says so.

These are EXPERT IMAGE READS, not findings mined out of report text, so they belong at the confident end
rather than the hedged one — the ordering is what carries the mechanism, and no level below 0.68 is
appropriate for a confirmed injury. The exact values matter little for encoder pretraining and matter
a great deal for honesty about what the labels claim.
"""

EXPERT_CONFIDENCE = 1.0
"""Every row is a double-blind read by a diagnostic radiologist, so nothing here is a hedged extraction."""

RIJEKA_CORPUS = "rijeka"
"""The origin name, spelled as the merged tables already spell it — `merged3_tagged_labels.csv` reads
canonical 4406, mrnet 1248, rijeka 736, and a second spelling here would split one corpus into two."""

RIJEKA_PLANE = "Sagittal"
RIJEKA_SEQUENCE = "PD"
"""The published protocol: sagittal proton-density fat-suppressed, one acquisition for the whole corpus.
This is the predicate `merged_cache` relies on to place every kneeMRI exam in `SLOTS[0]`."""

RIJEKA_MATRIX = 320
"""Pixels across, stated by the volumes themselves and the denominator `RIJEKA_SPACING` divides."""


@dataclass(frozen=True)
class RijekaVolume:
    """One exam: what it is called, where its pixels are, and what the reader said about the ligament."""

    identifier: str
    member: str
    diagnosis: int

    def level(self) -> float:
        """This verdict on our level scale — see `READER_LEVEL` for why each value is what it is."""
        return READER_LEVEL[self.diagnosis]


class RijekaCorpus:
    """Reads the kneeMRI archive into a `StudyCache`-shaped cache and a labels table beside it.

    The archive is read in place rather than extracted: it is 3.3 GB of pickles that are each used once,
    and a second copy on disk buys nothing.
    """

    def __init__(self, archive: Path, cache: StudyCache, spacing: float = RIJEKA_SPACING) -> None:
        self.archive = archive
        self.cache = cache
        self.spacing = spacing

    def volumes(self) -> list[RijekaVolume]:
        """The exams the archive actually holds, in metadata order, joined on the volume filename.

        THE JOIN IS THE DENOMINATOR AND IT IS NOT THE PUBLISHED ONE. `metadata.csv` carries all 917
        published rows while the mirror ships 736 volumes, so joining the other way — or trusting the
        table's length — would report rates over exams whose pixels are not here. Every row this returns
        has both halves.
        """
        with zipfile.ZipFile(self.archive) as archive:
            members = {Path(name).name: name for name in archive.namelist() if name.endswith(VOLUME_SUFFIX)}
            table = pd.read_csv(io.BytesIO(archive.read(self._metadata(archive))))
        present = table[table[VOLUME_FILE].isin(members)]
        logger.info(f"{len(present)} of {len(table)} published exams ship here: {self._share(table, present)}")
        return [
            RijekaVolume(
                identifier=str(Path(str(row[VOLUME_FILE])).stem),
                member=members[str(row[VOLUME_FILE])],
                diagnosis=int(row[DIAGNOSIS]),
            )
            for _, row in present.iterrows()
        ]

    def metadata(self) -> list[StudyMetadata]:
        """What this corpus was acquired with, per exam, in the shape every corpus answers in.

        THE ACQUISITION IS A CONSTANT HERE and that is a property of the corpus rather than a shortcut:
        kneeMRI publishes one protocol, sagittal proton-density fat-suppressed on a 320 px matrix, which
        is why `merged_cache` can place every one of these exams in `SLOTS[0]` without asking. So the
        frame is 736 identical rows, and identical is the honest answer.

        THE SPACING IS DERIVED AND SAYS SO. `RIJEKA_SPACING` reconstructs 0.5 mm/px from the matrix and a
        field of view the corpus never publishes, checked against the ACL box sizes. It is recorded
        because it is the number this corpus's entire geometry already runs on, and it is recorded under
        `pixel_spacing_mm` rather than `slice_spacing_mm` because it is an in-plane measure — the
        through-plane spacing is not known here, and filling it with this value would be a wrong field
        rather than an absent one.

        EVERYTHING ABOUT THE PATIENT IS ABSENT. `metadata.csv` carries the ligament verdict and the
        volume filename; no age, no sex, no scanner, no field strength. Those cells stay None.
        """
        return [
            StudyMetadata(
                identifier=volume.identifier,
                corpus=RIJEKA_CORPUS,
                plane=RIJEKA_PLANE,
                sequence=RIJEKA_SEQUENCE,
                fat_suppressed=True,
                fluid_sensitive=True,
                pixel_spacing_mm=self.spacing,
                matrix_px=RIJEKA_MATRIX,
            )
            for volume in self.volumes()
        ]

    @staticmethod
    def _share(table: pd.DataFrame, present: pd.DataFrame) -> str:
        """The join as a percentage of what was published, so no caller has to compute it to be honest."""
        return f"{100.0 * len(present) / max(len(table), 1):.1f}% of the published set"

    @staticmethod
    def _metadata(archive: zipfile.ZipFile) -> str:
        """Where the table sits in the archive, which is not fixed across mirrors of this corpus."""
        for name in archive.namelist():
            if Path(name).name == METADATA:
                return name
        raise FileNotFoundError(f"{METADATA} is not in this archive")

    def build(self) -> None:
        """Decode every exam through `ForeignBuild`, each into `SLOTS[0]`.

        The mask comes out all ones. Every exam in this corpus acquired the one sequence it has, so there
        is no slot for the presence mask to deny — and saying so explicitly is what lets the head's
        softmax renormalise over this corpus the same way it does over ours.
        """
        volumes = self.volumes()
        ForeignBuild(self.cache).write([volume.identifier for volume in volumes], self._studies(volumes))

    def _studies(self, volumes: list[RijekaVolume]) -> Iterator[list[SlotVolume]]:
        with zipfile.ZipFile(self.archive) as archive:
            for volume in volumes:
                yield [SlotVolume(0, self._pixels(archive, volume), self.spacing)]

    @staticmethod
    def _pixels(archive: zipfile.ZipFile, volume: RijekaVolume) -> NDArray[np.float32]:
        """One exam's 12-bit volume as float32.

        `pickle` because that is the format the corpus is published in, and there is no other reader for
        it. The archive is a fixed download whose contents were listed and characterised before this ran
        — 736 arrays, every one an `ndarray` — but the risk is real rather than nil, so it is named here
        instead of being waved through: DO NOT point this module at an untrusted archive.
        """
        return np.asarray(pickle.loads(archive.read(volume.member)), dtype=np.float32)

    def labels(self, destination: Path) -> None:
        """The labels table, carrying the ONE finding this corpus reads and no column for the other eleven.

        Absence is the mechanism, not an omission. `Targets.pair` gives a missing target column weight
        0.0 and score 0.5, so eleven of the twelve heads receive no gradient from this corpus at all,
        while ACL receives a fully-weighted expert read. Writing 0.5 into those columns instead would
        teach the model that every Rijeka knee is exactly average on eleven findings nobody looked at.
        """
        volumes = self.volumes()
        table = pd.DataFrame(
            {
                STUDY: [volume.identifier for volume in volumes],
                ACL: [volume.level() for volume in volumes],
                CONFIDENCE: EXPERT_CONFIDENCE,
            }
        )
        ForeignBuild.write_table(table, destination, [ACL])
