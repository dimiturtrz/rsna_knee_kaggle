"""KMAR-50K as STUDIES rather than as pairs — the clean volumes, grouped by subject and routed by slot.

`kmar_corpus` reads this corpus as artifact/clean PAIRS, which is what calibrating a motion operator
needs. This module reads the same files for the other use: a fourth foreign corpus decoded into the
arrays the trainer already loads, on the three mechanisms `rijeka_corpus` audits and this module does
not repeat — `ReportFolds.assign` folding an unknown identifier on itself, `Targets.pair` weighting an
absent column at 0.0, and `train.py` never reading the NaN rows it reindexes.

WHAT IT ADDS OVER THE OTHER THREE, which is the argument for paying for the decode. It is the only
corpus on this machine whose licence clears: CC BY 4.0, against Rijeka's CC BY-NC-ND, MRNet's
no-adaptation agreement, fastMRI's no-derived-files clause and SKM-TEA's non-commercial RUA. Every
`merged_*` number this project owns is therefore an OFFLINE result about what more data does; a member
fitted over this corpus is the first one that could also be SHIPPED.

THE CLEAN SIDE, NOT THE ARTIFACT SIDE. `GroundTruthData_*` is the motion-free reference acquisition —
ordinary diagnostic imaging, which is what a supervised arm wants. The artifact side is a paired
degradation and belongs to the operator-calibration use, not to this one.

IT CARRIES NO LABELS AND THIS MODULE WRITES NONE. The metadata JSON is `[]` and neither cohort table
records a finding, so a decode is all that happens here. What supervises an arm over these studies is a
separate question with a separate answer, and `kmar_corpus` states why the obvious one — clean volumes
as unlabelled pretraining — is the SSL-as-init path this project has watched collapse five times.

THE SUBJECT IS THE STUDY AND THE FILENAME IS THE ONLY THING THAT SAYS SO. `2020_104_coronal_0.0.nii`
and `2020_104_transection_0.0.nii` are one knee read twice; nothing in either cohort table joins them.
So the subject is the stem with its plane and repeat index removed, which is what `SUBJECT_PATTERN`
takes — anchored on the three plane words rather than split on underscores, because a subject id
containing one (`2021_80-XXX_MR1202103160326`) would otherwise shift every field by one.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import UInt8
from numpy.typing import NDArray
from torch import Tensor

from core.data.corpus.kmar_corpus import KmarCorpus, KmarPair
from core.data.corpus.slots import KMAR_SLOTS, Plane, Slot
from core.data.corpus.study_cache import StudyCache
from core.data.corpus.study_metadata import StudyMetadata

logger = logging.getLogger(__name__)

KMAR_CORPUS = "kmar"
"""The origin name, spelled once — a second spelling splits one corpus into two in the merged tables."""

SUBJECT_PATTERN = re.compile(r"^(?P<subject>.+?)_(?:sagittal|coronal|transection)_[\d.]+$")
"""The subject out of a stem, anchored on the plane word the corpus always writes."""

PLANE_WORDS: dict[str, str] = {"sag": Plane.SAGITTAL, "cor": Plane.CORONAL, "tra": Plane.AXIAL}
"""The corpus's `SeriesDescription` plane abbreviations, in our vocabulary.

`tra` is its spelling of transverse and the filenames spell the same acquisition `transection`; both
mean axial. The description is what is matched here, because it is the field the weighting axis is also
read from and using two sources for one routing decision is how they come to disagree.
"""

FLUID_WEIGHTINGS = ("pd", "t2")
"""Proton density and T2 — the descriptions that are fluid-sensitive. T1 is the complement, and a
description naming neither is not routed at all rather than defaulted into one."""

FAT_SUPPRESSED = "_fs"
"""Fat suppression as this corpus spells it, in every variant (`pd_tse_fs_cor`, `qpd_tse_fs_sag`)."""

PROGRESS_EVERY = 50
"""Decoded studies between progress lines."""


@dataclass(frozen=True)
class KmarStudy:
    """One subject's clean acquisitions, each already placed in the slot its description names."""

    identifier: str
    volumes: dict[int, Path]
    """Slot position in `KMAR_SLOTS` to the clean volume that fills it.

    A dict rather than a list because most subjects fill ONE of the five — 918 of 1,101, measured
    2026-09-12 — so the absent positions are the normal case and have to be representable as absent
    rather than as a path that happens to be missing.
    """
    spacing: float

    def present(self, slots: int) -> list[float]:
        """The presence flag per slot: 1.0 where this subject has an acquisition, 0.0 where it does not.

        This is the mask the head's softmax renormalises over, and it is the one thing that separates
        this corpus from Rijeka's. Rijeka fills `SLOTS[0]` for every exam, so its mask is a constant and
        carries no information; here the filled position varies across the corpus, so a member reading
        these studies meets a mask that actually says something.
        """
        return [1.0 if position in self.volumes else 0.0 for position in range(slots)]


class KmarStudies:
    """The clean side of KMAR-50K grouped into studies, ready to decode into a `StudyCache`."""

    def __init__(self, root: Path, cache: StudyCache) -> None:
        self.corpus = KmarCorpus(root)
        self.cache = cache

    def studies(self) -> list[KmarStudy]:
        """Every subject with at least one routable acquisition, in subject order.

        THE DENOMINATOR IS REPORTED, because this corpus's published name says 50K and 1,339 paired
        volumes are what the fetch left on disk. Volumes whose description routes nowhere are counted
        out loud rather than dropped silently: measured 2026-09-12 that is 19 of 1,339 (1.4%) — thirteen
        coronal and two sagittal fluid-weighted WITHOUT fat suppression, four free-text scanner strings
        and two reading `pd_tse_fs_con`. The two `COR_FLUID_NOFS`-shaped ones would need
        `EXTENDED_SLOTS`, which no canonical merge could then copy.
        """
        grouped: dict[str, dict[int, Path]] = {}
        spacings: dict[str, float] = {}
        skipped: Counter[str] = Counter()
        for pair in self.corpus.pairs():
            position = self.position(pair.description)
            if position is None:
                skipped[pair.description or "no cohort row"] += 1
                continue
            subject = self.subject(pair.identifier)
            grouped.setdefault(subject, {})[position] = pair.clean
            spacings.setdefault(subject, pair.spacing or 0.0)
        studies = [KmarStudy(subject, volumes, spacings[subject]) for subject, volumes in sorted(grouped.items())]
        filled = Counter(len(study.volumes) for study in studies)
        logger.info(f"{len(studies)} subjects, slots filled per subject {dict(sorted(filled.items()))}")
        logger.info(f"{sum(skipped.values())} volumes routed nowhere: {skipped.most_common(5)}")
        return studies

    @staticmethod
    def subject(identifier: str) -> str:
        """The stem with its plane and repeat index removed, or the stem itself if it carries neither."""
        found = SUBJECT_PATTERN.match(identifier)
        return identifier if found is None else found.group("subject")

    @staticmethod
    def position(description: str) -> int | None:
        """Which of `KMAR_SLOTS` this description names, or None when it names none of them.

        NONE IS A REAL ANSWER AND NOT A FAILURE. A coronal proton-density acquisition without fat
        suppression is a perfectly ordinary meniscus sequence that these five slots cannot express, and
        routing it into the nearest one would put a non-suppressed series in a fat-suppressed channel —
        the error `slots` measures at a fifth of our own corpus when the shipped flags are trusted.
        """
        plane = next((value for word, value in PLANE_WORDS.items() if word in description), None)
        if plane is None:
            return None
        candidate = Slot(
            name="",
            plane=plane,
            fluid_sensitive=any(word in description for word in FLUID_WEIGHTINGS),
            fat_suppressed=FAT_SUPPRESSED in description,
        )
        for position, slot in enumerate(KMAR_SLOTS):
            if (slot.plane, slot.fluid_sensitive, slot.fat_suppressed) == (
                candidate.plane,
                candidate.fluid_sensitive,
                candidate.fat_suppressed,
            ):
                return position
        return None

    def metadata(self) -> list[StudyMetadata]:
        """What each subject was acquired with, in the shape every corpus here answers in.

        THE PLANE AND SEQUENCE ARE THE FIRST FILLED SLOT'S, which is exact for the 83% of subjects that
        fill one and a genuine narrowing for the rest. `StudyMetadata` carries one acquisition per study
        and this corpus can carry up to four, so the alternative is a field that lies about which; the
        cache's own mask is where the full answer lives.

        THE SPACING IS RECORDED, NOT DERIVED — the one foreign corpus here where that is true. Rijeka's
        and MRNet's are reconstructed from a matrix and an assumed field of view; this one publishes
        `PixelSpacing` per acquisition, 0.2578 to 0.703 mm/px, and `SlotReader.window` crops in
        millimetres, so the physical extent is right per study rather than on average.
        """
        return [
            StudyMetadata(
                identifier=study.identifier,
                corpus=KMAR_CORPUS,
                plane=KMAR_SLOTS[min(study.volumes)].plane,
                sequence="PD" if KMAR_SLOTS[min(study.volumes)].fluid_sensitive else "T1",
                fat_suppressed=KMAR_SLOTS[min(study.volumes)].fat_suppressed,
                fluid_sensitive=KMAR_SLOTS[min(study.volumes)].fluid_sensitive,
                pixel_spacing_mm=study.spacing,
            )
            for study in self.studies()
        ]

    def stack(self, volume: NDArray[np.float32], spacing: float) -> UInt8[Tensor, "s h w"]:
        """One volume as the cache holds it: band-sampled, physically cropped, percentile-scaled.

        Every step is `SlotReader`'s, called on an array instead of on a directory of DICOMs — the same
        argument `rijeka_corpus.stack` makes. A corpus normalised its own way would arrive at the encoder
        in a different intensity space, and a transfer measured over it would be measuring the difference
        between two normalisations rather than between two corpora.
        """
        reader = self.cache.reader()
        sampled = volume[reader.band_indices(len(volume), self.cache.slices)]
        taken = reader.window(sampled, spacing)
        cropped = sampled if taken is None else sampled[:, taken[0], taken[1]]
        return reader.sized(cropped, self.cache.image_size)

    def build(self) -> None:
        """Write `images_<key>.npy`, `index_<key>.json` and — last — `mask_<key>.npy`.

        The mask goes last for the reason `StudyCache.build` writes it last: its presence is what says
        the arrays beside it are complete, so a build that dies half way leaves a cache that reads as
        unbuilt rather than as a corpus with black studies in it.

        THE MASK IS NOT ALL ONES HERE, which is where this differs from every other foreign builder. Most
        subjects fill one of the five slots, so the zeros are the majority of the array and they are the
        honest statement of what was acquired.
        """
        studies = self.studies()
        self.cache.prepare()
        images = np.lib.format.open_memmap(
            self.cache.images_path, mode="w+", dtype=np.uint8, shape=self.cache.shape(len(studies))
        )
        mask = np.zeros((len(studies), self.cache.slots), dtype=np.float32)
        for index, study in enumerate(studies):
            for position, path in study.volumes.items():
                if position < self.cache.slots:
                    images[index, position] = self.stack(self.corpus.volume(path), study.spacing).numpy()
            mask[index] = study.present(self.cache.slots)
            if index % PROGRESS_EVERY == 0:
                logger.info(f"{index}/{len(studies)} decoded")
        images.flush()
        self.cache.record([study.identifier for study in studies])
        self.cache.report(mask)
        np.save(self.cache.mask_path, mask)
        logger.info(f"wrote {self.cache.shape(len(studies))} uint8 -> {self.cache.images_path}")

    @staticmethod
    def paired(pairs: list[KmarPair]) -> int:
        """How many of these pairs carry a cohort description at all — the join's own denominator.

        Reported rather than assumed because `pairs()` keeps a volume whose cohort row is absent, and a
        routing pass over such a pair would silently count it as unroutable rather than as unlisted.
        """
        return sum(1 for pair in pairs if pair.description)
