"""The KMAR-50K motion-artifact corpus, read as PAIRS — and it is not a corpus of labels.

WHAT THIS IS FOR, stated first because the obvious use of it is closed. KMAR-50K ships the same knee
twice: once as acquired with patient motion and once as a motion-free reference, under identical
filenames in `ArtifactData_part*` and `GroundTruthData_part*`. Its metadata JSON is `[]` and neither
cohort table carries a finding, so there is NO pathology supervision here at all. Using its clean
volumes as unlabelled pretraining data is the SSL-as-init path, which this project has now watched
collapse five separate times, and nothing about this corpus makes that sixth attempt different.

The one honest use is the pairs themselves. A motion-artifact operator is a train-time augmentation, so
it enters through `Augmentation` rather than through an encoder initialisation — the one door that is
still open. And an operator built this way is not a guessed transform: the pairs let its severity be
CALIBRATED against measured artifact rather than dialled to taste. `kmar_stats` is that measurement and
it comes first; the operator is not written until the number it must reproduce exists.

WHAT IS ACTUALLY ON DISK, since the name says 50K and 50K is not here. 1,339 artifact volumes and 1,342
ground-truth volumes across the two parts, plus two `Testing*` directories holding one file each — an
interrupted fetch, not the published corpus. Every rate this module reports is over the pairs it built,
and `pairs()` reports both counts so the denominator is never the published one by accident.

THE FILENAME IS THE JOIN AND IT IS NOT EXACT. Disk carries `2020_1_transection_0.0.nii`; the cohort
table carries `2020_1_transection_0.nii.gz`. The trailing index is the same integer written two ways, so
a naive stem join silently matches nothing and reports a corpus with no metadata. `_key` normalises it.

WHY THE SPACING MATTERS EVEN THOUGH NOTHING IS DECODED INTO A CACHE. `PixelSpacing` in the cohort tables
ranges from 0.2578 to 0.703 mm/px across these files — nearly a factor of three. Ghosting is periodic in
the phase-encode direction in PIXELS, so a residual statistic measured without rescaling to a common
mm/px would be measuring the acquisition matrix as much as the motion.
"""

import ast
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

VOLUME_SUFFIX = ".nii"
ARTIFACT_DIRECTORY = "ArtifactData"
CLEAN_DIRECTORY = "GroundTruthData"

COHORT_FILES = ("TrainingCohort.csv", "TestingCohort.csv")
"""Both cohort tables, which do NOT share a first column name — `NII_FileName` against `Patient`.

Read positionally for that reason. Naming one of them would drop the other's 103 rows, and the two
together are the only source of pixel spacing this corpus has.
"""

SPACING_COLUMN = "PixelSpacing"
DESCRIPTION_COLUMN = "SeriesDescription"
KEY_PATTERN = re.compile(r"^(?P<body>.+?_(?:sagittal|coronal|transection))_(?P<index>[\d.]+)$")
"""Subject, plane and repeat index out of a stem, with the index left as written.

The three plane words are the corpus's own and `transection` is its spelling of axial. Anchoring on them
rather than splitting on underscores is what keeps a subject id containing an underscore from shifting
every field by one.
"""

VOLUME_AXES = 3
"""A NIfTI holding a volume rather than a single plane. A 2D file is one slice and is read as one."""

BACKGROUND_PERCENTILE = 40.0
"""The intensity below which a pixel is treated as outside the knee, per volume.

Not a segmentation and not claimed to be one. It exists so the residual can be split into signal that
landed ON the anatomy and signal that landed OFF it, because that split is what separates motion
GHOSTING — which deposits replicas of the object into empty space along the phase-encode axis — from
blur, which does not leave the object at all. An operator that reproduces the total residual while
putting none of it in the background is not reproducing motion.
"""


@dataclass(frozen=True)
class KmarCohortRow:
    """What the cohort tables record about one acquisition, for the two columns anything here reads.

    A row rather than two parallel dicts because both come out of the same pass over the same tables,
    and a spacing present where a description is absent would be a state neither caller could represent.
    """

    spacing: float
    description: str


@dataclass(frozen=True)
class KmarPair:
    """One knee acquired twice: with motion and without, at a known in-plane spacing."""

    identifier: str
    artifact: Path
    clean: Path
    plane: str
    spacing: float | None
    description: str = ""
    """The cohort's `SeriesDescription`, lowercased, or empty where no cohort row lists this file.

    It is what says which SLOT this volume occupies — `pd`/`t2` against `t1` is the weighting axis and
    `_fs` is fat suppression — so it is a recorded fact per volume rather than a modal claim about the
    corpus. `kmar_studies` is what reads it; a pair carries it because the cohort join already happened
    there and a second pass over the tables would be the same read twice.
    """
    """None when neither cohort table lists this file — a real state, not an error.

    The two tables cover 1,444 rows and the fetch left 2,681 volumes on disk, so a pair with no metadata
    row is expected. It is carried rather than dropped so a caller can report how much of its measurement
    ran without a spacing instead of quietly measuring a smaller corpus.
    """


class KmarCorpus:
    """The artifact/clean pairs on disk, joined by filename, with spacing from the cohort tables."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def pairs(self) -> list[KmarPair]:
        """Every stem present in BOTH an artifact and a ground-truth directory, in filename order.

        The intersection is the corpus. The two sides differ by three files here, and a join that
        assumed them equal would pair a volume with a different knee — which no downstream statistic
        could detect, because both halves would still be knees.
        """
        artifacts = self._volumes(ARTIFACT_DIRECTORY)
        cleans = self._volumes(CLEAN_DIRECTORY)
        cohort = self.cohort()
        shared = sorted(set(artifacts) & set(cleans))
        pairs = [
            KmarPair(
                stem,
                artifacts[stem],
                cleans[stem],
                self._plane(stem),
                row.spacing if (row := cohort.get(self._key(stem))) else None,
                row.description if row else "",
            )
            for stem in shared
        ]
        located = sum(1 for pair in pairs if pair.spacing is not None)
        logger.info(
            f"{len(pairs)} pairs of {len(artifacts)} artifact and {len(cleans)} clean volumes, "
            f"{located} with a cohort spacing"
        )
        return pairs

    def _volumes(self, kind: str) -> dict[str, Path]:
        """Stem to path for one side of the corpus, across every part directory under the root."""
        found = {}
        for directory in sorted(self.root.rglob(f"{kind}*")):
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob(f"*{VOLUME_SUFFIX}")):
                found[path.stem] = path
        return found

    def _plane(self, stem: str) -> str:
        """The acquisition plane out of the filename, or an empty string if it is not one of the three."""
        found = KEY_PATTERN.match(stem)
        return "" if found is None else found.group("body").rsplit("_", 1)[-1]

    def _key(self, stem: str) -> str:
        """A stem reduced to the form both the disk and the cohort tables agree on.

        `2020_1_transection_0.0` and `2020_1_transection_0.nii.gz` are the same acquisition written two
        ways, and the difference is one float format. Comparing them raw is the join that matches nothing.
        """
        found = KEY_PATTERN.match(stem)
        if found is None:
            return stem
        return f"{found.group('body')}_{int(float(found.group('index')))}"

    def spacings(self) -> dict[str, float]:
        """Millimetres per pixel in plane, keyed the way `_key` keys a stem."""
        return {key: row.spacing for key, row in self.cohort().items()}

    def cohort(self) -> dict[str, KmarCohortRow]:
        """Both cohort tables as one mapping, keyed the way `_key` keys a stem.

        `PixelSpacing` arrives as the string ` [0.5, 0.5]` — a Python literal inside a quoted CSV field,
        so it is parsed as one rather than pulled apart with a regex. The two axes are equal on every row
        read here; the first is taken and a row whose axes disagree would be taking the wrong one, which
        is why the value is read rather than assumed square anywhere downstream.

        `SeriesDescription` arrives quoted and space-padded the same way and is lowercased once, here,
        because every consumer of it matches substrings (`_fs`, `t1`, `sag`) and a corpus that spells one
        acquisition two ways would route the same sequence into two different slots.
        """
        rows: dict[str, KmarCohortRow] = {}
        for name in COHORT_FILES:
            # rglob, because the tables ship one level down beside the part directories rather than at the
            # root the volumes are found from. Looking only at the root finds nothing and reports a corpus
            # with no spacing at all, which is a plausible-looking answer and a wrong one.
            for path in sorted(self.root.rglob(name)):
                table = pd.read_csv(path, skipinitialspace=True)
                for _, row in table.iterrows():
                    stem = str(row.iloc[0]).strip().strip("'").split(".nii")[0]
                    rows[self._key(stem)] = KmarCohortRow(
                        spacing=float(ast.literal_eval(str(row[SPACING_COLUMN]).strip())[0]),
                        description=str(row.get(DESCRIPTION_COLUMN, "")).strip().strip("'").lower(),
                    )
        logger.info(f"{len(rows)} cohort rows carrying a pixel spacing")
        return rows

    def volume(self, path: Path) -> NDArray[np.float32]:
        """One NIfTI as slices-first float32, which is the axis order every other corpus here uses.

        NIfTI is written x, y, slice; our caches and `SlotReader` are written slice, y, x. Transposing
        at the read is the only place that has to know, and getting it wrong would hand a statistic
        about the phase-encode direction a volume whose in-plane axes are the slice axis.
        """
        loaded = np.asarray(Nifti1Image.from_filename(path).get_fdata(), dtype=np.float32)
        return np.transpose(loaded, (2, 0, 1)) if loaded.ndim == VOLUME_AXES else loaded[np.newaxis]
