"""The fastMRI+ annotation pair — `knee.csv` and `knee_file_list.csv` — read once, one way, for every reader.

Three readers used to open these two files each, and they disagreed about the file list's last line:
`Knee data` is a footer, not a study, and only one of the three stripped it. It never reached a label row
(no volume is called `Knee data`), but it did reach every "studies read" count, one too high.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BOXES = "knee.csv"
FILE_LIST = "knee_file_list.csv"
FILE_LIST_FOOTER = "Knee data"

FILE_COLUMN = "file"
SLICE_COLUMN = "slice"
LABEL_COLUMN = "label"
X_COLUMN = "x"
Y_COLUMN = "y"
WIDTH_COLUMN = "width"
HEIGHT_COLUMN = "height"
GEOMETRY = (X_COLUMN, Y_COLUMN, WIDTH_COLUMN, HEIGHT_COLUMN)

COMPARTMENT_TROUGH = 165.0
"""Where the medial and lateral compartments part, in reconstruction pixels of the 320-wide matrix.

MEASURED, NOT CHOSEN. Every one of the 973 train exams is `CORPD_FBK` or `CORPDFS_FBK` — CORONAL, read
off the `acquisition` attribute of all 973 files and confirmed by `protocolName` COR in the ISMRMRD
header. So image x IS the medial-lateral axis, and the 5,658 meniscus box centres (`x + width/2`) fall in
two clusters with a deep trough between them: over a 32-bin histogram the interior minimum sits at 165.0,
against peaks near 90 and 235 carrying more than a thousand boxes each. The value is the argmin of that
histogram between 100 and 220, not a midpoint and not a median — the median, 196.5, lies INSIDE the upper
cluster and would cut it in half.

This matters because the plane is what makes the axis mean anything. On a SAGITTAL knee the same centre-x
would be the anterior-posterior axis and the two clusters would be the anterior and posterior horns of one
meniscus, which is a different finding entirely. Check the acquisition before reusing this constant on
another corpus; `Annotations/brain.csv` obviously has no compartments at all.
"""


@dataclass(frozen=True)
class FastmriAnnotations:
    """The box table with its column names stripped, and the studies a reader opened, footer excluded."""

    boxes: pd.DataFrame
    listed: tuple[str, ...]

    @classmethod
    def read(cls, directory: Path) -> "FastmriAnnotations":
        """Both files, once.

        `header=None` is load-bearing. The file list has no header row, so a default read promotes the
        first study into the column name and loses it — the one-row error that made every rate in
        `fastmri_corpus` wrong on its first draft.
        """
        boxes = pd.read_csv(directory / BOXES)
        boxes.columns = [str(column).strip() for column in boxes.columns]
        names = pd.read_csv(directory / FILE_LIST, header=None).iloc[:, 0].astype(str)
        return cls(boxes, tuple(name for name in names if name != FILE_LIST_FOOTER))

    def studies(self) -> frozenset[str]:
        """The studies a reader opened."""
        return frozenset(self.listed)

    def terms(self) -> dict[str, frozenset[str]]:
        """Per study that was READ, the set of vocabulary terms boxed anywhere in it.

        Keyed by every listed study, so a study read and found clean appears with an empty set and earns
        negatives — the 199 studies that separate this corpus's read exams from the 974 that carry a box.
        Dropping them would raise every reported rate by a fifth. A boxed study missing from the list is
        admitted too: a box is itself a read.
        """
        seen: dict[str, frozenset[str]] = dict.fromkeys(self.listed, frozenset())
        for identifier, group in self.boxes.groupby(FILE_COLUMN):
            seen[str(identifier)] = frozenset(str(label) for label in group[LABEL_COLUMN])
        logger.info(f"{len(seen)} studies read, {sum(1 for terms in seen.values() if terms)} carrying a box")
        return seen

    def straddling(self) -> dict[str, frozenset[str]]:
        """Per study, the terms boxed on BOTH sides of `COMPARTMENT_TROUGH` — located without a laterality.

        Keyed like `terms`, so a study with no box appears with an empty set rather than absent, and a term
        boxed only once cannot straddle by arithmetic. Boxes with no coordinates are dropped: 13 of the
        16,167 rows carry a study-level label and no geometry, and averaging a NaN into a centre would make
        every term in those studies vanish from both sides at once.

        The 16,167 box coordinates were read and discarded here until 2026-09-11 (knee-1w6t), which is why
        `Medial Meniscus` and `Lateral Meniscus` shipped holding not one positive between them.
        """
        straddling: dict[str, frozenset[str]] = dict.fromkeys(self.listed, frozenset())
        located = self.boxes[self.boxes[X_COLUMN].notna() & self.boxes[WIDTH_COLUMN].notna()].copy()
        located["centre"] = located[X_COLUMN] + located[WIDTH_COLUMN] / 2
        for identifier, group in located.groupby(FILE_COLUMN):
            below = set(group.loc[group["centre"] < COMPARTMENT_TROUGH, LABEL_COLUMN].astype(str))
            above = set(group.loc[group["centre"] >= COMPARTMENT_TROUGH, LABEL_COLUMN].astype(str))
            straddling[str(identifier)] = frozenset(below & above)
        carrying = sum(1 for terms in straddling.values() if terms)
        logger.info(f"{len(located)} located boxes, {carrying} studies carry a term in both compartments")
        return straddling

    def located(self) -> pd.DataFrame:
        """Every box that carries geometry, as drawn: 16,154 of 16,167 rows, the rest study-level labels."""
        located = self.boxes.dropna(subset=list(GEOMETRY))
        return located[[FILE_COLUMN, SLICE_COLUMN, LABEL_COLUMN, *GEOMETRY]].reset_index(drop=True)
