"""fastMRI+ boxes in the three frames they get used in: as drawn, in the h5 volume, and in our cache.

AS DRAWN is not the h5 frame. The readers labelled DICOMs made by flipping each reconstruction up/down
(the fastMRI+ README says so), and an overlay settles it: read unflipped, a `Meniscus Tear` box sits
inside the femoral condyle; flipped, it sits on the meniscus, and an ACL box sits in the notch. So a
box's native top row is `rows - y - height`, and `x` is untouched — which is why the x-only compartment
rule in `FastmriCorpus.straddled` was never wrong about it.

THE CACHE FRAME IS REACHED THROUGH `SlotReader` ITSELF — the band, the window and the resize that
`FastmriCorpus.stack` takes — because the crop centre is a strategy, and a re-derived window that
disagreed with the cache by one centre would place every box beside its lesion with every gate green.

The ORIGINAL slice index is kept. The nearest kept band slice and its distance in original slices sit
beside it, so a box outside the band is marked rather than dropped (knee-s10a); what distance is still
close enough is a training decision, not a decode one.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from jaxtyping import Float

from core.data.corpus.fastmri_corpus import (
    FILE_COLUMN,
    LABEL_COLUMN,
    PROGRESS_EVERY,
    RECONSTRUCTION,
    SLICE_COLUMN,
    WIDTH_COLUMN,
    X_COLUMN,
    FastmriVolume,
)
from core.data.corpus.study_cache import StudyCache

logger = logging.getLogger(__name__)

BOXES = "knee.csv"
FILE_LIST = "knee_file_list.csv"
FILE_LIST_FOOTER = "Knee data"
BOX_TABLE = "fastmri_boxes_{key}.csv"
STUDY_TABLE = "fastmri_box_studies_{key}.csv"
Y_COLUMN = "y"
HEIGHT_COLUMN = "height"
GEOMETRY = (X_COLUMN, Y_COLUMN, WIDTH_COLUMN, HEIGHT_COLUMN)
NATIVE = ("native_x0", "native_y0", "native_x1", "native_y1")
CACHED = ("cache_x0", "cache_y0", "cache_x1", "cache_y1")


@dataclass(frozen=True)
class BoxFrame:
    """Where one reconstruction's pixels land in the cache: the kept slices, the window, the resize."""

    kept: tuple[int, ...]
    top: int
    left: int
    rows: int
    columns: int
    size: int

    def place(self, boxes: pd.DataFrame) -> pd.DataFrame:
        """Native boxes with their cache slice, their distance from it, and their cache corners.

        Corners are pixel EDGES, so the bilinear resize (`align_corners=False`) is a pure scale about the
        window's own origin. Clipped to the frame; a box the window cut away entirely reads `visible`
        False rather than vanishing.
        """
        gaps = np.abs(boxes[SLICE_COLUMN].to_numpy()[:, None] - np.asarray(self.kept)[None, :])
        placed = boxes.copy()
        placed["cache_slice"] = gaps.argmin(axis=1)
        placed["slice_gap"] = gaps.min(axis=1)
        scales = (self.size / self.columns, self.size / self.rows)
        offsets = (self.left, self.top)
        for index, (native, cached) in enumerate(zip(NATIVE, CACHED, strict=True)):
            axis = index % 2
            placed[cached] = ((boxes[native] - offsets[axis]) * scales[axis]).clip(0, self.size)
        placed["visible"] = (placed["cache_x1"] > placed["cache_x0"]) & (placed["cache_y1"] > placed["cache_y0"])
        return placed


class FastmriBoxes:
    """The fastMRI+ box table, decoded per study into the frame of one `StudyCache`."""

    def __init__(self, annotations: Path, cache: StudyCache) -> None:
        self.annotations = annotations
        self.cache = cache

    def read(self) -> pd.DataFrame:
        """Every box that carries geometry, as drawn: 16,154 of 16,167 rows, the rest study-level labels."""
        boxes = pd.read_csv(self.annotations / BOXES)
        boxes.columns = [str(column).strip() for column in boxes.columns]
        located = boxes.dropna(subset=list(GEOMETRY))
        return located[[FILE_COLUMN, SLICE_COLUMN, LABEL_COLUMN, *GEOMETRY]].reset_index(drop=True)

    def listed(self) -> frozenset[str]:
        """The studies a reader opened. The list has no header; its last line, `Knee data`, is a footer."""
        listed = pd.read_csv(self.annotations / FILE_LIST, header=None).iloc[:, 0].astype(str)
        return frozenset(listed) - {FILE_LIST_FOOTER}

    @staticmethod
    def native(boxes: pd.DataFrame, rows: int) -> pd.DataFrame:
        """Boxes with their corners in the h5 volume's own row order — the drawn image was flipped up/down."""
        flipped = boxes.copy()
        flipped[NATIVE[0]] = boxes[X_COLUMN]
        flipped[NATIVE[1]] = rows - boxes[Y_COLUMN] - boxes[HEIGHT_COLUMN]
        flipped[NATIVE[2]] = boxes[X_COLUMN] + boxes[WIDTH_COLUMN]
        flipped[NATIVE[3]] = rows - boxes[Y_COLUMN]
        return flipped

    def frame(self, pixels: Float[np.ndarray, "depth rows columns"], spacing: float) -> BoxFrame:
        """The frame `FastmriCorpus.stack` puts this volume in, asked of the same reader."""
        reader = self.cache.reader()
        kept = reader.band_indices(len(pixels), self.cache.slices)
        taken = reader.window(pixels[kept], spacing)
        rows, columns = (slice(0, pixels.shape[1]), slice(0, pixels.shape[2])) if taken is None else taken
        return BoxFrame(
            kept=tuple(kept),
            top=int(rows.start),
            left=int(columns.start),
            rows=int(rows.stop - rows.start),
            columns=int(columns.stop - columns.start),
            size=self.cache.image_size,
        )

    def tables(self, volumes: list[FastmriVolume]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """One row per box, and one per READ study on disk — a study with no box keeps its row at zero.

        The study rows are where the negatives live: a box table alone cannot say that a reader opened a
        study and found nothing, and those are a sixth of the corpus.
        """
        boxes = self.read()
        grouped = dict(tuple(boxes.groupby(FILE_COLUMN)))
        listed = self.listed()
        placed: list[pd.DataFrame] = []
        studies: list[dict[str, str | int | float]] = []
        for index, volume in enumerate(volume for volume in volumes if volume.identifier in listed):
            with h5py.File(volume.path, "r") as handle:
                pixels = np.asarray(handle[RECONSTRUCTION][:], dtype=np.float32)
            frame = self.frame(pixels, volume.spacing)
            own = grouped.get(volume.identifier, boxes.iloc[:0])
            placed.append(frame.place(self.native(own, pixels.shape[1])))
            studies.append(
                {
                    FILE_COLUMN: volume.identifier,
                    "slot": volume.slot,
                    "boxes": len(own),
                    "depth": len(pixels),
                    "kept": " ".join(str(kept) for kept in frame.kept),
                    "top": frame.top,
                    "left": frame.left,
                    "rows": frame.rows,
                    "columns": frame.columns,
                    "size": frame.size,
                    "spacing_mm": volume.spacing,
                }
            )
            if index % PROGRESS_EVERY == 0:
                logger.info(f"{index} studies framed")
        return pd.concat(placed, ignore_index=True), pd.DataFrame(studies)

    def write(self, volumes: list[FastmriVolume], directory: Path) -> None:
        """Both tables under `directory`, named for the cache key — a box in a 224 frame is not one in a 336 frame."""
        boxes, studies = self.tables(volumes)
        directory.mkdir(parents=True, exist_ok=True)
        boxes.to_csv(directory / BOX_TABLE.format(key=self.cache.key), index=False)
        studies.to_csv(directory / STUDY_TABLE.format(key=self.cache.key), index=False)
        logger.info(
            f"{len(boxes)} boxes over {int((studies['boxes'] > 0).sum())} studies, "
            f"{int((studies['boxes'] == 0).sum())} read with none, {int((~boxes['visible']).sum())} cropped away, "
            f"{int((boxes['slice_gap'] > 0).sum())} off a kept slice -> {directory}"
        )
