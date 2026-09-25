"""SKM-TEA boxes and tissue masks in the frame `fastmri_boxes` places from: slice, then rows, then columns.

`python -m core.data.corpus.skmtea_boxes` writes both tables under the processed root's `labels/`.

A BOX IS DRAWN IN THE RAW-DATA-TRACK MASK'S OWN ARRAY ORDER, (SI, AP, LR): `bbox` is the three starts then
the three extents. Measured, not read — no SKM-TEA document states it: over the first sixty boxes the
fraction of labelled voxels inside a box is 0.228 in that order and 0.06 or less under every other
permutation, and in the dicom-track mask the winner moves to the permutation its own axcodes predict.

The frame `BoxFrame.place` expects is the h5 one: a sagittal slice index, then image rows, then columns.
So LR is the slice, SI the row and AP the column, and a box's corners are named `NATIVE` exactly as a
fastMRI+ box's are. The ORIGINAL slice indices are kept — 18 of 155 scans run RL rather than LR, and they
are recorded rather than flipped, because the side a box sits on is read from the mask, which is anatomy.

The mask values are the six segmentation channels of StanfordMIMI/skm-tea `register.py`, which counts them
from zero; on disk they are one-based with 0 for background. Checked against the boxes' own tissue tags:
patellar-cartilage boxes hold mostly 1, femoral 2, meniscus 5 and 6.

Box `image_id`s repeat across the train/val/test files, so a box is keyed by its split AND its image id;
`scan_id` is unique over all 155 scans.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import UInt8
from nibabel.nifti1 import Nifti1Image

from core import cli
from core.data.corpus.fastmri_annotations import LABEL_COLUMN, SLICE_COLUMN
from core.data.corpus.fastmri_boxes import NATIVE
from core.data.corpus.foreign_build import PROGRESS_EVERY

logger = logging.getLogger(__name__)

SKMTEA = "skmtea"
ANNOTATIONS = "annotations/v1.0.0"
SPLITS = ("train", "val", "test")
MASKS = "segmentation_masks/raw-data-track"
METADATA = "all_metadata.csv"
BOX_TABLE = "skmtea_boxes.csv"
STUDY_TABLE = "skmtea_box_studies.csv"
SCAN_COLUMN = "scan_id"
SAGITTAL = (2, 0, 1)
"""(SI, AP, LR) -> (LR, SI, AP): slice, row, column."""
TISSUES = (
    "background",
    "patellar cartilage",
    "femoral cartilage",
    "tibial cartilage medial",
    "tibial cartilage lateral",
    "meniscus medial",
    "meniscus lateral",
)
MEDIAL = (3, 5)
LATERAL = (4, 6)
SIDES = np.array(["lateral", "medial"])
FIRST = ("native_y0", "native_x0", "slice_first")
LAST = ("native_y1", "native_x1", "slice_last")


@dataclass(frozen=True)
class TissueMask:
    """One scan's segmentation, turned into slice-first order."""

    labels: UInt8[np.ndarray, "depth rows columns"]

    @staticmethod
    def read(path: Path) -> "TissueMask":
        return TissueMask(
            np.ascontiguousarray(
                np.asarray(Nifti1Image.from_filename(path).dataobj, dtype=np.uint8).transpose(SAGITTAL)
            )
        )

    def centre(self, labels: tuple[int, ...]) -> float:
        """The mean slice of every voxel carrying one of `labels` — where one side of the joint sits."""
        return float(np.nonzero(np.isin(self.labels, labels))[0].mean())

    def regions(self, boxes: pd.DataFrame) -> pd.DataFrame:
        """Boxes with the tissue most of their labelled voxels carry, its share of the box, and the side.

        The side is the nearer, along the slice axis, of the medial and lateral compartments' centres, so a
        femoral or patellar box — tissues the segmentation does not split — still gets one. A box that
        holds no labelled voxel, or that is malformed, reads `background` at share 0.
        """
        read = boxes.copy()
        tissues: list[int] = []
        shares: list[float] = []
        for box in boxes[[*FIRST, *LAST]].to_numpy(dtype=int):
            low = np.clip(box[:3][[2, 0, 1]], 0, self.labels.shape)
            high = np.clip(box[3:][[2, 0, 1]], low, self.labels.shape)
            counts = np.bincount(self.labels[low[0] : high[0], low[1] : high[1], low[2] : high[2]].ravel(), minlength=7)
            tissue = (int(counts[1:].argmax()) + 1) * int(counts[1:].any())
            tissues.append(tissue)
            shares.append(float(counts[tissue] * bool(tissue) / max(counts.sum(), 1)))
        read["tissue_label"] = tissues
        read["region"] = np.array(TISSUES)[tissues]
        read["region_share"] = shares
        middle = (boxes["slice_first"] + boxes["slice_last"]) / 2
        nearer = (middle - self.centre(MEDIAL)).abs() < (middle - self.centre(LATERAL)).abs()
        read["side"] = SIDES[nearer.to_numpy(dtype=int)]
        return read


class SkmteaBoxes:
    """The SKM-TEA detection boxes, decoded per scan against its tissue mask."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def read(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Every box as drawn, and every scan in the three split files, one row each."""
        boxes: list[dict[str, object]] = []
        images: list[dict[str, object]] = []
        for split in SPLITS:
            document = json.loads((self.root / ANNOTATIONS / f"{split}.json").read_text())
            categories = {category["id"]: category for category in document["categories"]}
            tissues = {tissue["id"]: tissue["name"] for tissue in document["tissues"]}
            named = {image["id"]: image[SCAN_COLUMN] for image in document["images"]}
            images += [{**image, "split": split} for image in document["images"]]
            boxes += [
                {
                    SCAN_COLUMN: named[box["image_id"]],
                    "split": split,
                    "annotation": box["id"],
                    LABEL_COLUMN: categories[box["category_id"]]["name"],
                    "supercategory": categories[box["category_id"]]["supercategory"],
                    "tissue": tissues.get(box["tissue_id"], ""),
                    "confidence": box["confidence"],
                    "labeler": box["labeler"],
                    "bbox": box["bbox"],
                }
                for box in document["annotations"]
            ]
        return pd.DataFrame(boxes), pd.DataFrame(images)

    @staticmethod
    def native(boxes: pd.DataFrame, shape: tuple[int, int, int]) -> pd.DataFrame:
        """Corners in the h5 frame, from a (SI, AP, LR) box; `valid` is False for a negative or overflowing one.

        `shape` is the mask's slice-first shape. Nine of 476 boxes are not valid — a negative extent, or a
        start of 2478 in a 512 matrix — and they are kept and marked rather than repaired, since no reading
        of them is more than a guess.
        """
        drawn = np.asarray(boxes["bbox"].tolist(), dtype=float).reshape(-1, 6)
        placed = boxes.drop(columns="bbox")
        for axis, (first, last) in enumerate(zip(FIRST, LAST, strict=True)):
            placed[first] = drawn[:, axis]
            placed[last] = drawn[:, axis] + drawn[:, axis + 3]
        placed[SLICE_COLUMN] = ((placed["slice_first"] + placed["slice_last"]) // 2).astype(int)
        drawn_shape = np.asarray(shape)[[1, 2, 0]]
        placed["valid"] = ((drawn[:, 3:] > 0) & (drawn[:, :3] >= 0) & (drawn[:, :3] + drawn[:, 3:] <= drawn_shape)).all(
            axis=1
        )
        return placed[[column for column in placed if column not in NATIVE] + list(NATIVE)]

    def tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """One row per box, and one per scan — a scan with no box keeps its row at zero."""
        boxes, images = self.read()
        grouped = dict(tuple(boxes.groupby(SCAN_COLUMN)))
        medial = pd.read_csv(self.root / METADATA).set_index("MTR_ID")["MedialDirection"]
        placed: list[pd.DataFrame] = []
        studies: list[dict[str, object]] = []
        for index, image in enumerate(images.to_dict("records")):
            scan = str(image[SCAN_COLUMN])
            spacing = list(image["voxel_spacing"])
            mask = TissueMask.read(self.root / MASKS / f"{scan}.nii.gz")
            own = grouped.get(scan, boxes.iloc[:0])
            placed.append(mask.regions(self.native(own, mask.labels.shape)))
            studies.append(
                {
                    SCAN_COLUMN: scan,
                    "split": image["split"],
                    "subject": image["subject_id"],
                    "timepoint": image["timepoint"],
                    "boxes": len(own),
                    "depth": mask.labels.shape[0],
                    "rows": mask.labels.shape[1],
                    "columns": mask.labels.shape[2],
                    "orientation": " ".join(map(str, list(image["orientation"]))),
                    "medial_direction": medial[scan],
                    "slice_mm": spacing[2],
                    "pixel_mm": spacing[0],
                    "medial_slice": mask.centre(MEDIAL),
                    "lateral_slice": mask.centre(LATERAL),
                }
            )
            if index % PROGRESS_EVERY == 0:
                logger.info(f"{index} scans read")
        return pd.concat(placed, ignore_index=True), pd.DataFrame(studies)

    def write(self, directory: Path) -> None:
        boxes, studies = self.tables()
        directory.mkdir(parents=True, exist_ok=True)
        boxes.to_csv(directory / BOX_TABLE, index=False)
        studies.to_csv(directory / STUDY_TABLE, index=False)
        logger.info(
            f"{len(boxes)} boxes over {int((studies['boxes'] > 0).sum())} of {len(studies)} scans, "
            f"{int((~boxes['valid']).sum())} malformed -> {directory}"
        )


def main() -> None:
    command = cli.Command(__doc__, logger)
    arguments = command.parsed()
    root = command.rooted(arguments)
    SkmteaBoxes(root.raw(SKMTEA)).write(root.processed() / "labels")
    logger.info("SKMTEA BOXES COMPLETE")


if __name__ == "__main__":
    main()
