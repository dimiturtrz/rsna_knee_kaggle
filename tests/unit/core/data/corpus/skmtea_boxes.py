"""SKM-TEA boxes through the axis order, the mask regions and the side, on a corpus the test writes itself."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from core.data.corpus.fastmri_boxes import BoxFrame
from core.data.corpus.skmtea_boxes import (
    ANNOTATIONS,
    BOX_TABLE,
    MASKS,
    METADATA,
    STUDY_TABLE,
    SkmteaBoxes,
    TissueMask,
)

SI, AP, LR = 8, 10, 6
LATERAL_BOX = [5.0, 4.0, 4.0, 2.0, 4.0, 2.0]
PATELLAR_BOX = [1.0, 0.0, 2.0, 2.0, 2.0, 2.0]
MALFORMED_BOX = [5.0, 4.0, 0.0, -2.0, 4.0, 2.0]
EMPTY_BOX = [0.0, 8.0, 0.0, 1.0, 1.0, 1.0]


def _labels() -> np.ndarray:
    """A (SI, AP, LR) mask: medial tibial cartilage on the low LR slices, lateral on the high, a patella."""
    labels = np.zeros((SI, AP, LR), dtype=np.uint8)
    labels[5:7, 4:8, 0:2] = 3
    labels[5:7, 4:8, 4:6] = 4
    labels[1:3, 0:2, 2:4] = 1
    return labels


def _root(tmp_path: Path) -> Path:
    """Two splits whose image ids collide, a third with a scan and no box, and one malformed box."""
    (tmp_path / ANNOTATIONS).mkdir(parents=True)
    (tmp_path / MASKS).mkdir(parents=True)
    categories = [{"id": 1, "name": "Cartilage Lesion (1)", "supercategory": "Cartilage Lesion"}]
    tissues = [{"id": 6, "name": "Tibial Cartilage"}]
    boxes: dict[str, list[list[float]]] = {
        "train": [LATERAL_BOX],
        "val": [PATELLAR_BOX, MALFORMED_BOX],
        "test": [],
    }
    for index, (split, drawn) in enumerate(boxes.items()):
        scan = f"MTR_{index:03d}"
        document = {
            "categories": categories,
            "tissues": tissues,
            "images": [
                {
                    "id": 1,
                    "scan_id": scan,
                    "subject_id": index,
                    "timepoint": 0,
                    "voxel_spacing": [0.3125, 0.3125, 0.8],
                    "orientation": ["SI", "AP", "LR"],
                }
            ],
            "annotations": [
                {"id": n, "image_id": 1, "category_id": 1, "tissue_id": 6, "bbox": box, "confidence": 3.0, "labeler": 1}
                for n, box in enumerate(drawn)
            ],
        }
        (tmp_path / ANNOTATIONS / f"{split}.json").write_text(json.dumps(document))
        nib.save(nib.Nifti1Image(_labels(), np.eye(4)), tmp_path / MASKS / f"{scan}.nii.gz")
    pd.DataFrame({"MTR_ID": ["MTR_000", "MTR_001", "MTR_002"], "MedialDirection": ["L", "R", "L"]}).to_csv(
        tmp_path / METADATA, index=False
    )
    return tmp_path


def _mask(tmp_path: Path) -> TissueMask:
    return TissueMask.read(_root(tmp_path) / MASKS / "MTR_000.nii.gz")


def test_tissue_mask_read(tmp_path: Path):
    """(SI, AP, LR) on disk comes back slice-first: (LR, SI, AP)."""
    mask = _mask(tmp_path)

    assert mask.labels.shape == (LR, SI, AP)
    assert mask.labels[4, 5, 4] == 4, "voxel (SI 5, AP 4, LR 4) sits at slice 4, row 5, column 4"


def test_skmtea_boxes_read(tmp_path: Path):
    """Every box and every scan across the splits, each box named for its own split's scan."""
    boxes, images = SkmteaBoxes(_root(tmp_path)).read()

    assert boxes["scan_id"].tolist() == ["MTR_000", "MTR_001", "MTR_001"], "image id 1 in val is not train's"
    assert images["split"].tolist() == ["train", "val", "test"]
    assert boxes.loc[0, "tissue"] == "Tibial Cartilage"


def test_centre(tmp_path: Path):
    """The mean slice of the named labels."""
    mask = _mask(tmp_path)

    assert mask.centre((3,)) == 0.5
    assert mask.centre((4,)) == 4.5


def test_native():
    """SI is the row, AP the column, LR the slice; a negative extent is kept and marked; `place` reads it."""
    drawn = pd.DataFrame({"scan_id": ["a", "a"], "bbox": [LATERAL_BOX, MALFORMED_BOX]})

    native = SkmteaBoxes.native(drawn, (LR, SI, AP))

    first = native.iloc[0]
    assert (first["native_y0"], first["native_y1"], first["native_x0"], first["native_x1"]) == (5.0, 7.0, 4.0, 8.0)
    assert (first["slice_first"], first["slice_last"], first["slice"]) == (4.0, 6.0, 5)
    assert native["valid"].tolist() == [True, False]
    placed = BoxFrame(kept=(1, 5), top=0, left=0, rows=SI, columns=AP, size=AP).place(native)
    assert placed.loc[0, "slice_gap"] == 0, "the fastMRI+ placement takes this table as it stands"


def test_regions(tmp_path: Path):
    """The dominant tissue, its share of the box, and the nearer compartment — background when nothing is there."""
    mask = _mask(tmp_path)
    drawn = pd.DataFrame({"bbox": [LATERAL_BOX, PATELLAR_BOX, EMPTY_BOX]})

    read = mask.regions(SkmteaBoxes.native(drawn, mask.labels.shape))

    assert read["region"].tolist() == ["tibial cartilage lateral", "patellar cartilage", "background"]
    assert read["region_share"].tolist() == [1.0, 1.0, 0.0]
    assert read["side"].tolist()[0] == "lateral"
    assert read["side"].tolist()[2] == "medial"


def test_tables(tmp_path: Path):
    """Colliding image ids resolve by split, and the scan without a box keeps its row at zero."""
    boxes, studies = SkmteaBoxes(_root(tmp_path)).tables()

    assert len(boxes) == 3
    assert boxes.groupby("scan_id").size().to_dict() == {"MTR_000": 1, "MTR_001": 2}
    assert studies["boxes"].tolist() == [1, 2, 0]
    assert studies["medial_direction"].tolist() == ["L", "R", "L"]
    assert int((~boxes["valid"]).sum()) == 1


def test_write(tmp_path: Path):
    """Both tables land under their names."""
    SkmteaBoxes(_root(tmp_path)).write(tmp_path / "labels")

    assert len(pd.read_csv(tmp_path / "labels" / BOX_TABLE)) == 3
    assert len(pd.read_csv(tmp_path / "labels" / STUDY_TABLE)) == 3
