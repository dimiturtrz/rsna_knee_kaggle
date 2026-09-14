"""SKM-TEA slices through the DICOM order, the mask transpose and the split table, on a tar the test writes."""

import io
import json
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from core.data.corpus.skmtea_boxes import ANNOTATIONS, SCAN_COLUMN, SPLITS
from core.data.corpus.skmtea_slices import (
    IMAGES,
    LABELS,
    MASKS,
    SIZE,
    SLICE_TABLE,
    SPLIT_COLUMN,
    TARBALLS,
    DicomVolume,
    SkmteaSlices,
    SlicedScan,
)

DEPTH, SIDE = 20, 8
SCAN = "MTR_001"
KEPT = 6
"""Slices 2, 5, 8, 11, 14 and 17 of twenty."""
WHITE_BYTE = 250


def _dataset(echo: int, position: float, value: int, description: str = "Sag DESS") -> Dataset:
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = generate_uid()
    meta.MediaStorageSOPInstanceUID = generate_uid()
    dataset = Dataset()
    dataset.file_meta = meta
    dataset.SeriesDescription = description
    dataset.EchoNumbers = echo
    dataset.ImagePositionPatient = [position, 0.0, 0.0]
    dataset.Rows = dataset.Columns = SIDE
    dataset.BitsAllocated = dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.SamplesPerPixel = 1
    dataset.PixelRepresentation = 0
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.PixelData = np.full((SIDE, SIDE), value, dtype=np.uint16).tobytes()
    return dataset


def _datasets() -> list[Dataset]:
    """Slices stored in DESCENDING position, each slice's value its ascending rank, plus a T2 map to skip."""
    drawn = [_dataset(echo, float(DEPTH - rank), DEPTH - rank, "Sag DESS") for echo in (1, 2) for rank in range(DEPTH)]
    return [*drawn, _dataset(1, 0.0, 999, "NOT DIAGNOSTIC: DESS T2 map")]


def _tar(path: Path) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for index, dataset in enumerate(_datasets()):
            buffer = io.BytesIO()
            dataset.save_as(buffer, enforce_file_format=True)
            info = tarfile.TarInfo(f"{index}.dcm")
            info.size = buffer.tell()
            buffer.seek(0)
            archive.addfile(info, buffer)


def _root(tmp_path: Path) -> Path:
    (tmp_path / TARBALLS).mkdir(parents=True)
    (tmp_path / MASKS).mkdir(parents=True)
    (tmp_path / ANNOTATIONS).mkdir(parents=True)
    _tar(tmp_path / TARBALLS / f"{SCAN}.tar.gz")
    labels = np.zeros((SIDE, SIDE, DEPTH), dtype=np.uint8)
    labels[:, :, 11] = 2
    nib.save(nib.Nifti1Image(labels, np.eye(4)), tmp_path / MASKS / f"{SCAN}.nii.gz")
    for split in SPLITS:
        images: list[dict[str, str]] = [{SCAN_COLUMN: SCAN}] if split == "val" else []
        (tmp_path / ANNOTATIONS / f"{split}.json").write_text(json.dumps({"images": images}))
    return tmp_path


def test_of() -> None:
    volume = DicomVolume.of(_datasets())
    assert volume.echoes.shape == (2, DEPTH, SIDE, SIDE)
    assert volume.echoes[0, :, 0, 0].tolist() == list(range(1, DEPTH + 1))


def test_stream(tmp_path: Path) -> None:
    _tar(tmp_path / "scan.tar.gz")
    assert len(list(DicomVolume.stream(tmp_path / "scan.tar.gz"))) == 2 * DEPTH + 1


def test_read(tmp_path: Path) -> None:
    _tar(tmp_path / "scan.tar.gz")
    assert DicomVolume.read(tmp_path / "scan.tar.gz").echoes.shape == (2, DEPTH, SIDE, SIDE)


def test_cut() -> None:
    volume = DicomVolume.of(_datasets())
    labels = np.zeros((DEPTH, SIDE, SIDE), dtype=np.uint8)
    labels[5] = 3
    sliced = SlicedScan.cut(volume, labels)
    assert sliced.images.shape == (KEPT, 2, SIZE, SIZE)
    assert sliced.masks[:, 0, 0].tolist() == [0, 3, 0, 0, 0, 0]
    assert sliced.images[-1, 0].min() >= WHITE_BYTE


def test_cut_refuses_a_mask_of_another_shape() -> None:
    with pytest.raises(ValueError, match="do not cover"):
        SlicedScan.cut(DicomVolume.of(_datasets()), np.zeros((DEPTH, SIDE, SIDE + 1), dtype=np.uint8))


def test_splits(tmp_path: Path) -> None:
    assert SkmteaSlices(_root(tmp_path)).splits() == {SCAN: "val"}


def test_scan(tmp_path: Path) -> None:
    root = _root(tmp_path)
    sliced = SkmteaSlices(root).scan(root / TARBALLS / f"{SCAN}.tar.gz")
    assert sliced.masks[:, 0, 0].tolist() == [0, 0, 0, 2, 0, 0]


def test_write(tmp_path: Path) -> None:
    root = _root(tmp_path / "raw")
    table = SkmteaSlices(root).write(tmp_path / "out", workers=1)
    assert table[SPLIT_COLUMN].tolist() == ["val"] * KEPT
    assert np.load(tmp_path / "out" / IMAGES).shape == (KEPT, 2, SIZE, SIZE)
    assert np.load(tmp_path / "out" / LABELS).shape == (KEPT, SIZE, SIZE)
    assert (tmp_path / "out" / SLICE_TABLE).exists()
