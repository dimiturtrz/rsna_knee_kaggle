"""Synthetic DICOM series, so the data layer is tested without touching the 570 GB corpus."""

from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

SEED = 2026


def write_series(
    directory: Path,
    slices: int = 8,
    *,
    description: str = "SAG PD FS",
    repetition: float = 3000.0,
    echo: float = 25.0,
    scan_options: str = "FS",
    laterality: str | None = "L",
    spacing: float = 0.3,
    size: int = 64,
    normal: tuple[float, float, float] = (1.0, 0.0, 0.0),
    reversed_names: bool = True,
    seed: int = SEED,
) -> Path:
    """A series whose file NAMES run opposite to its geometry, as the real corpus's UIDs do."""
    directory.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)
    for index in range(slices):
        data = Dataset()
        data.file_meta = FileMetaDataset()
        data.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        data.file_meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
        data.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        data.SOPClassUID = pydicom.uid.MRImageStorage
        data.SOPInstanceUID = data.file_meta.MediaStorageSOPInstanceUID
        data.Modality = "MR"
        data.SeriesDescription = description
        data.SequenceName = ""
        data.ScanOptions = scan_options
        data.ScanningSequence = "SE"
        data.RepetitionTime = repetition
        data.EchoTime = echo
        if laterality is not None:
            data.Laterality = laterality
        data.PixelSpacing = [spacing, spacing]
        data.Rows, data.Columns = size, size
        data.InstanceNumber = index + 1
        # Position advances along the given normal, so k = p.n is strictly increasing with `index`.
        data.ImagePositionPatient = [
            float(normal[0] * index * 3),
            float(normal[1] * index * 3),
            float(normal[2] * index * 3),
        ]
        # cross([0,1,0], [0,0,1]) = (+1,0,0): the normal points along +x, so k = p.n INCREASES with
        # `index` while the file names run the other way — which is the disagreement being tested.
        data.ImageOrientationPatient = (
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0] if normal == (0.0, 0.0, 1.0) else [0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        )
        data.BitsAllocated = 16
        data.BitsStored = 16
        data.HighBit = 15
        data.PixelRepresentation = 0
        data.SamplesPerPixel = 1
        data.PhotometricInterpretation = "MONOCHROME2"
        pixels = (generator.integers(0, 4096, (size, size))).astype(np.uint16)
        pixels[index % size, :] = 4095  # a per-slice marker, so order is checkable from the pixels
        data.PixelData = pixels.tobytes()
        # Names deliberately counter-ordered: sorting by file name must NOT reproduce the stack.
        stem = f"{slices - index:04d}" if reversed_names else f"{index:04d}"
        data.save_as(directory / f"{stem}.dcm", enforce_file_format=True)
    return directory


@pytest.fixture
def series(tmp_path: Path) -> Path:
    """One sagittal PD fat-suppressed series of eight slices."""
    return write_series(tmp_path / "study" / "series")
