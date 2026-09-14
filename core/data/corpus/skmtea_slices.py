"""SKM-TEA sagittal DESS slices with their tissue masks, at the size the RSNA 336 px cache holds.

`python -m core.data.corpus.skmtea_slices` writes `skmtea_images_336.npy` (slice, echo, rows, columns),
`skmtea_masks_336.npy` (slice, rows, columns) and `skmtea_slices_336.csv` (scan and split per slice)
under the processed root's `locator/`.

THE IMAGES EXIST ONLY AS DICOM TARBALLS. The raw-data-track h5 files hold k-space; `dicoms/MTR_*.tar.gz`
(155 files, 19 GB) hold the reconstructions. Each tar carries 480 files: the two echoes of the sagittal
DESS at 160 slices of 512 x 512, plus a series described `NOT DIAGNOSTIC: DESS T2 map`, which is
skipped. A tar is read as a stream and decoded in memory; nothing is extracted to disk.

THE ALIGNMENT IS MEASURED, BY OVERLAY: slices sorted by `ImagePositionPatient[0]` ascending sit on the
dicom-track mask transposed (2, 1, 0) — (slice LR, row SI, column AP) — and the reverse order visibly
misses. The picture that results has the femur at the top and anterior on the left, which is how the
RSNA sagittal slots are drawn.
"""

import io
import json
import logging
import tarfile
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn.functional as F
from jaxtyping import Float32, UInt8
from nibabel.nifti1 import Nifti1Image
from pydicom.dataset import Dataset

from core import cli
from core.data.corpus.fastmri_corpus import PROGRESS_EVERY
from core.data.corpus.skmtea_boxes import ANNOTATIONS, SCAN_COLUMN, SKMTEA, SPLITS

logger = logging.getLogger(__name__)

TARBALLS = "dicoms"
MASKS = "segmentation_masks/dicom-track"
DICOM_TRACK = (2, 1, 0)
"""(AP, SI, LR) on disk -> (LR, SI, AP): slice, row, column."""
NOT_DIAGNOSTIC = "NOT DIAG"
ECHOES = (1, 2)
SIZE = 336
"""The RSNA cache's `images_336px_*` side, so a slice trained here is drawn at the scale it is read at."""
BAND = (0.1, 0.9)
"""The RSNA cache's slice band: the outer tenth of a sagittal stack is past the condyles."""
STEP = 3
"""Every third slice: 1.6 mm slices, so neighbours 4.8 mm apart rather than near-duplicates."""
WHITE = 0.995
"""The intensity quantile drawn as white, per echo: a few flow-artefact voxels would otherwise set it."""
QUANTILE_STRIDE = 97
"""`torch.quantile` refuses inputs past 16M elements; one echo is 41M, and a stride-97 subsample is not."""
BYTE = 255
WORKERS = 12
LOCATOR = "locator"
IMAGES = f"skmtea_images_{SIZE}.npy"
LABELS = f"skmtea_masks_{SIZE}.npy"
SLICE_TABLE = f"skmtea_slices_{SIZE}.csv"
SPLIT_COLUMN = "split"


@dataclass(frozen=True)
class DicomVolume:
    """One scan's echoes, stacked slice-first in the order the mask is drawn in."""

    echoes: Float32[np.ndarray, "echo depth rows columns"]

    @staticmethod
    def of(datasets: Iterable[Dataset]) -> "DicomVolume":
        slices: dict[int, list[tuple[float, np.ndarray]]] = {echo: [] for echo in ECHOES}
        for dataset in datasets:
            if NOT_DIAGNOSTIC in str(dataset.SeriesDescription):
                continue
            position = float(dataset.ImagePositionPatient[0])
            slices[int(dataset.EchoNumbers)].append((position, dataset.pixel_array.astype(np.float32)))
        return DicomVolume(
            np.stack([np.stack([pixels for _, pixels in sorted(slices[echo], key=lambda s: s[0])]) for echo in ECHOES])
        )

    @staticmethod
    def read(tarball: Path) -> "DicomVolume":
        return DicomVolume.of(DicomVolume.stream(tarball))

    @staticmethod
    def stream(tarball: Path) -> Iterator[Dataset]:
        """Every DICOM in a gzipped tar, decoded in memory in the order the tar holds them."""
        with tarfile.open(tarball, "r|gz") as archive:
            for member in archive:
                handle = archive.extractfile(member) if member.isfile() else None
                if handle is not None:
                    yield pydicom.dcmread(io.BytesIO(handle.read()))


@dataclass(frozen=True)
class SlicedScan:
    """A scan's kept slices, both echoes and the mask, resized to `SIZE`."""

    images: UInt8[np.ndarray, "slice echo size size"]
    masks: UInt8[np.ndarray, "slice size size"]

    @staticmethod
    def cut(volume: DicomVolume, labels: UInt8[np.ndarray, "depth rows columns"]) -> "SlicedScan":
        if volume.echoes.shape[1:] != labels.shape:
            message = f"echoes {volume.echoes.shape} do not cover mask {labels.shape}"
            raise ValueError(message)
        depth = labels.shape[0]
        keep = np.arange(int(BAND[0] * depth), int(BAND[1] * depth), STEP)
        pixels = torch.from_numpy(volume.echoes[:, keep])
        white = torch.quantile(pixels.flatten(1)[:, ::QUANTILE_STRIDE], WHITE, dim=1).clamp(min=1e-6)
        pixels = pixels / white.view(-1, 1, 1, 1)
        images = F.interpolate(pixels.transpose(0, 1), size=SIZE, mode="bilinear", antialias=True).clamp(0, 1)
        masks = F.interpolate(torch.from_numpy(labels[keep]).float()[:, None], size=SIZE, mode="nearest")[:, 0]
        return SlicedScan((images * BYTE).byte().numpy(), masks.byte().numpy())


@dataclass(frozen=True)
class SkmteaSlices:
    """The SKM-TEA raw root, read into the slice arrays the locator trains on."""

    raw: Path

    def splits(self) -> dict[str, str]:
        """Scan id to split, from the v1.0.0 annotation files — 86 / 33 / 36 scans on the full corpus."""
        return {
            image[SCAN_COLUMN]: split
            for split in SPLITS
            for image in json.loads((self.raw / ANNOTATIONS / f"{split}.json").read_text())["images"]
        }

    def scan(self, tarball: Path) -> SlicedScan:
        scan = tarball.name.split(".")[0]
        labels = np.asarray(Nifti1Image.from_filename(self.raw / MASKS / f"{scan}.nii.gz").dataobj, dtype=np.uint8)
        return SlicedScan.cut(DicomVolume.read(tarball), np.ascontiguousarray(labels.transpose(DICOM_TRACK)))

    def write(self, out: Path, workers: int = WORKERS) -> pd.DataFrame:
        out.mkdir(parents=True, exist_ok=True)
        splits = self.splits()
        found = sorted((self.raw / TARBALLS).glob("*.tar.gz"))
        tarballs = [path for path in found if path.name.split(".")[0] in splits]
        rows: list[str] = []
        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        with ProcessPoolExecutor(workers) as pool:
            for index, (tarball, sliced) in enumerate(zip(tarballs, pool.map(self.scan, tarballs), strict=True)):
                rows += [tarball.name.split(".")[0]] * len(sliced.masks)
                images.append(sliced.images)
                masks.append(sliced.masks)
                if index % PROGRESS_EVERY == 0:
                    logger.info("%d/%d %s %s", index, len(tarballs), tarball.name, sliced.images.shape)
        np.save(out / IMAGES, np.concatenate(images))
        np.save(out / LABELS, np.concatenate(masks))
        table = pd.DataFrame({SCAN_COLUMN: rows, SPLIT_COLUMN: [splits[scan] for scan in rows]})
        table.to_csv(out / SLICE_TABLE, index=False)
        return table


def main() -> None:
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--workers", type=int, default=WORKERS)
    arguments = command.parsed()
    root = command.rooted(arguments)
    table = SkmteaSlices(root.raw(SKMTEA)).write(root.processed() / LOCATOR, arguments.workers)
    command.report(f"{split}: {count} slices" for split, count in table[SPLIT_COLUMN].value_counts().items())
    logger.info("SKMTEA SLICES COMPLETE")


if __name__ == "__main__":
    main()
