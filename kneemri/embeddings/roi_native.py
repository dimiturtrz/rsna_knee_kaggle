"""The ROI cache again, every sagittal box re-cut from DICOM decoded at twice the cache grid.

`python -m kneemri.embeddings.roi_native --slices 20 --image-size 336 --band-low 0.1 --band-high 0.9`
reads the `roi` cache and its box table, copies every study's six canonical slots, decodes each study's
sagittal series at `SCALE` x the grid (same 130 mm window, same centre, same band) and writes the three
box crops resized DOWN to the grid, as `images_<key>_roinative.npy`. Index and mask are the `roi` ones.

WHY. `roi_cache` upsampled its boxes out of 0.387 mm/px pixels. RSNA sagittal PixelSpacing is p50 0.312
and p10 0.212 mm (200 series, 73% finer than the cache), so an 87 mm median box holds ~1.24x the linear
detail the upsample discarded — and the small-structure labels the crop existed for (Fracture, MCL)
were exactly the ones that fell. The box frame is the 130 mm window at any grid, so a box scales by
`SCALE` and nothing else moves (prototype: crops correlate 0.999 with the `roi` ones).

THE ROUTINE DECODE PATH, not a batch extraction: `Study.bag` over RSNA's own `train_series`, as
`build_cache` does. OFFLINE ONLY — the boxes come from an SKM-TEA-fitted locator (knee-vgz5b.7).
"""

import csv
import json
import logging
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import UInt8
from torch import Tensor

from core import cli
from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.slots import ROI_NATIVE_SCHEME, ROI_SCHEME, SLOTS, Plane
from core.data.corpus.study import SlotPipeline, Study
from core.data.corpus.study_cache import CorpusSplit, StudyCache
from kneemri.embeddings.roi_cache import LABELS
from kneemri.models.locator import BYTE

logger = logging.getLogger(__name__)

SCALE = 2
"""Decode grid over cache grid. 2x the 336 grid is 0.193 mm/px, finer than the p10 acquired spacing."""

WORKERS = 8
"""Decode processes. Measured 2026-09-15: 4407 studies in 28 min beside a training arm that held 2600 img/s."""

NAMES = tuple(slot.name for slot in SLOTS)
SAGITTAL_NAMES = tuple(slot.name for slot in SLOTS if slot.plane is Plane.SAGITTAL)
PLANES: dict[str, str] = {}


@dataclass(frozen=True)
class NativeBox:
    """One acquired sagittal slot's box, in cache-grid pixels, as the `roi` table recorded it."""

    slot: str
    top: int
    left: int
    side: int
    blank: bool

    def scaled(self, scale: int) -> "NativeBox":
        return replace(self, top=self.top * scale, left=self.left * scale, side=self.side * scale)

    @staticmethod
    def read(table: Path) -> dict[str, list["NativeBox"]]:
        boxes: dict[str, list[NativeBox]] = {}
        with table.open(newline="") as handle:
            for row in csv.DictReader(handle):
                box = NativeBox(row["slot"], int(row["top"]), int(row["left"]), int(row["side"]), row["blank"] == "1")
                boxes.setdefault(row["study"], []).append(box)
        return boxes

    def recut(self, stack: UInt8[Tensor, "s h w"], size: int) -> UInt8[Tensor, "s size size"]:
        """This box's window of a stack decoded `SCALE` x finer, resized to the cache grid."""
        box = self.scaled(SCALE)
        window = stack[:, box.top : box.top + box.side, box.left : box.left + box.side].float()
        resized = F.interpolate(window[:, None], size=(size, size), mode="bilinear", align_corners=False)
        return resized[:, 0].round().clamp(0, BYTE).to(torch.uint8)


@dataclass(frozen=True)
class NativeJob:
    """One study's re-cut, as the pool receives it."""

    position: int
    study: str
    boxes: tuple[NativeBox, ...]
    source: StudyCache
    destination: StudyCache
    split: CorpusSplit

    def run(self, planes: dict[str, str]) -> int:  # devtools-ignore: test-mirror
        """Copy the study's `roi` row, then overwrite its sagittal views; 0 when the decode failed.

        A failed decode keeps the upsampled `roi` crops rather than a black image the mask calls acquired.
        """
        images = np.load(self.destination.images_path, mmap_mode="r+")
        images[self.position] = np.load(self.source.images_path, mmap_mode="r")[self.position]
        size = self.source.image_size
        reading = self.source.reading
        pipeline = SlotPipeline(reader=reading.reader(size * SCALE))
        try:
            bag, _ = Study.from_directory(self.split.source / self.study, planes, pipeline).bag(self.source.slices)
        except (OSError, ValueError, RuntimeError, IndexError) as failure:
            logger.info(f"  decode failed for {self.study}: {type(failure).__name__}: {failure}")
            return 0
        for box in self.boxes:
            if not box.blank:
                column = len(SLOTS) + SAGITTAL_NAMES.index(box.slot)
                images[self.position, column] = box.recut(bag[NAMES.index(box.slot)], size).numpy()
        images.flush()
        return 1

    @staticmethod
    def planted(planes: dict[str, str]) -> None:  # devtools-ignore: test-mirror
        """Series UID to plane, planted once per worker rather than pickled into every job."""
        PLANES.update(planes)

    @staticmethod
    def pooled(job: "NativeJob") -> int:  # devtools-ignore: test-mirror
        return job.run(PLANES)


def main() -> None:
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument("--workers", type=int, default=WORKERS)
    arguments = command.parsed()
    root = command.rooted(arguments)
    named = CacheFlags.cache(arguments, root.processed())
    source, destination = named.under(ROI_SCHEME), named.under(ROI_NATIVE_SCHEME)
    if destination.images_path.exists():
        raise FileExistsError(destination.images_path)
    split = CorpusSplit(root.raw())
    studies = source.recorded()
    boxes = NativeBox.read(source.directory / LABELS / f"roi_boxes_{source.key}.csv")
    np.lib.format.open_memmap(
        destination.images_path, mode="w+", dtype=np.uint8, shape=source.shape(len(studies))
    ).flush()
    jobs = [
        NativeJob(position, study, tuple(boxes.get(study, [])), source, destination, split)
        for position, study in enumerate(studies)
    ]
    with ProcessPoolExecutor(arguments.workers, initializer=NativeJob.planted, initargs=(split.planes(),)) as pool:
        decoded = sum(pool.map(NativeJob.pooled, jobs, chunksize=4))
    destination.index_path.write_text(json.dumps(studies))
    shutil.copy(source.mask_path, destination.mask_path)
    command.report([str(destination.images_path), f"decoded {decoded}/{len(studies)}"])


if __name__ == "__main__":
    main()
