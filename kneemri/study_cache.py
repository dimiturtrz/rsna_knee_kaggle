"""Decode every study once, into a memory-mapped uint8 cache the trainer can read many times.

The pipeline is read-bound, not arithmetic-bound: a study is on the order of 150 files, which is
affordable once and not once per epoch. So the slot images are decoded a single time and held as uint8,

    bytes = studies x slots x slices x pixels^2,

which for 4,407 studies at six slots, twelve slices and 336 pixels is 35.8 GB — memory-mapped rather
than resident, so the trainer pages in what it touches and a restart costs nothing.

The exponent on the resolution is what makes this a real constraint rather than a detail: the cache
grows with the SQUARE of pixels and only linearly with slices, so slice coverage is the cheap axis and
resolution the expensive one.
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from kneemri.slot_reader import IMAGE_SIZE, SlotReader
from kneemri.slots import SLOTS
from kneemri.study import SlotPipeline, Study

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedCorpus:
    """The decoded corpus as the trainer sees it: pixels and the presence mask, indexed alike.

    The two always travel together — a mask without its pixels says a slot was acquired and cannot show
    it — so they are one argument rather than two.
    """

    images: NDArray[np.uint8]
    mask: NDArray[np.float32]

    @property
    def image_size(self) -> int:
        """Pixels per side, read off the cache rather than assumed by whoever consumes it."""
        return int(self.images.shape[-1])

    @property
    def slices(self) -> int:
        """How many slices per slot the cache holds, which bounds the windows a member can read."""
        return int(self.images.shape[2])


class StudyCache:
    """A memory-mapped `[studies, slots, slices, size, size]` uint8 array plus its presence mask."""

    def __init__(
        self, directory: Path, slices: int = 12, image_size: int = IMAGE_SIZE, slots: int = len(SLOTS)
    ) -> None:
        self.directory = directory
        self.slices = slices
        self.image_size = image_size
        self.slots = slots

    @property
    def key(self) -> str:
        """Everything that decides the cache's CONTENTS, in its filename.

        The slot count used to be absent, while `shape` read it from the module-level `SLOTS`. The
        docstring here claimed the name stated everything that decides the contents, and it did not:
        extending the scheme to nine slots — which the census says would recover 96% of the series we
        currently cannot route — would have written a differently-shaped array to the SAME path and
        destroyed the six-slot cache every existing member was fitted on. Same defect as recording a
        module constant in a member's fingerprint instead of the run's own value.

        The canonical six-slot cache keeps its original name so the arrays already on disk, and the
        members fitted on them, stay loadable without renaming a file that a running trainer holds
        memory-mapped. Any other scheme is named explicitly.
        """
        base = f"{self.image_size}px_{self.slices}sl"
        return base if self.slots == len(SLOTS) else f"{base}_{self.slots}slot"

    @property
    def images_path(self) -> Path:
        """Where the pixel cache lives."""
        return self.directory / f"images_{self.key}.npy"

    @property
    def mask_path(self) -> Path:
        """The per-study presence mask, one flag per slot."""
        return self.directory / f"mask_{self.key}.npy"

    @property
    def index_path(self) -> Path:
        """The study order the two arrays are indexed by."""
        return self.directory / f"index_{self.key}.json"

    def shape(self, studies: int) -> tuple[int, int, int, int, int]:
        """The cache's dimensions for a given number of studies."""
        return (studies, self.slots, self.slices, self.image_size, self.image_size)

    def build(self, source: Path, identifiers: list[str], planes: dict[str, str], workers: int = 16) -> None:
        """Decode every study into the cache, in parallel, writing straight into the memory map.

        Workers write disjoint slices of the same file, so no locking is needed and nothing is returned
        through the pool but a flag — the pixels never cross a process boundary.

        Raises `ValueError` if asked for a slot count the decoder cannot fill. The workers route with
        the module-level scheme, so a cache declaring nine slots would be filled six-wide and the
        remaining three left as zeros — an array of exactly the right shape, quietly holding blank
        sequences, which is this project's recurring failure mode rather than a hypothetical one.
        Extending the scheme means changing `SLOTS` and rebuilding, not passing a bigger number here.
        """
        if self.slots != len(SLOTS):
            message = (
                f"cache declares {self.slots} slots but the decoder routes with {len(SLOTS)}; "
                "the extra slots would be written as zeros"
            )
            raise ValueError(message)
        self.directory.mkdir(parents=True, exist_ok=True)
        images = np.lib.format.open_memmap(
            self.images_path, mode="w+", dtype=np.uint8, shape=self.shape(len(identifiers))
        )
        del images
        mask = np.zeros((len(identifiers), self.slots), dtype=np.float32)
        self.index_path.write_text(json.dumps(identifiers), encoding="utf-8")

        arguments = [
            (position, identifier, str(source), str(self.images_path), self.slices, self.image_size, planes)
            for position, identifier in enumerate(identifiers)
        ]
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for position, flags in pool.map(StudyCache.decode_one, arguments, chunksize=8):
                mask[position] = flags
                done += 1
                if done % 200 == 0:
                    logger.info(f"  cached {done}/{len(identifiers)} studies")
        np.save(self.mask_path, mask)

    def load(self) -> tuple[CachedCorpus, list[str]]:
        """The cache, memory-mapped read-only, with its study order."""
        images = np.load(self.images_path, mmap_mode="r")
        mask = np.load(self.mask_path)
        identifiers = json.loads(self.index_path.read_text(encoding="utf-8"))
        return CachedCorpus(images, mask), identifiers

    @staticmethod
    def planes_of(series_table: Path) -> dict[str, str]:
        """Series UID to anatomical plane, the one thing the slot predicate takes from the shipped table."""
        table = pd.read_csv(series_table)
        return table.set_index("SeriesInstanceUID")["Anatomical_Plane"].to_dict()

    @staticmethod
    def decode_one(argument: tuple[int, str, str, str, int, int, dict[str, str]]) -> tuple[int, list[float]]:
        """Decode one study into its slot of the memory map. Top-level so a process pool can pickle it.

        The reader is built at the CACHE's resolution, not at its own default. Taking the default worked
        only while the two happened to agree; at any other resolution the decoded bag simply would not fit
        the map, and the shape error was being caught and reported as an absent study.
        """
        position, identifier, source, images_path, slices, image_size, planes = argument
        images = np.load(images_path, mmap_mode="r+")
        try:
            pipeline = SlotPipeline(reader=SlotReader(image_size=image_size))
            study = Study.from_directory(Path(source) / identifier, planes, pipeline)
            bag, mask = study.bag(slices)
            images[position] = bag.numpy()
            images.flush()
            return position, mask.tolist()
        except (OSError, ValueError, RuntimeError, IndexError) as failure:
            # A study that will not decode is reported ABSENT in every slot rather than black in all of
            # them: the mask can express "not acquired", and a black image the mask calls acquired is a
            # silent lie the head cannot detect. The reason is PRINTED rather than swallowed — an
            # unreported decode failure is indistinguishable from a black knee.
            logger.info(f"  decode failed for {identifier}: {type(failure).__name__}: {failure}")
            return position, [0.0] * len(SLOTS)
