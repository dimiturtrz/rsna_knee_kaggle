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
from jaxtyping import Float
from numpy.typing import NDArray

from kneemri.crop_centre import Crop
from kneemri.slot_reader import IMAGE_SIZE, SlotReader
from kneemri.slots import SCHEMES, SlotScheme
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


@dataclass(frozen=True)
class DecodeRequest:
    """One study's decode job, as the pool receives it.

    These seven values were a positional tuple that had to be unpacked identically in two places, and
    the slot SCHEME was about to become an eighth. A worker argument is configuration, not control
    flow, so it gets a type — and the type is what makes it impossible to pass the scheme in the
    position where the resolution used to be.
    """

    position: int
    identifier: str
    source: str
    images_path: str
    slices: int
    image_size: int
    scheme: str
    planes: dict[str, str]
    crop: Crop = Crop()


class StudyCache:
    """A memory-mapped `[studies, slots, slices, size, size]` uint8 array plus its presence mask."""

    CANONICAL = "canonical"
    """The scheme whose caches keep their historic names, and the one the published members use."""

    def __init__(
        self,
        directory: Path,
        slices: int = 12,
        image_size: int = IMAGE_SIZE,
        scheme: str = CANONICAL,
        crop: Crop | None = None,
    ) -> None:
        self.directory = directory
        self.slices = slices
        self.image_size = image_size
        self.scheme = scheme
        self.crop = crop or Crop()

    @property
    def slots(self) -> int:
        """How wide the slot axis is — derived from the scheme, so the two cannot disagree."""
        return len(SCHEMES[self.scheme])

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

        THE PHYSICAL CROP IS HERE FOR THE SAME REASON THE SLOT COUNT IS, and it was missing. An 80 mm
        cache at 336 px and 12 slices would have written to `images_336px_12sl.npy` — the path the 130 mm
        arrays already occupy, which every existing member was fitted on — and destroyed them, silently,
        with no shape change to notice. The default crop keeps the historic name for the same reason the
        canonical scheme does.
        """
        base = f"{self.image_size}px_{self.slices}sl{self.crop.suffix()}"
        return base if self.scheme == self.CANONICAL else f"{base}_{self.scheme}"

    @property
    def scheme_slots(self) -> list[str]:
        """The slot names this cache is built over, in order — the width of its second axis."""
        return [slot.name for slot in SCHEMES[self.scheme]]

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

        A cache cannot declare more slots than its decoder fills: `slots` is DERIVED from the scheme
        and the scheme is what the workers route with, so the two cannot disagree. This briefly had a
        ValueError guarding exactly that mismatch, which the derivation then made unreachable — an
        invariant is better held by construction than by a check.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        images = np.lib.format.open_memmap(
            self.images_path, mode="w+", dtype=np.uint8, shape=self.shape(len(identifiers))
        )
        del images
        mask = np.zeros((len(identifiers), self.slots), dtype=np.float32)
        self.index_path.write_text(json.dumps(identifiers), encoding="utf-8")

        arguments = [
            DecodeRequest(
                position=position,
                identifier=identifier,
                source=str(source),
                images_path=str(self.images_path),
                slices=self.slices,
                image_size=self.image_size,
                scheme=self.scheme,
                planes=planes,
                crop=self.crop,
            )
            for position, identifier in enumerate(identifiers)
        ]
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for position, flags in pool.map(StudyCache.decode_one, arguments, chunksize=8):
                mask[position] = flags
                done += 1
                if done % 200 == 0:
                    logger.info(f"  cached {done}/{len(identifiers)} studies")
        self.report(mask)
        # LAST, and only once the mask has been judged: `wait_for_cache` keys on this file existing, so
        # writing it is the signal that the cache is usable. A build that decoded nothing must not get
        # that far.
        np.save(self.mask_path, mask)

    def report(self, mask: Float[np.ndarray, "studies slots"]) -> None:
        """Log what the build actually produced, and refuse a cache that decoded nothing.

        The build used to log progress and then `CACHE BUILD COMPLETE` unconditionally. That message is
        printed whether every study decoded or none did — and `decode_one` reports a study ABSENT rather
        than raising, by design, so a systematic failure looks exactly like a corpus of empty studies.
        It nearly cost a 54 GB build: a nine-slot scheme emitted six-slot bags, every write raised a
        shape error, all 4,407 studies were caught and reported absent, and the build would have
        announced completion over an array of zeros with a mask saying nothing was acquired (knee-gv3).

        Coverage per slot is worth logging even when it is fine, because it is the number that prices
        an arm: the extended scheme was argued for on a slot census that had to be recovered by loading
        the mask separately, months after the builds that produced it.
        """
        filled = mask.sum(axis=0)
        studies = mask.shape[0]
        for position, slot in enumerate(self.scheme_slots):
            share = 100 * filled[position] / studies
            logger.info(f"  {slot:22s} {int(filled[position]):5d}/{studies} studies ({share:.1f}%)")
        empty = int((mask.sum(axis=1) == 0).sum())
        logger.info(f"  studies with no slot at all: {empty}/{studies}")
        if empty == studies:
            message = (
                f"every one of {studies} studies decoded to nothing — the cache is zeros. "
                "The mask has NOT been written, so nothing downstream will treat this build as usable."
            )
            raise ValueError(message)

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
    def decode_one(request: "DecodeRequest") -> tuple[int, list[float]]:
        """Decode one study into its slot of the memory map. Top-level so a process pool can pickle it.

        The reader is built at the CACHE's resolution, not at its own default. Taking the default worked
        only while the two happened to agree; at any other resolution the decoded bag simply would not fit
        the map, and the shape error was being caught and reported as an absent study.
        """
        images = np.load(request.images_path, mmap_mode="r+")
        try:
            pipeline = SlotPipeline(
                scheme=SlotScheme(SCHEMES[request.scheme]),
                reader=SlotReader(
                    crop_mm=request.crop.mm, image_size=request.image_size, centre=request.crop.located()
                ),
            )
            study = Study.from_directory(Path(request.source) / request.identifier, request.planes, pipeline)
            bag, mask = study.bag(request.slices)
            images[request.position] = bag.numpy()
            images.flush()
            return request.position, mask.tolist()
        except (OSError, ValueError, RuntimeError, IndexError) as failure:
            # A study that will not decode is reported ABSENT in every slot rather than black in all of
            # them: the mask can express "not acquired", and a black image the mask calls acquired is a
            # silent lie the head cannot detect. The reason is PRINTED rather than swallowed — an
            # unreported decode failure is indistinguishable from a black knee.
            logger.info(f"  decode failed for {request.identifier}: {type(failure).__name__}: {failure}")
            return request.position, [0.0] * len(SCHEMES[request.scheme])
