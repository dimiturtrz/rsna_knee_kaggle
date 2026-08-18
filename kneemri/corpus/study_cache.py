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
from functools import cached_property
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from jaxtyping import Float
from numpy.typing import NDArray

from kneemri.corpus.slot_reader import IMAGE_SIZE, Reading
from kneemri.corpus.slots import SCHEMES, Routing
from kneemri.corpus.study import SlotPipeline, Study
from kneemri.corpus.window_reader import FileReader, MappedReader, WindowReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CachedCorpus:
    """The decoded corpus as the trainer sees it: pixels and the presence mask, indexed alike.

    The two always travel together — a mask without its pixels says a slot was acquired and cannot show
    it — so they are one argument rather than two.
    """

    images: NDArray[np.uint8]
    mask: NDArray[np.float32]

    @cached_property
    def reader(self) -> WindowReader:
        """Explicit file reads where the pixels are a memory map, plain indexing where they are not.

        A corpus assembled in memory — a test fixture, a held slab already sliced out — has no file to
        read, and the map's own path is the only one that pays for the machinery.
        """
        images = self.images
        if isinstance(images, np.memmap) and images.filename:
            return FileReader(images, Path(images.filename))
        return MappedReader(images)

    def window(self, batch: NDArray[np.int64], start: int, width: int) -> NDArray[np.uint8]:
        """`width` slices of every slot, from `start`, for each position in `batch`."""
        return self.reader.read(batch, start, width)

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
    planes: dict[str, str]
    routing: Routing = Routing()
    reading: Reading = Reading()


NAMED_BY = {
    "slices": "slices",
    "image_size": "image_size",
    "routing": "routing",
    "reading": "reading",
    "first": "cache_studies",
}
"""Each cache field, against the name its namer states it under. Data rather than five keyword lines,
so the axis list is one object a test can hold against `StudyCache`'s own fields."""


class CacheNaming(Protocol):
    """Whatever knows which arrays it reads — a run's options, or a member's recorded config.

    Structural, so this module names neither: `training` and `inference` both import `corpus`, and a
    concrete type here would close two cycles at once.

    The two sides had each unified this constructor for themselves, and their docstrings tell the same
    story from opposite ends — `main` and `arm_check` spelled it out on the train side, the oracle, the
    witness check and the curve on the serve side. Five call sites, two homes, one axis list. The axis
    that goes into one home and not the other is not a crash: it is a run whose weights were fitted on
    pixels its own serve path resolves away from, both sides agreeing they read `the cache` and
    resolving to different files. The cache key exists to make that impossible from the build side; this
    protocol is the same guarantee from the reading side.
    """

    @property
    def slices(self) -> int: ...

    @property
    def image_size(self) -> int: ...

    @property
    def routing(self) -> Routing: ...

    @property
    def reading(self) -> Reading: ...

    @property
    def cache_studies(self) -> int:
        """How many studies of the split — `first` here, because a cache states it and a reader asks it."""
        ...


@dataclass(frozen=True)
class StudyCache:
    """A memory-mapped `[studies, slots, slices, size, size]` uint8 array plus its presence mask.

    A dataclass rather than a hand-written `__init__`, because every field here is a value that NAMES
    the file — six of them once truncation joined the shape, the reading and the router, which is one
    past the arg-count gate. The generated constructor is the same call the five call sites already
    make by keyword, and it drops two `x or X()` defaults that only existed to let a caller pass None.
    """

    directory: Path
    slices: int = 12
    image_size: int = IMAGE_SIZE
    routing: Routing = Routing()
    reading: Reading = Reading()
    first: int = 0
    """How many studies of the split this cache holds; 0 is all of them."""

    @classmethod
    def of(cls, naming: CacheNaming, directory: Path) -> "StudyCache":
        """The cache a run wrote or a member was fitted on — the axis list, spelled ONCE for both.

        A new axis is added to `NAMED_BY` and to `CacheNaming`, and every caller either supplies it or
        fails to type-check. Written as two agreeing constructors — which is how it arrived, one on the
        run's options and one on the member's config — the same addition is a silent divergence between
        the pixels a run trained on and the pixels its own members resolve to at serve time.
        """
        return cls(directory, **{field: getattr(naming, stated) for field, stated in NAMED_BY.items()})

    @property
    def slots(self) -> int:
        """How wide the slot axis is — derived from the scheme, so the two cannot disagree."""
        return len(SCHEMES[self.routing.scheme])

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

        THE SLICE BAND IS THE THIRD OF THESE and was the last one missing. It decides which slices exist
        to be sampled, so a wider band at the same size, slice count and scheme is a different array
        under the same shape and the same name — and it is the axis an unlabeled-corpus rebuild wants
        first. The historic band names nothing, exactly like the historic crop.

        TRUNCATION IS THE FOURTH, and it is the one that makes a new reading axis smokeable at all. A
        full decode is two hours, so an arm on a reading nothing has built could only be smoked by
        paying for the whole corpus first — and a partial decode written under the FULL cache's name
        would overwrite a 71.6 GB artifact every member on disk was fitted on, with a shape that
        differs only in its first axis. `_first{N}` says so in the filename, so the two cannot collide
        and a trainer pointed at a partial cache is reading something that admits what it is.
        """
        truncation = f"_first{self.first}" if self.first else ""
        return f"{self.image_size}px_{self.slices}sl{self.reading.suffix()}{self.routing.suffix()}{truncation}"

    @property
    def scheme_slots(self) -> list[str]:
        """The slot names this cache is built over, in order — the width of its second axis."""
        return [slot.name for slot in SCHEMES[self.routing.scheme]]

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

    def enrolled(self, identifiers: list[str]) -> list[str]:
        """The studies this cache actually holds, given the split's full list.

        The truncation is a property of the CACHE because it is in the cache's name, and a caller that
        applied it itself could log one count and write another.
        """
        return identifiers[: self.first] if self.first else identifiers

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

        THE TRUNCATION IS APPLIED HERE, not by the caller, for the same reason: `first` is in the key,
        so a caller that truncated its own list would be free to write 60 studies under the name that
        promises 4407 — and the name is the only thing a later trainer has to go on.
        """
        identifiers = self.enrolled(identifiers)
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
                routing=self.routing,
                planes=planes,
                reading=self.reading,
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

    def decoded(self) -> bool:
        """Whether this cache is on disk and usable — all three files, not just the pixels.

        The mask is written LAST, after the build has judged it, so a cache missing only its mask is a
        decode that died or produced nothing. Asking for the three together is what `arm_check` means by
        UNCACHED and what the smoke means by "decode this first", and the two spelled it separately.
        """
        return self.images_path.is_file() and self.mask_path.is_file() and self.index_path.is_file()

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
                scheme=request.routing.built(),
                reader=request.reading.reader(request.image_size),
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
            return request.position, [0.0] * len(SCHEMES[request.routing.scheme])


@dataclass(frozen=True)
class CorpusSplit:
    """The shipped tables a decode reads before it can build: which studies, and each series' plane.

    Beside `StudyCache` because it is exactly the arguments `build` takes, and a dataclass rather than
    three lines in a `main` because eight callers now assemble them: the build CLI, the smoke that
    decodes a few dozen studies, the four audits (`crop_coverage`, `diagnose_laterality`,
    `diagnose_matrix`, `decode_bench`) and the two evaluations (`oof_oracle`, `stack_coverage`) that
    each used to spell out `train.csv`, `StudyInstanceUID` and
    `train_series` for themselves. Assembled eight times they would drift on the one thing that must not:
    which tables name the split — and an audit reading a different study list than the cache was built
    from is a measurement about a corpus nobody trained on.
    """

    raw: Path
    split: str = "train_series"

    @property
    def source(self) -> Path:
        """The directory of DICOMs the decode walks."""
        return self.raw / self.split

    @property
    def studies_table(self) -> Path:
        """The per-STUDY table: one row per study, and where the labels join."""
        return self.raw / ("train.csv" if self.split == "train_series" else "test.csv")

    @property
    def series_table(self) -> Path:
        """The per-SERIES table: one row per series, carrying the plane a slot routes on.

        Named apart from `studies_table` because reading the wrong one of the two is silent: both carry
        `StudyInstanceUID`, so a study list taken from the series table is the same identifiers with
        multiplicity, and anything that counts or indexes by position then quietly means something else.
        """
        return self.raw / f"{self.split}.csv"

    def identifiers(self) -> list[str]:
        """Every study in the split, in the order the cache will index them by."""
        return pd.read_csv(self.studies_table)["StudyInstanceUID"].tolist()

    def planes(self) -> dict[str, str]:
        """Series UID to anatomical plane, which is what the slot predicate routes on."""
        return StudyCache.planes_of(self.series_table)
