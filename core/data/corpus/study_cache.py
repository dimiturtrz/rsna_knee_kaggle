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
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from jaxtyping import Float, UInt8
from numpy.typing import NDArray

from core.data.corpus.slot_reader import IMAGE_SIZE, Reading, SlotReader
from core.data.corpus.slots import SCHEMES, Routing
from core.data.corpus.study import SlotPipeline, Study
from core.data.corpus.window_reader import FileReader, MappedReader, WindowReader
from core.denominator import Denominator
from core.paths import TRAIN_TABLE

logger = logging.getLogger(__name__)

WITNESS_SHARDS = "witness_shards"
"""Where a witness-sized decode of a reading lands, so it can never be written over the reading itself."""

FLUSH_EVERY = 200
"""Studies between progress flushes — how much a killed build has to decode a second time.

The record is a few hundred kilobytes against a decode of seconds per study, so the flush itself is
free; this is set by how much work is acceptable to lose, not by what the write costs.
"""


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

    def window(self, batch: NDArray[np.int64], *, start: int, width: int) -> NDArray[np.uint8]:
        """`width` slices of every slot, from `start`, for each position in `batch`."""
        return self.reader.read(batch, start=start, width=width)

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

    def under(self, scheme: str) -> "StudyCache":
        """This same cache read under another slot scheme, which names a DIFFERENT set of files.

        The one axis a merge's two sources and its destination differ on. Everything else that names a
        cache — size, slices, band, crop, truncation — is carried rather than respelled, which is what
        keeps a derived cache at the geometry the corpus was actually decoded at.
        """
        return replace(self, routing=replace(self.routing, scheme=scheme))

    def witnessing(self) -> "StudyCache":
        """This same reading, in the subdirectory a WITNESS-SIZED decode of it belongs in.

        A witness is eight named studies. Checking one costs a `load()` of the full corpus array — tens
        of gigabytes memory-mapped to read about ten megabytes of it — and, worse, it costs the array
        EXISTING: nine members on the roster lost their decode to disk reclamation and can therefore
        never be witnessed again, `res392_ws_s2026` (the best single member this project has fitted)
        among them. The pixels those members were scored on are gone; the eight studies each of them
        witnessed are still in the raw corpus and decode in seconds.

        THE SUBDIRECTORY IS THE WHOLE SAFETY ARGUMENT, and it is deliberately not a key suffix. `first`
        writes `_first{N}` into the stem for exactly this hazard, and its own docstring records what was
        at stake: a partial decode landing on the full cache's name would overwrite a 71.6 GB artifact
        that every member on disk was fitted on, differing only in its first axis. A suffix makes that
        collision unlikely — one `replace()` with the wrong field, one caller that builds a name by hand,
        and the shard lands on the corpus. A different DIRECTORY makes it impossible: `images_{key}.npy`
        here and `images_{key}.npy` there are different files whatever the key says, so no shard can ever
        be written over the decode it is standing in for.

        The reading is CARRIED, not respelled — size, slices, band, crop, scheme and truncation all come
        with it — because a shard decoded at any other geometry would witness a different function than
        the member was fitted as, and would do it while reporting a clean residual.
        """
        return replace(self, directory=self.directory / WITNESS_SHARDS)

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

    @property
    def progress_path(self) -> Path:
        """The partial mask and its per-study done flags — what makes an interrupted build resumable.

        A full decode is hours, and `open_memmap(mode="w+")` recreates the pixel file zeroed, so
        without this a kill costs the entire run however far it had got. On 2026-08-11 a build sharing
        the machine with a trainer was killed by the OS with neither process leaving an error; the cost
        was the whole decode. This file bounds that cost to the studies since the last flush.

        It is removed once `mask_path` is written, so its presence means "a build stopped here" and
        never "this cache is finished".
        """
        return self.directory / f"progress_{self.key}.npz"

    def enrolled(self, identifiers: list[str]) -> list[str]:
        """The studies this cache actually holds, given the split's full list.

        The truncation is a property of the CACHE because it is in the cache's name, and a caller that
        applied it itself could log one count and write another.
        """
        return identifiers[: self.first] if self.first else identifiers

    def shape(self, studies: int) -> tuple[int, int, int, int, int]:
        """The cache's dimensions for a given number of studies."""
        return (studies, self.slots, self.slices, self.image_size, self.image_size)

    def reader(self) -> SlotReader:
        """The reader that produces THIS cache's pixels — its geometry at its resolution.

        The two are one fact and were being assembled by each caller from `reading` and `image_size`
        separately, which is the mismatch `decode_one` was already bitten by: a reader built at its own
        default rather than at the cache's simply does not fit the map.
        """
        return self.reading.reader(self.image_size)

    def prepare(self) -> None:
        """Make the directory the three files land in."""
        self.directory.mkdir(parents=True, exist_ok=True)

    def record(self, identifiers: list[str]) -> None:
        """Write the study order the pixel and mask arrays are indexed by.

        The cache owns it because the order is what makes a position mean a study: `_resumed` compares
        this file study by study before believing a partial build, and a writer that spelled the JSON
        itself would be free to write an order the resume check then rejects.
        """
        self.index_path.write_text(json.dumps(identifiers), encoding="utf-8")

    def recorded(self) -> list[str]:
        """The study order a previous build wrote, WITHOUT memory-mapping the pixels.

        The symmetric read of `record`, and the cache's own answer to "which studies are in here" — a
        caller that spelled the JSON itself could read an order the resume check would reject. `load`
        goes through it too, so the two can never disagree; this exists apart from `load` because the
        order alone is a few kilobytes and the array beside it is tens of gigabytes.
        """
        return json.loads(self.index_path.read_text(encoding="utf-8"))

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

        A BUILD THAT WAS INTERRUPTED RESUMES. Only studies missing from the progress record are put in
        the pool, so the cost of a kill is the studies since the last flush rather than the whole
        decode — see `progress_path` for what that cost used to be.
        """
        identifiers = self.enrolled(identifiers)
        self.prepare()
        mask, done = self._resumed(identifiers) or self._started(identifiers)

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
            if not done[position]
        ]
        for decoded, (position, flags) in enumerate(self._decode_all(arguments, workers), start=1):
            mask[position] = flags
            done[position] = True
            if decoded % FLUSH_EVERY == 0:
                logger.info(f"  cached {int(done.sum())}/{len(identifiers)} studies")
                self._flush(mask, done)
        self.report(mask)
        # LAST, and only once the mask has been judged: `wait_for_cache` keys on this file existing, so
        # writing it is the signal that the cache is usable. A build that decoded nothing must not get
        # that far.
        np.save(self.mask_path, mask)
        self.progress_path.unlink(missing_ok=True)

    def _started(self, identifiers: list[str]) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        """A fresh build: the pixel file recreated zeroed, nothing decoded, the study order recorded."""
        images = np.lib.format.open_memmap(
            self.images_path, mode="w+", dtype=np.uint8, shape=self.shape(len(identifiers))
        )
        del images
        self.record(identifiers)
        return np.zeros((len(identifiers), self.slots), dtype=np.float32), np.zeros(len(identifiers), dtype=np.bool_)

    def _resumed(self, identifiers: list[str]) -> tuple[NDArray[np.float32], NDArray[np.bool_]] | None:
        """What an interrupted build of exactly this cache left behind, or None to start from zero.

        The study ORDER is compared study by study rather than trusted from the key, because the key
        fixes the array's shape and not its index: resuming onto a different order would write each
        study's pixels at another study's position, a corruption no shape check could ever see.

        A study that was mid-decode when the kill landed has partial pixels and a false flag, so it is
        simply decoded again over the top. Only a flushed study is believed.
        """
        if not (self.progress_path.exists() and self.images_path.exists() and self.index_path.exists()):
            return None
        if json.loads(self.index_path.read_text(encoding="utf-8")) != identifiers:
            return None
        saved = np.load(self.progress_path)
        mask, done = saved["mask"], saved["done"]
        if mask.shape != (len(identifiers), self.slots) or done.shape != (len(identifiers),):
            return None
        logger.info(f"  resuming: {int(done.sum())}/{len(identifiers)} studies already on disk")
        return mask, done

    def _flush(self, mask: NDArray[np.float32], done: NDArray[np.bool_]) -> None:
        """The progress record written through a scratch file, so a kill mid-write cannot corrupt it.

        A handle is passed rather than the path because `np.savez` appends `.npz` to a name that lacks
        it, which would put the scratch write somewhere other than the file being replaced.
        """
        scratch = self.progress_path.with_name(f"{self.progress_path.name}.part")
        with scratch.open("wb") as handle:
            np.savez(handle, mask=mask, done=done)
        scratch.replace(self.progress_path)

    @staticmethod
    def _decode_all(arguments: list[DecodeRequest], workers: int) -> Iterator[tuple[int, list[float]]]:
        """Every request decoded — through a pool, or INLINE when one worker was asked for.

        A pool of one is a process that is spawned, paid for, and then used serially, and spawning is
        exactly what fails under load: on 2026-08-20 fourteen tests died with exit status 3221225794
        (0xC0000142, STATUS_DLL_INIT_FAILED) — Windows refusing to start any new process while a
        training arm, a CPU witness and a gate session shared the machine — and the same tree passed
        serially minutes later. `workers=1` is what the unit tests pass, so this is also the difference
        between a suite that survives a busy machine and one that flakes on it.

        `decode_one` is a staticmethod over a frozen request that writes through a memory map, so the
        inline path computes the same thing by the same code; only the process boundary is gone.
        """
        if workers == 1:
            yield from (StudyCache.decode_one(request) for request in arguments)
            return
        with ProcessPoolExecutor(max_workers=workers) as pool:
            yield from pool.map(StudyCache.decode_one, arguments, chunksize=8)

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
        recorded = self.mask_path.is_file() and self.index_path.is_file()
        return recorded and (self.images_path.is_file() or self.wider() is not None)

    def wider(self) -> "StudyCache | None":
        """A decoded cache over the same studies whose slot axis BEGINS with this one's, if one is on disk.

        Schemes append slots and never insert them (`EXTENDED_SLOTS`), so a narrower cache's pixels are
        the leading slots of a wider one decoded at the same geometry — measured 2026-10-02 on the 336 px
        band pair, 4,407 of 4,407 studies byte-identical in pixels and mask. That made the 56 GB six-slot
        array a copy, deleted; its mask and index stay, and say which studies and slots it holds.
        """
        narrow = self.scheme_slots
        candidates = (self.under(scheme) for scheme in SCHEMES)
        return next(
            (
                cache
                for cache in candidates
                if cache.slots > self.slots
                and cache.scheme_slots[: self.slots] == narrow
                and cache.images_path.is_file()
                and cache.index_path.is_file()
                and cache.recorded() == self.recorded()
            ),
            None,
        )

    def pixels(self) -> UInt8[np.ndarray, "n s d h w"]:
        """The pixel cache memory-mapped read-only — its own file, else the leading slots of a wider one."""
        if self.images_path.is_file():
            return np.load(self.images_path, mmap_mode="r")
        wider = self.wider()
        if wider is None:
            raise FileNotFoundError(self.images_path)
        return np.load(wider.images_path, mmap_mode="r")[:, : self.slots]

    def load(self) -> tuple[CachedCorpus, list[str]]:
        """The cache, memory-mapped read-only, with its study order."""
        return CachedCorpus(self.pixels(), np.load(self.mask_path)), self.recorded()

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
    three lines in a `main` because seven callers now assemble them: the build CLI, the smoke that
    decodes a few dozen studies, the four audits (`crop_coverage`, `diagnose_laterality`,
    `diagnose_matrix`, `decode_bench`) and the evaluation (`oof_oracle`) that
    each used to spell out `train.csv`, `StudyInstanceUID` and
    `train_series` for themselves. Assembled seven times they would drift on the one thing that must not:
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
        return self.raw / (TRAIN_TABLE if self.split == "train_series" else "test.csv")

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

    def sample(self, size: int) -> "CorpusSample":
        """Every k-th study, so the draw is a property of the corpus ORDER rather than of a seed.

        Repeating a measurement then cannot quietly answer with different studies, and two tools asking
        for the same size are asking about the same studies rather than about two draws that match in
        count alone.
        """
        identifiers = self.identifiers()
        return CorpusSample(identifiers[:: max(1, len(identifiers) // size)][:size], len(identifiers))

    def named(self, listing: Path) -> "CorpusSample":
        """The studies written in a file, checked against the corpus rather than trusted.

        A name that is not in the corpus is an error and not a skip: silently dropping it would shrink the
        denominator of every statistic downstream without saying so, and a rate over the studies that
        happened to match is not a rate over the ones that were asked for.
        """
        identifiers = self.identifiers()
        wanted = [line.strip() for line in listing.read_text().splitlines() if line.strip()]
        missing = sorted(set(wanted) - set(identifiers))
        if missing:
            message = f"{len(missing)} of {len(wanted)} named studies are not in the corpus, first: {missing[0]}"
            raise ValueError(message)
        return CorpusSample(wanted, len(identifiers))


@dataclass(frozen=True)
class CorpusSample:
    """The studies a corpus tool drew, and the two denominators any share of them must be read against.

    It owns the DRAW as well as the sentence that states it, because they are one fact. Three audits
    (`band_coverage`, `crop_coverage`, `crop_alignment`) each sliced their own every-k-th sample and then
    wrote their own wording of "from N sampled of M in the corpus" — and a share over the series that
    happened to decode is not a share over the corpus, which this project has printed as the other three
    times in a single session. Owning only the sentence would leave the numbers inside it free to come
    from somewhere the sentence never saw.
    """

    chosen: list[str]
    corpus: int

    def line(self, measured: int, unit: str) -> str:
        """The denominator sentence a coverage report opens with, in ONE wording.

        `unit` because the three tools measure different things — series, studies, routed slots — and the
        count is the only part of the sentence that is theirs.
        """
        return f"{measured} {unit} measured, from {len(self.chosen)} studies sampled of {self.corpus} in the corpus"

    def denominator(self, measured: int, unit: str) -> Denominator:
        """The refusal owed when a tool decoded NOTHING.

        Every one of these audits drops a study that will not decode, which is right — an unreadable
        series is silent rather than evidence. But a wholesale decode failure then prints a header, no
        rows, and exits 0, which is the exact shape `Denominator` was written for.
        """
        return Denominator(measured, f"{unit} decoded from {len(self.chosen)} sampled studies")
