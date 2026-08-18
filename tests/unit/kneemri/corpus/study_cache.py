import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from kneemri.corpus.crop_centre import Crop
from kneemri.corpus.slot_reader import Band, Reading
from kneemri.corpus.slots import EXTENDED_SLOTS, SLOTS, Routing
from kneemri.corpus.study_cache import NAMED_BY, CachedCorpus, CorpusSplit, DecodeRequest, StudyCache
from kneemri.corpus.window_reader import FileReader, MappedReader

from ..conftest import write_series


def test_shape(tmp_path: Path):
    """The cache is [studies, slots, slices, size, size] — the slot axis is the head's axis."""
    assert StudyCache(tmp_path, slices=4, image_size=16).shape(7) == (7, len(SLOTS), 4, 16, 16)


def test_images_path(tmp_path: Path):
    """The file name states everything that decides the contents, so two configs cannot collide."""
    path = StudyCache(tmp_path, slices=4, image_size=16).images_path
    assert path.name == "images_16px_4sl.npy"


def test_mask_path(tmp_path: Path):
    assert StudyCache(tmp_path, slices=4, image_size=16).mask_path.name == "mask_16px_4sl.npy"


def test_index_path(tmp_path: Path):
    assert StudyCache(tmp_path, slices=4, image_size=16).index_path.name == "index_16px_4sl.json"


def test_planes_of(tmp_path: Path):
    """The plane is the one thing the slot predicate takes from the shipped series table."""
    table = tmp_path / "series.csv"
    pd.DataFrame({"SeriesInstanceUID": ["s1", "s2"], "Anatomical_Plane": ["Sagittal", "Axial"]}).to_csv(
        table, index=False
    )
    assert StudyCache.planes_of(table) == {"s1": "Sagittal", "s2": "Axial"}


def _built(tmp_path: Path):
    """A one-study cache, built and loaded — the fixture the corpus accessors are read from."""
    source = tmp_path / "src"
    write_series(source / "study0" / "sag", slices=4, description="SAG PD FS", size=32)
    cache = StudyCache(tmp_path / "cache", slices=2, image_size=8)
    cache.build(source, ["study0"], {"sag": "Sagittal"}, workers=1)
    corpus, _ = cache.load()
    return corpus


def test_build(tmp_path: Path):
    """Decoding writes each study into its own slice of the map, and records what was filled."""
    source = tmp_path / "src"
    write_series(source / "study0" / "sag", slices=4, description="SAG PD FS", size=32)
    cache = StudyCache(tmp_path / "cache", slices=2, image_size=8)
    cache.build(source, ["study0"], {"sag": "Sagittal"}, workers=1)
    corpus, identifiers = cache.load()
    assert corpus.images.shape == (1, len(SLOTS), 2, 8, 8)
    assert identifiers == ["study0"]
    assert corpus.mask[0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert int(np.asarray(corpus.images[0, 0]).max()) > 0, "the filled slot carries pixels"


def test_enrolled(tmp_path: Path):
    """The cache decides which studies it holds, because the count is in its own name."""
    identifiers = ["study0", "study1", "study2"]

    assert StudyCache(tmp_path).enrolled(identifiers) == identifiers, "0 enrols the whole split"
    assert StudyCache(tmp_path, first=2).enrolled(identifiers) == ["study0", "study1"]
    assert StudyCache(tmp_path, first=9).enrolled(identifiers) == identifiers, "asking past the end is the whole split"


def test_build_truncated(tmp_path: Path):
    """A partial decode holds N studies AND says so in its name, so it cannot overwrite the full one.

    THE DEFECT THIS PINS: a full build is two hours, so a reading nothing has decoded could not be
    smoked before it was queued — and a truncated decode written under the full cache's name would
    destroy 71.6 GB of arrays every member on disk was fitted on, differing only in one axis length.
    """
    source = tmp_path / "src"
    for study in ("study0", "study1", "study2"):
        write_series(source / study / "sag", slices=4, description="SAG PD FS", size=32)
    cache = StudyCache(tmp_path / "cache", slices=2, image_size=8, first=2)

    cache.build(source, ["study0", "study1", "study2"], {"sag": "Sagittal"}, workers=1)

    corpus, identifiers = cache.load()
    assert identifiers == ["study0", "study1"], "the cache enrols the first N, and its index says which"
    assert corpus.images.shape[0] == 2
    assert cache.key.endswith("_first2"), "the truncation is in the name, not only in the contents"
    assert cache.images_path != StudyCache(tmp_path / "cache", slices=2, image_size=8).images_path


def test_decoded(tmp_path: Path):
    """A cache is usable only with all THREE files, and the mask is the one a dead build never writes."""
    cache = StudyCache(tmp_path, slices=2, image_size=8)
    assert not cache.decoded(), "nothing on disk"

    cache.images_path.write_text("pixels", encoding="utf-8")
    cache.index_path.write_text("[]", encoding="utf-8")
    assert not cache.decoded(), "the mask is written LAST, so pixels without it are a build that died"

    cache.mask_path.write_text("mask", encoding="utf-8")
    assert cache.decoded()


def test_source(tmp_path: Path):
    """The split names its own DICOM directory, so no caller spells it beside the table it reads."""
    assert CorpusSplit(tmp_path).source == tmp_path / "train_series"
    assert CorpusSplit(tmp_path, "test_series").source == tmp_path / "test_series"


def test_studies_table(tmp_path: Path):
    """The per-study table, whose name does NOT follow the split's — which is the trap it exists to close."""
    assert CorpusSplit(tmp_path).studies_table == tmp_path / "train.csv"
    assert CorpusSplit(tmp_path, "test_series").studies_table == tmp_path / "test.csv"


def test_series_table(tmp_path: Path):
    """The per-series table, kept apart from the study one because reading the wrong of the two is silent.

    Both carry `StudyInstanceUID`, so a study list taken from the series table is the same identifiers
    with multiplicity — no error, and every count or positional index downstream then means something else.
    """
    assert CorpusSplit(tmp_path).series_table == tmp_path / "train_series.csv"
    assert CorpusSplit(tmp_path, "test_series").series_table == tmp_path / "test_series.csv"
    assert CorpusSplit(tmp_path).series_table != CorpusSplit(tmp_path).studies_table


def test_identifiers(tmp_path: Path):
    """Which studies the split holds — and the TABLE differs from the split's own name, which is the trap."""
    pd.DataFrame({"StudyInstanceUID": ["s0", "s1"]}).to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["t0"]}).to_csv(tmp_path / "test.csv", index=False)

    assert CorpusSplit(tmp_path).identifiers() == ["s0", "s1"]
    assert CorpusSplit(tmp_path, "test_series").identifiers() == ["t0"], "any other split reads test.csv"


def test_planes(tmp_path: Path):
    """The one thing the slot predicate takes from the shipped series table, read off the split."""
    pd.DataFrame({"SeriesInstanceUID": ["e0"], "Anatomical_Plane": ["SAG"]}).to_csv(
        tmp_path / "train_series.csv", index=False
    )

    assert CorpusSplit(tmp_path).planes() == {"e0": "SAG"}


def test_load(tmp_path: Path):
    """Loading maps the cache read-only rather than reading 35 GB into memory."""
    source = tmp_path / "src"
    write_series(source / "study0" / "sag", slices=4, size=32)
    cache = StudyCache(tmp_path / "cache", slices=2, image_size=8)
    cache.build(source, ["study0"], {"sag": "Sagittal"}, workers=1)
    corpus, _ = cache.load()
    assert isinstance(corpus.images, np.memmap)


def test_image_size(tmp_path: Path):
    """The corpus reports its own resolution, so no consumer has to assume the default."""
    corpus = _built(tmp_path)
    assert corpus.image_size == 8


def test_slices(tmp_path: Path):
    """And its slice depth, which bounds the windows any member can be read over."""
    assert _built(tmp_path).slices == 2


def test_decode_one(tmp_path: Path):
    """One study decoded into its own row of the map, returning the mask of what was filled."""
    source = tmp_path / "src"
    write_series(source / "study0" / "sag", slices=4, description="SAG PD FS", size=32)
    cache = StudyCache(tmp_path / "cache", slices=2, image_size=8)
    cache.directory.mkdir(parents=True, exist_ok=True)
    np.lib.format.open_memmap(cache.images_path, mode="w+", dtype=np.uint8, shape=cache.shape(1))
    position, flags = StudyCache.decode_one(
        DecodeRequest(
            position=0,
            identifier="study0",
            source=str(source),
            images_path=str(cache.images_path),
            slices=2,
            image_size=8,
            planes={"sag": "Sagittal"},
        )
    )
    assert position == 0
    assert flags == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_decode_one_reports_a_study_it_cannot_decode_as_absent(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    """A study that will not decode comes back ABSENT in every slot, named in the log, and does not raise.

    Absent rather than black because the mask can express "not acquired" and cannot express "this row is
    zeros for a reason" — a black study the mask calls acquired is a lie the head has no way to detect.
    Not raising because one bad study inside a process pool would otherwise kill a 4,400-study build; the
    guard against that becoming silent is `report`, which refuses a mask that is empty everywhere.
    """
    cache = StudyCache(tmp_path / "cache", slices=2, image_size=8)
    cache.directory.mkdir(parents=True, exist_ok=True)
    np.lib.format.open_memmap(cache.images_path, mode="w+", dtype=np.uint8, shape=cache.shape(1))

    with caplog.at_level("INFO"):
        position, flags = StudyCache.decode_one(
            DecodeRequest(
                position=0,
                identifier="missing",
                source=str(tmp_path / "src"),
                images_path=str(cache.images_path),
                slices=2,
                image_size=8,
                planes={},
            )
        )

    assert position == 0, "the row still has to be reported, or the pool's map loses a study"
    assert flags == [0.0] * 6, "every slot absent — none of them black-but-acquired"
    assert "decode failed for missing" in caplog.text, "and the reason is printed, not swallowed"


@dataclass(frozen=True)
class Naming:
    """A `CacheNaming` with every axis set to something other than its default, so a dropped one shows."""

    slices: int = 8
    image_size: int = 224
    routing: Routing = Routing(scheme="extended")
    reading: Reading = Reading(crop=Crop(mm=90.0))
    cache_studies: int = 40


def test_of(tmp_path: Path):
    """Every axis a namer states reaches the cache, and the axis LIST is this constructor's alone.

    The run's options and the member's recorded config each built this by hand, agreeing by inspection
    across two packages. The field-set assertion is what makes a seventh axis a failure here rather than
    a silent divergence between the pixels a run trained on and the ones its members later resolve to.
    """
    naming = Naming()

    cache = StudyCache.of(naming, tmp_path)

    assert cache == StudyCache(
        tmp_path,
        slices=naming.slices,
        image_size=naming.image_size,
        routing=naming.routing,
        reading=naming.reading,
        first=naming.cache_studies,
    )
    assert set(vars(cache)) - {"directory"} == set(NAMED_BY), (
        "a new cache axis: state it on `CacheNaming` and take it in `NAMED_BY`, or both sides drift again"
    )


def test_key(tmp_path: Path):
    """The canonical scheme keeps its historic name; any other slot count is named explicitly.

    The slot count decides the array's SHAPE, so leaving it out of the filename meant a nine-slot
    build would overwrite the six-slot cache in place. The canonical name is preserved so arrays
    already on disk stay loadable — a rename cannot be done while a trainer holds one mapped.
    """
    canonical = StudyCache(tmp_path, slices=12, image_size=336)
    assert canonical.key == "336px_12sl"
    assert canonical.images_path.name == "images_336px_12sl.npy"

    extended = StudyCache(tmp_path, slices=12, image_size=336, routing=Routing(scheme="extended"))
    assert extended.key == "336px_12sl_extended"
    assert extended.images_path != canonical.images_path, "a different slot count must not collide"
    assert extended.mask_path != canonical.mask_path
    assert extended.index_path != canonical.index_path


def test_shape_follows_the_slot_count(tmp_path: Path):
    """The shape reports the cache's OWN slot count, not the module-level scheme's."""
    assert StudyCache(tmp_path, slices=12, image_size=336, routing=Routing(scheme="extended")).shape(4407)[1] == 9


def test_slots(tmp_path: Path):
    """The slot axis is derived from the scheme, so a cache cannot disagree with its own decoder.

    This replaced a ValueError that checked exactly that mismatch. Deriving the width made the check
    unreachable, and an invariant held by construction beats one held by a guard.
    """
    assert StudyCache(tmp_path).slots == len(SLOTS)
    assert StudyCache(tmp_path, routing=Routing(scheme="extended")).slots == len(SLOTS) + 3


def test_scheme_slots():
    """The slot names a cache is built over, in order — the width of its own second axis."""
    cache = StudyCache(Path("nowhere"))
    assert cache.scheme_slots == [slot.name for slot in SLOTS]
    assert len(StudyCache(Path("nowhere"), routing=Routing(scheme="extended")).scheme_slots) == len(EXTENDED_SLOTS)


def test_report(caplog: pytest.LogCaptureFixture):
    """Coverage is logged per slot, and a build that decoded NOTHING is refused rather than announced.

    `decode_one` reports a study absent rather than raising, by design, so a systematic failure looks
    exactly like a corpus of empty studies — and the build used to print CACHE BUILD COMPLETE either
    way. A nine-slot scheme emitting six-slot bags would have finished over 54 GB of zeros (knee-gv3).
    """
    cache = StudyCache(Path("nowhere"))
    healthy = np.zeros((4, len(SLOTS)), dtype=np.float32)
    healthy[:, 0] = 1.0
    healthy[:2, 1] = 1.0
    with caplog.at_level(logging.INFO):
        cache.report(healthy)
    assert "4/4" in caplog.text and "studies with no slot at all: 0/4" in caplog.text

    with pytest.raises(ValueError, match="decoded to nothing"):
        cache.report(np.zeros((4, len(SLOTS)), dtype=np.float32))


def test_key_separates_a_different_crop(tmp_path: Path):
    """An 80mm cache must NOT write over the 130mm arrays every member was fitted on.

    The key carried resolution, depth and scheme but not the physical crop, so a tighter-crop cache at
    the same size and depth resolved to `images_336px_12sl.npy` — the path already holding 99 GB of
    decoded arrays, with no shape change for anything to notice. The default keeps the historic name
    for the same reason the canonical scheme does: those files are on disk and memory-mapped.
    """
    default = StudyCache(tmp_path, slices=12, image_size=336)
    tighter = StudyCache(tmp_path, slices=12, image_size=336, reading=Reading(crop=Crop(mm=80.0)))

    assert default.key == "336px_12sl", "the historic name is unchanged"
    assert tighter.key != default.key
    assert "80mm" in tighter.key
    assert tighter.images_path != default.images_path
    assert tighter.mask_path != default.mask_path


def test_key_separates_a_different_centre(tmp_path: Path):
    """A tissue-centred cache is different pixels, so it must not share the image-centred file."""
    default = StudyCache(tmp_path, slices=12, image_size=336)
    located = StudyCache(tmp_path, slices=12, image_size=336, reading=Reading(crop=Crop(centre="tissue")))

    assert default.key == "336px_12sl", "the historic name means the historic centre"
    assert "tissue" in located.key
    assert located.images_path != default.images_path

    both = StudyCache(tmp_path, slices=12, image_size=336, reading=Reading(crop=Crop(mm=80.0, centre="tissue")))
    assert both.key == "336px_12sl_80mm_tissue", "the two axes compose rather than shadowing each other"


def test_decode_one_reads_the_band_the_request_states(tmp_path: Path):
    """The band on the request must reach the READER, not stop at the cache's name.

    The failure this repository has hit nine times is a value that travels as far as a config and is
    then answered by a consumer's own default — and until 2026-08-15 the band was exactly that: recorded
    on every member, keyed by nothing, and read by a `SlotReader` taking its module constant. A key that
    separates two bands while the decode ignores them is worse than no key at all, because the two files
    would then hold identical pixels under different names.

    Four acquired slices sampled two deep: the historic band takes the shallowest and the middle, and a
    deep band takes two others. Different slices are different bytes, which is the only proof that does
    not trust the plumbing it is checking.
    """
    source = tmp_path / "src"
    write_series(source / "study0" / "sag", slices=4, description="SAG PD FS", size=32)

    def decoded(name: str, band: Band) -> NDArray[np.uint8]:
        cache = StudyCache(tmp_path / name, slices=2, image_size=8, reading=Reading(band=band))
        cache.directory.mkdir(parents=True, exist_ok=True)
        np.lib.format.open_memmap(cache.images_path, mode="w+", dtype=np.uint8, shape=cache.shape(1))
        StudyCache.decode_one(
            DecodeRequest(
                position=0,
                identifier="study0",
                source=str(source),
                images_path=str(cache.images_path),
                slices=2,
                image_size=8,
                planes={"sag": "Sagittal"},
                reading=Reading(band=band),
            )
        )
        return np.load(cache.images_path)

    historic = decoded("historic", Band())
    deep = decoded("deep", Band(0.6, 1.0))

    assert not np.array_equal(historic, deep), "a stated band that changes no pixel never reached the reader"


def test_key_separates_a_different_band(tmp_path: Path):
    """A wider band samples different SLICES, so it must not write over the historic arrays.

    The third axis the key was missing, and the one an unlabeled-corpus rebuild wants first: the shape
    is identical, so nothing downstream would have noticed a 0.10-0.90 build landing on the file every
    member was fitted on. The default names nothing, exactly like the historic crop and scheme.
    """
    default = StudyCache(tmp_path, slices=12, image_size=336)
    wider = StudyCache(tmp_path, slices=12, image_size=336, reading=Reading(band=Band(0.1, 0.9)))

    assert default.key == "336px_12sl", "the historic name means the historic band"
    assert wider.key == "336px_12sl_0.1-0.9band"
    assert wider.images_path != default.images_path
    assert wider.mask_path != default.mask_path

    both = StudyCache(tmp_path, slices=12, image_size=336, reading=Reading(crop=Crop(mm=80.0), band=Band(0.1, 0.9)))
    assert both.key == "336px_12sl_80mm_0.1-0.9band", "the axes compose rather than shadowing each other"


def test_key_separates_a_different_tie(tmp_path: Path):
    """A last-maximal cache holds different SERIES in its slots, so it needs its own file."""
    default = StudyCache(tmp_path, slices=12, image_size=336)
    theirs = StudyCache(tmp_path, slices=12, image_size=336, routing=Routing(tie="last"))

    assert default.key == "336px_12sl"
    assert theirs.key == "336px_12sl_last"
    assert theirs.images_path != default.images_path


def test_reader(tmp_path: Path):
    """A mapped corpus reads its file outright; one assembled in memory has no file, and indexes."""
    written = np.zeros((2, 2, 2, 2, 2), dtype=np.uint8)
    np.save(tmp_path / "images.npy", written)
    mask = np.ones((2, 2), dtype=np.float32)

    assert isinstance(CachedCorpus(np.load(tmp_path / "images.npy", mmap_mode="r"), mask).reader, FileReader)
    assert isinstance(CachedCorpus(written, mask).reader, MappedReader)


def test_window(tmp_path: Path):
    """Either way round, the window is the slices asked for, for the positions asked for."""
    written = np.random.default_rng(1).integers(0, 255, size=(4, 2, 6, 3, 3), dtype=np.uint8)
    path = tmp_path / "images.npy"
    np.save(path, written)
    mask = np.ones((4, 2), dtype=np.float32)

    mapped = CachedCorpus(np.load(path, mmap_mode="r"), mask)
    resident = CachedCorpus(written, mask)
    batch = np.array([3, 0, 2])
    assert np.array_equal(mapped.window(batch, 1, 2), written[batch, :, 1:3])
    assert np.array_equal(resident.window(batch, 1, 2), written[batch, :, 1:3])
