"""The shared cache flags, on a parser built here so the argv a CLI accepts is the thing under test."""

import argparse
from pathlib import Path

from kneemri.corpus.cache_flags import CacheFlags
from kneemri.corpus.crop_centre import Crop
from kneemri.corpus.slot_reader import Band, Reading
from kneemri.corpus.slots import Routing


def test_add_to():
    """Every axis that selects a cache file must arrive from ONE flag set, with the historic defaults.

    THE DEFECT THIS PINS: four entrypoints each spelled these out, and two of them had no crop or band
    flag at all — so they loaded the default arrays whatever was on disk. The two that did have them had
    already diverged on how to spell the band, which is the state just before a shell script silently
    reads the wrong pixels.
    """
    parser = argparse.ArgumentParser()

    CacheFlags.add_to(parser)
    defaults = parser.parse_args([])

    assert CacheFlags.cache(defaults, Path(".")).key == "336px_12sl", "the defaults name the historic cache"
    every = parser.parse_args(
        [
            *("--slices", "6", "--image-size", "448", "--crop-mm", "80", "--centre", "tissue"),
            *("--band-low", "0.1", "--band-high", "0.9", "--scheme", "extended", "--tie", "last"),
        ]
    )
    assert CacheFlags.cache(every, Path(".")).key == "448px_6sl_80mm_tissue_0.1-0.9band_extended_last"


def test_cache(tmp_path: Path):
    """The flags name a cache with no axis left for the caller to remember."""
    parser = argparse.ArgumentParser()
    CacheFlags.add_to(parser)
    tighter = parser.parse_args(["--crop-mm", "80", "--band-low", "0.1", "--band-high", "0.9", "--tie", "last"])

    cache = CacheFlags.cache(tighter, tmp_path)

    assert cache.directory == tmp_path
    assert cache.reading == Reading(crop=Crop(mm=80.0), band=Band(low=0.1, high=0.9))
    assert cache.routing == Routing(tie="last")
    assert cache.images_path != CacheFlags.cache(parser.parse_args([]), tmp_path).images_path
