"""The command-line flags that name a cache — one home, shared by every CLI that reads one.

Its own module because `study_cache` holds one subject, and this is a second: the CACHE is a file on
disk, and these are the flags a shell uses to ask for it. They are separated in the code for the same
reason they are joined in the repo — an axis added to the file's name has exactly one place to be
added to the command line, and no CLI can be left behind spelling out three of six fields.
"""

import argparse
from pathlib import Path

from kneemri.corpus.crop_centre import CENTRES, CROP_MM, DEFAULT_CENTRE, Crop
from kneemri.corpus.slot_reader import IMAGE_SIZE, SLICE_BAND, Band, Reading
from kneemri.corpus.slots import CANONICAL_SCHEME, FIRST_MAXIMAL, LAST_MAXIMAL, SCHEMES, Routing
from kneemri.corpus.study_cache import StudyCache


class CacheFlags:
    """The command-line flags that NAME a cache, in one place, plus the cache they name.

    Six values decide which arrays a CLI reads — size, depth, scheme, tie, crop and band — and four
    entrypoints spelled them out separately. They diverged the moment there were two: the trainer took
    `--band-low/--band-high`, the builder took `--band LOW HIGH`, and the two analysis CLIs could not
    name a non-default reading AT ALL, so they silently loaded the default arrays whatever was on disk.
    That is the serve-side mismatch `Member.cache` closed for member-driven loads, arriving through
    argparse instead.

    The dest names are the fields of `TrainingOptions`, because that dataclass is built straight from
    the namespace; a flag added here therefore lands on the run's recorded config rather than needing to
    be remembered twice.
    """

    @staticmethod
    def add_to(parser: argparse.ArgumentParser) -> None:
        """Attach every flag that selects a cache file."""
        parser.add_argument("--slices", type=int, default=12, help="slices per slot the cache holds")
        parser.add_argument(
            "--image-size",
            type=int,
            default=IMAGE_SIZE,
            help="in-plane resolution. Bytes grow with the SQUARE of this, and the cache path carries "
            "it, so a new resolution never disturbs an existing cache or the members fitted on it.",
        )
        parser.add_argument(
            "--crop-mm",
            type=float,
            default=CROP_MM,
            help="physical extent cropped about the centre. The default keeps the historic cache NAME, "
            "so any other value reads its own file rather than the arrays every member was fitted on.",
        )
        parser.add_argument(
            "--centre",
            default=DEFAULT_CENTRE,
            choices=sorted(CENTRES),
            help="where the physical crop is taken about. 'image' is the historic behaviour and keeps "
            "the historic cache name; 'tissue' locates the knee, which a tight crop needs (knee-yrf).",
        )
        parser.add_argument(
            "--band-low",
            type=float,
            default=SLICE_BAND[0],
            help="the shallowest fraction of a series' depth the slices are drawn from. The default is "
            "the historic reading and keeps the historic cache name.",
        )
        parser.add_argument(
            "--band-high",
            type=float,
            default=SLICE_BAND[1],
            help="the deepest such fraction. The default pair is what every member on disk was fitted "
            "under; a wider band is a different array under the same shape, so it is named for itself.",
        )
        parser.add_argument(
            "--scheme",
            choices=tuple(SCHEMES),
            default=CANONICAL_SCHEME,
            help="which slot scheme to route with. 'extended' adds COR_FLUID_NOFS, AX_T1 and "
            "AX_FLUID_NOFS, the three combinations the canonical six cannot express - 96%% of the "
            "series that currently match no slot at all. It reads a SEPARATE cache.",
        )
        parser.add_argument(
            "--tie",
            default=FIRST_MAXIMAL,
            choices=(FIRST_MAXIMAL, LAST_MAXIMAL),
            help="which of several equally dense candidates fills a slot. 'first' is ours and keeps the "
            "historic cache name; the published members reproduce their own predictions better under "
            "'last' (knee-evn), so a graft-matching cache is built with it.",
        )
        parser.add_argument(
            "--cache-studies",
            type=int,
            default=0,
            help="decode only the first N studies of the split (0 = all of them). The SMOKE knob for a "
            "reading that has never been decoded: a full build is two hours, so a new axis could not be "
            "smoked before it was queued. It is in the cache NAME (`_first{N}`), so a partial decode "
            "can never overwrite the full arrays. Distinct from --studies, which subsamples a cache "
            "that already exists.",
        )

    @staticmethod
    def cache(arguments: argparse.Namespace, directory: Path) -> StudyCache:
        """The cache those flags name, with no axis left for the caller to remember.

        The namespace-shaped half of one rule; `TrainingOptions.cache` is the dataclass-shaped half, and
        `Member.cache` the serve-side one. The two argv halves are pinned equal on real argv by
        `tests/integration/argv_to_cache_key.py`, since a namespace and a frozen dataclass cannot share
        an accessor without a structural type this repo's fitness gate reads as a second subject.
        """
        return StudyCache(
            directory,
            slices=arguments.slices,
            image_size=arguments.image_size,
            routing=Routing(scheme=arguments.scheme, tie=arguments.tie),
            reading=Reading(
                crop=Crop(mm=arguments.crop_mm, centre=arguments.centre),
                band=Band(low=arguments.band_low, high=arguments.band_high),
            ),
            first=arguments.cache_studies,
        )
