"""Build the decode-once training cache: `python -m kneemri.build_cache [--slices N] [--workers N]`.

A real module rather than a stdin script because the decode runs in a process pool: on Windows the pool
spawns fresh interpreters that re-import the parent module, and a `__main__` read from stdin has no file
to import, so every child dies with BrokenProcessPool.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from core.paths import DataRoot
from kneemri.crop_centre import CENTRES, CROP_MM, DEFAULT_CENTRE, Crop
from kneemri.folds import ReportFolds
from kneemri.slot_reader import IMAGE_SIZE, SLICE_BAND, Band, Reading
from kneemri.slots import CANONICAL_SCHEME, FIRST_MAXIMAL, LAST_MAXIMAL, SCHEMES, Routing
from kneemri.study_cache import StudyCache

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slices", type=int, default=12, help="slices cached per slot")
    parser.add_argument(
        "--scheme",
        choices=tuple(SCHEMES),
        default=CANONICAL_SCHEME,
        help="which slot scheme to route with. 'extended' adds COR_FLUID_NOFS, AX_T1 and "
        "AX_FLUID_NOFS, the three combinations the canonical six cannot express - measured at 96%% of "
        "the 2,421 series (9.9%%) that currently match no slot at all. It writes a SEPARATE cache; the "
        "canonical arrays and the members fitted on them are untouched.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="in-plane resolution. Bytes grow with the SQUARE of this, and the cache path carries it, "
        "so a new resolution never disturbs an existing cache or the members fitted on it.",
    )
    parser.add_argument(
        "--crop-mm",
        type=float,
        default=CROP_MM,
        help="physical extent cropped about the centre. The default keeps the historic cache NAME, so "
        "any other value writes to its own file rather than over the arrays every member was fitted on.",
    )
    parser.add_argument(
        "--centre",
        default=DEFAULT_CENTRE,
        choices=sorted(CENTRES),
        help="where the physical crop is taken about. 'image' is the historic behaviour and keeps the "
        "historic cache name; 'tissue' locates the knee, which a tight crop needs (knee-yrf).",
    )
    parser.add_argument(
        "--band",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=list(SLICE_BAND),
        help="the fraction of each series' depth the slices are drawn from. The default is the "
        "historic reading and keeps the historic cache name; a wider band is a different array under "
        "the same shape, so it is named for itself rather than written over what is there.",
    )
    parser.add_argument(
        "--print-key",
        action="store_true",
        help="print the cache key this configuration resolves to and exit. The arm script used to "
        "rebuild the key in bash, so every new reading axis silently broke its own completion check.",
    )
    parser.add_argument(
        "--tie",
        default=FIRST_MAXIMAL,
        choices=(FIRST_MAXIMAL, LAST_MAXIMAL),
        help="which of several equally dense candidates fills a slot. 'first' is ours and keeps the "
        "historic cache name; the published members reproduce their own predictions better under "
        "'last' (knee-evn), so a graft-matching cache is built with it.",
    )
    parser.add_argument("--workers", type=int, default=20, help="decode processes")
    parser.add_argument("--split", default="train_series", help="which split to decode")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--holdout-fold", type=int, default=None, help="cache only this fold's holdout studies")
    parser.add_argument("--out", type=Path, default=None, help="cache directory (default: the processed root)")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = DataRoot.from_config(arguments.paths)
    raw = root.raw()
    table = "train.csv" if arguments.split == "train_series" else "test.csv"
    identifiers = pd.read_csv(raw / table)["StudyInstanceUID"].tolist()
    planes = StudyCache.planes_of(raw / arguments.split.replace("_series", "_series.csv"))

    if arguments.holdout_fold is not None:
        # A whole-corpus cache at 32 slices would be 95 GB to answer a question about one fold's
        # holdout. Restricting to the studies actually being scored keeps it at 20 GB.
        folds = ReportFolds()
        _, holdout = folds.holdout(folds.assign(str(raw / "train.csv"), identifiers), arguments.holdout_fold)
        identifiers = [identifiers[index] for index in holdout]
        logger.info(f"restricted to fold {arguments.holdout_fold}'s holdout: {len(identifiers)} studies")

    cache = StudyCache(
        arguments.out or root.processed(),
        slices=arguments.slices,
        image_size=arguments.image_size,
        routing=Routing(scheme=arguments.scheme, tie=arguments.tie),
        reading=Reading(crop=Crop(mm=arguments.crop_mm, centre=arguments.centre), band=Band.of(arguments.band)),
    )
    if arguments.print_key:
        # The arm script's completion check needs the same key StudyCache writes, and it used to
        # rebuild it in bash — so every reading axis added since silently broke that check.
        #
        # STDOUT, because the key is the ANSWER and not a log line. `logging` writes to stderr, so
        # `KEY="$(... --print-key)"` captured the empty string while the key still appeared on the
        # terminal — and the arm then checked `mask_.npy`, reporting ARM FAILED over a decode that had
        # written all 4407 studies and its mask (2026-08-15).
        sys.stdout.write(f"{cache.key}\n")
        return
    logger.info(f"building {cache.shape(len(identifiers))} uint8 -> {cache.images_path}")
    cache.build(raw / arguments.split, identifiers, planes, workers=arguments.workers)
    logger.info("CACHE BUILD COMPLETE")


if __name__ == "__main__":
    main()
