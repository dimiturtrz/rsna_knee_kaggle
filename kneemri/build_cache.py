"""Build the decode-once training cache: `python -m kneemri.build_cache [--slices N] [--workers N]`.

A real module rather than a stdin script because the decode runs in a process pool: on Windows the pool
spawns fresh interpreters that re-import the parent module, and a `__main__` read from stdin has no file
to import, so every child dies with BrokenProcessPool.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from core.paths import DataRoot
from kneemri.study_cache import StudyCache

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slices", type=int, default=12, help="slices cached per slot")
    parser.add_argument("--workers", type=int, default=20, help="decode processes")
    parser.add_argument("--split", default="train_series", help="which split to decode")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = DataRoot.from_config(arguments.paths)
    raw = root.raw()
    table = "train.csv" if arguments.split == "train_series" else "test.csv"
    identifiers = pd.read_csv(raw / table)["StudyInstanceUID"].tolist()
    planes = StudyCache.planes_of(raw / arguments.split.replace("_series", "_series.csv"))

    cache = StudyCache(root.processed(), slices=arguments.slices)
    logger.info(f"building {cache.shape(len(identifiers))} uint8 -> {cache.images_path}")
    cache.build(raw / arguments.split, identifiers, planes, workers=arguments.workers)
    logger.info("CACHE BUILD COMPLETE")


if __name__ == "__main__":
    main()
