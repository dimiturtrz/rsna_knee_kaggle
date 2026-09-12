"""Ingest the KMAR-50K clean side: `python -m core.data.corpus.build_kmar --archive PATH`.

A FOURTH corpus, written into the cache shape the trainer already reads. See `kmar_studies` for how a
subject's acquisitions are grouped and routed, and `slots.KMAR_SCHEME` for why the scheme name is a
safety requirement rather than a label.

IT IS A DIRECTORY, NOT AN ARCHIVE. Rijeka, MRNet and fastMRI each ship one zip this project reads in
place; KMAR-50K ships two extracted part trees, so `--archive` takes the root that holds them. The flag
keeps its name because every other builder spells it that way and a fourth spelling for the same
argument is how a runbook comes to be wrong.
"""

import logging
from pathlib import Path

from core import cli
from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.kmar_studies import KmarStudies
from core.data.corpus.slots import KMAR_SCHEME
from core.data.corpus.study_metadata import StudyMetadata
from core.wakelock import Wakelock

logger = logging.getLogger(__name__)

METADATA = "kmar_metadata.csv"


def main() -> None:
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument(
        "--archive", type=Path, required=True, help="the KMAR-50K root holding the part directories"
    )
    command.parser.add_argument("--out", type=Path, default=None, help="cache directory (default: the processed root)")
    arguments = command.parsed()

    if arguments.scheme != KMAR_SCHEME:
        # THE ONE ERROR THAT COSTS 71.6 GB, and the fourth corpus is the fourth chance at it.
        # `Routing.suffix()` is empty for the canonical scheme, so a 336px/12sl build under it writes to
        # `images_336px_12sl.npy` — the arrays every member on disk was fitted on — and destroys them
        # with a corpus differing only in its first axis. The flag defaults to canonical for every other
        # CLI, so this refuses rather than assumes.
        command.parser.error(f"--scheme {KMAR_SCHEME} is required: any other scheme writes onto our own corpus")

    root = command.rooted(arguments)
    cache = CacheFlags.cache(arguments, arguments.out or root.processed())
    corpus = KmarStudies(arguments.archive, cache)

    # NO LABELS TABLE IS WRITTEN, unlike every other builder here. This corpus records no finding of any
    # kind, and an empty table would read like a corpus that was labelled and came back negative.
    StudyMetadata.write(corpus.metadata(), root.processed() / "metadata" / METADATA)
    logger.info(f"building {cache.shape(len(corpus.studies()))} uint8 -> {cache.images_path}")
    # The decode runs unattended for minutes and the idle timer counts INPUT rather than load (knee-wyzj).
    with Wakelock():
        corpus.build()
    logger.info("KMAR BUILD COMPLETE")


if __name__ == "__main__":
    main()
