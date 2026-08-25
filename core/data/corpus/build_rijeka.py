"""Ingest the kneeMRI corpus: `python -m core.data.corpus.build_rijeka --archive PATH`.

A second corpus for pretraining, written into the cache shape the trainer already reads. See
`rijeka_corpus` for what is shared with the canonical decode and why, and `slots.RIJEKA_SCHEME` for why
the scheme name is a safety requirement rather than a label.
"""

import logging
from pathlib import Path

from core import cli
from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.rijeka_corpus import RIJEKA_SPACING, RijekaCorpus
from core.data.corpus.slots import RIJEKA_SCHEME
from core.wakelock import Wakelock

logger = logging.getLogger(__name__)

LABELS = "rijeka_acl.csv"


def main() -> None:
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument("--archive", type=Path, required=True, help="the kneeMRI dataset zip")
    command.parser.add_argument(
        "--spacing",
        type=float,
        default=RIJEKA_SPACING,
        help="millimetres per pixel in plane. Derived rather than recorded — the corpus states no "
        "spacing — and checked against its own ACL bounding boxes; see RIJEKA_SPACING.",
    )
    command.parser.add_argument("--out", type=Path, default=None, help="cache directory (default: the processed root)")
    arguments = command.parsed()

    if arguments.scheme != RIJEKA_SCHEME:
        # THE ONE ERROR THAT COSTS 71.6 GB. `Routing.suffix()` is empty for the canonical scheme, so a
        # 336px/12sl build under it writes to `images_336px_12sl.npy` — the arrays every member on disk
        # was fitted on — and destroys them with a corpus differing only in its first axis. The flag
        # defaults to canonical for every other CLI, so this refuses rather than assumes.
        command.parser.error(f"--scheme {RIJEKA_SCHEME} is required: any other scheme writes onto our own corpus")

    root = command.rooted(arguments)
    cache = CacheFlags.cache(arguments, arguments.out or root.processed())
    corpus = RijekaCorpus(arguments.archive, cache, spacing=arguments.spacing)

    corpus.labels(root.processed() / "labels" / LABELS)
    logger.info(f"building {cache.shape(len(corpus.volumes()))} uint8 -> {cache.images_path}")
    # The decode runs unattended for minutes and the idle timer counts INPUT rather than load (knee-wyzj).
    with Wakelock():
        corpus.build()
    logger.info("RIJEKA BUILD COMPLETE")


if __name__ == "__main__":
    main()
