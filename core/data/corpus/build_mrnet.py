"""Ingest the MRNet corpus: `python -m core.data.corpus.build_mrnet --archive PATH`.

A third corpus, written into the cache shape the trainer already reads. See `mrnet_corpus` for what is
shared with the canonical decode and why only ACL is supervised, and `slots.MRNET_SCHEME` for why the
scheme name is a safety requirement rather than a label.
"""

import logging
from pathlib import Path

from core import cli
from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.mrnet_corpus import MRNET_SPACING, MRNetCorpus
from core.data.corpus.slots import MRNET_SCHEME
from core.wakelock import Wakelock

logger = logging.getLogger(__name__)

LABELS = "mrnet_acl.csv"


def main() -> None:
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument("--archive", type=Path, required=True, help="the MRNet-v1.0 dataset zip")
    command.parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=MRNET_SPACING,
        help="millimetres per pixel in plane. Derived rather than recorded — the corpus states no "
        "spacing — and weaker evidence than Rijeka's; see MRNET_SPACING. Named apart from CacheFlags' "
        "`--spacing`, which selects how slices sit inside the band: two subjects, one word, and argparse "
        "refused the collision rather than letting either flag win.",
    )
    command.parser.add_argument("--out", type=Path, default=None, help="cache directory (default: the processed root)")
    arguments = command.parsed()

    if arguments.scheme != MRNET_SCHEME:
        # THE ONE ERROR THAT COSTS 71.6 GB, and the third corpus is the third chance at it.
        # `Routing.suffix()` is empty for the canonical scheme, so a 336px/12sl build under it writes to
        # `images_336px_12sl.npy` — the arrays every member on disk was fitted on — and destroys them
        # with a corpus differing only in its first axis. The flag defaults to canonical for every other
        # CLI, so this refuses rather than assumes.
        command.parser.error(f"--scheme {MRNET_SCHEME} is required: any other scheme writes onto our own corpus")

    root = command.rooted(arguments)
    cache = CacheFlags.cache(arguments, arguments.out or root.processed())
    corpus = MRNetCorpus(arguments.archive, cache, spacing=arguments.pixel_spacing)

    corpus.labels(root.processed() / "labels" / LABELS)
    logger.info(f"building {cache.shape(len(corpus.volumes()))} uint8 -> {cache.images_path}")
    # The decode runs unattended for minutes and the idle timer counts INPUT rather than load (knee-wyzj).
    with Wakelock():
        corpus.build()
    logger.info("MRNET BUILD COMPLETE")


if __name__ == "__main__":
    main()
