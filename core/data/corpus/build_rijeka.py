"""Ingest the kneeMRI corpus: `python -m core.data.corpus.build_rijeka --archive PATH`.

A second corpus for pretraining, written into the cache shape the trainer already reads. See
`rijeka_corpus` for what is shared with the canonical decode and why, and `slots.RIJEKA_SCHEME` for why
the scheme name is a safety requirement rather than a label.
"""

import logging
from pathlib import Path

from core import cli
from core.data.corpus.corpus_build import CorpusBuild
from core.data.corpus.rijeka_corpus import RIJEKA_SPACING, RijekaCorpus
from core.data.corpus.slots import RIJEKA_SCHEME

logger = logging.getLogger(__name__)

LABELS = "rijeka_acl.csv"
BUILD = CorpusBuild(RIJEKA_SCHEME, "RIJEKA", metadata="rijeka_metadata.csv")


def main() -> None:
    command = cli.Command(__doc__, logger)
    parser = BUILD.opening(command)
    parser.add_argument("--archive", type=Path, required=True, help="the kneeMRI dataset zip")
    parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=RIJEKA_SPACING,
        help="millimetres per pixel in plane. Derived rather than recorded — the corpus states no "
        "spacing — and checked against its own ACL bounding boxes; see RIJEKA_SPACING. Named apart "
        "from CacheFlags' `--spacing`, which selects how slices sit inside the band: two subjects, "
        "one word, and argparse refused the collision rather than letting either flag win.",
    )
    site = BUILD.opened(command)
    corpus = RijekaCorpus(site.arguments.archive, site.cache, spacing=site.arguments.pixel_spacing)
    corpus.labels(site.root.processed() / "labels" / LABELS)
    BUILD.run(site, corpus, corpus.volumes)


if __name__ == "__main__":
    main()
