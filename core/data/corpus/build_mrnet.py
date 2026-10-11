"""Ingest the MRNet corpus: `python -m core.data.corpus.build_mrnet --archive PATH`.

A third corpus, written into the cache shape the trainer already reads. See `mrnet_corpus` for what is
shared with the canonical decode and why only ACL is supervised, and `slots.MRNET_SCHEME` for why the
scheme name is a safety requirement rather than a label.
"""

import logging
from pathlib import Path

from core import cli
from core.data.corpus.corpus_build import CorpusBuild
from core.data.corpus.mrnet_corpus import MRNET_SPACING, MRNetCorpus
from core.data.corpus.mrnet_labels import TableLabels
from core.data.corpus.slots import MRNET_SCHEME

logger = logging.getLogger(__name__)

LABELS = "mrnet_acl.csv"
BUILD = CorpusBuild(MRNET_SCHEME, "MRNET", metadata="mrnet_metadata.csv")


def main() -> None:
    command = cli.Command(__doc__, logger)
    parser = BUILD.opening(command)
    parser.add_argument("--archive", type=Path, required=True, help="the MRNet-v1.0 dataset zip")
    parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=MRNET_SPACING,
        help="millimetres per pixel in plane. Derived rather than recorded — the corpus states no "
        "spacing — and weaker evidence than Rijeka's; see MRNET_SPACING. Named apart from CacheFlags' "
        "`--spacing`, which selects how slices sit inside the band: two subjects, one word, and argparse "
        "refused the collision rather than letting either flag win.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=None,
        help="directory holding the ACL tables, when the archive ships PIXELS ALONE. The Redivis export "
        "of this corpus does exactly that — 3,750 arrays and no CSV — and its tables are a separate "
        "download. Omit for the Stanford archive, which carries them inside.",
    )
    site = BUILD.opened(command)
    arguments = site.arguments
    tables = TableLabels(arguments.labels_dir) if arguments.labels_dir else None
    corpus = MRNetCorpus(arguments.archive, site.cache, spacing=arguments.pixel_spacing, tables=tables)
    corpus.labels(site.root.processed() / "labels" / LABELS)
    BUILD.run(site, corpus, corpus.volumes)


if __name__ == "__main__":
    main()
