"""Write MRNet's ACL table from a cache that is already built: `python -m core.data.corpus.build_mrnet_labels`.

Separate from `build_mrnet` because it needs no archive, and that is the entire point: the pixels here
were decoded on 2026-08-29 from a download that is no longer on the disk, so `build_mrnet` — which reads
its join, its labels and its pixels through one `zipfile` handle — cannot be re-run to produce the one
file that was missed. The cache's recorded index holds the same join that build resolved.

The decode is NOT repeated and cannot be from here. If the pixels are absent, run `build_mrnet`.
"""

import logging
from pathlib import Path

from core import cli
from core.data.corpus.build_mrnet import LABELS
from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.mrnet_cached import CachedMRNet
from core.data.corpus.mrnet_labels import TableLabels
from core.data.corpus.slots import MRNET_SCHEME

logger = logging.getLogger(__name__)


def main() -> None:
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="directory holding the ACL tables, matched by glob on split and finding",
    )
    command.parser.add_argument("--cache", type=Path, default=None, help="cache directory (default: processed root)")
    arguments = command.parsed()

    if arguments.scheme != MRNET_SCHEME:
        command.parser.error(f"--scheme {MRNET_SCHEME} is required: the cache to read is the one build_mrnet wrote")

    root = command.rooted(arguments)
    cache = CacheFlags.cache(arguments, arguments.cache or root.processed())
    CachedMRNet(cache, TableLabels(arguments.labels_dir)).labels(root.processed() / "labels" / LABELS)


if __name__ == "__main__":
    main()
