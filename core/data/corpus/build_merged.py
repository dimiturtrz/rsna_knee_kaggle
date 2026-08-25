"""Merge two decoded corpora into one cache: `python -m core.data.corpus.build_merged`.

Disk and CPU only — no card, no submission slot. See `merged_cache` for why the concatenation is legal
and `slots.MERGED_SCHEME` for why the destination needs a scheme name of its own.
"""

import logging
from pathlib import Path

from core import cli
from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.merged_cache import MergedCache
from core.data.corpus.slots import MERGED_SCHEME, RIJEKA_SCHEME, SCHEMES
from core.wakelock import Wakelock

logger = logging.getLogger(__name__)

LABELS = "report_labels_v2.csv"
EXTRA_LABELS = "rijeka_acl.csv"
MERGED_LABELS = "merged_labels.csv"


def main() -> None:
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument(
        "--extra-scheme", default=RIJEKA_SCHEME, choices=sorted(SCHEMES), help="the corpus being added"
    )
    command.parser.add_argument("--labels", type=Path, default=None, help=f"our labels table (default: {LABELS})")
    command.parser.add_argument("--extra-labels", type=Path, default=None, help=f"theirs (default: {EXTRA_LABELS})")
    command.parser.add_argument("--out", type=Path, default=None, help="cache directory (default: the processed root)")
    arguments = command.parsed()

    if MERGED_SCHEME in (arguments.scheme, arguments.extra_scheme):
        # Both flags name a SOURCE; the destination is always the merged one. Letting a source be the
        # destination too would read the merged cache as its own base and grow it on every run.
        command.parser.error(f"{MERGED_SCHEME} is the destination, never a source")

    root = command.rooted(arguments)
    directory = arguments.out or root.processed()
    base = CacheFlags.cache(arguments, directory)
    merger = MergedCache(base, base.under(arguments.extra_scheme), base.under(MERGED_SCHEME))

    merger.labels(
        arguments.labels or root.processed() / "labels" / LABELS,
        arguments.extra_labels or root.processed() / "labels" / EXTRA_LABELS,
        root.processed() / "labels" / MERGED_LABELS,
    )
    # An hour of sequential copy with no keyboard input, and the idle timer counts INPUT rather than load.
    with Wakelock():
        merger.build()
    logger.info("MERGE COMPLETE")


if __name__ == "__main__":
    main()
