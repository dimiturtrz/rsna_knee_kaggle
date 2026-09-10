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
from core.data.corpus.study_metadata import StudyMetadata
from core.wakelock import Wakelock

logger = logging.getLogger(__name__)

LABELS = "report_labels_v2.csv"
EXTRA_LABELS = "rijeka_acl.csv"
MERGED_LABELS = "merged_labels.csv"
METADATA = "metadata"
"""The directory each corpus's own metadata CSV is written to by its own builder."""


def main() -> None:
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument(
        "--extra-scheme", default=RIJEKA_SCHEME, choices=sorted(SCHEMES), help="the corpus being added"
    )
    command.parser.add_argument(
        "--merged-scheme",
        default=MERGED_SCHEME,
        choices=sorted(SCHEMES),
        help=(
            f"the cache being written (default: {MERGED_SCHEME}). Name a DIFFERENT one to merge a merge "
            f"— `--scheme {MERGED_SCHEME} --extra-scheme mrnet --merged-scheme merged3` adds a third "
            "corpus without growing the cache every merged member on disk was fitted on."
        ),
    )
    command.parser.add_argument("--labels", type=Path, default=None, help=f"our labels table (default: {LABELS})")
    command.parser.add_argument("--extra-labels", type=Path, default=None, help=f"theirs (default: {EXTRA_LABELS})")
    command.parser.add_argument("--out", type=Path, default=None, help="cache directory (default: the processed root)")
    command.parser.add_argument(
        "--labels-out",
        type=Path,
        default=None,
        help=(
            f"where the union table is written (default: {MERGED_LABELS}). A run that changes --labels MUST "
            "change this too: the default name is what every merged member on disk was fitted against, and "
            "overwriting it would leave those runs unreproducible with every gate still green."
        ),
    )
    command.parser.add_argument(
        "--labels-only",
        action="store_true",
        help=(
            "write the union table and stop. The images are a 41.8 GB sequential copy and the table is a "
            "concat of two CSVs, and a run that changes only which labels our rows carry needs the second "
            "without paying for the first — the arrays a label table indexes are the same arrays."
        ),
    )
    command.parser.add_argument(
        "--narrow-corpus",
        default=None,
        help=(
            "after merging, write a SECOND table in which this corpus's rows are blanked outside "
            "--narrow-graded. A corpus that grades three of our twelve findings supplies negatives on "
            "the other nine, and merging it shifts the prior toward absence on findings it cannot "
            "locate; blanking is how that hazard is measured against the same merge rather than argued."
        ),
    )
    command.parser.add_argument(
        "--narrow-graded",
        default="",
        help="comma-separated findings that corpus reads one-to-one, kept as scored; everything else NaN",
    )
    command.parser.add_argument("--narrow-out", type=Path, default=None, help="where that second table is written")
    arguments = command.parsed()

    if arguments.merged_scheme in (arguments.scheme, arguments.extra_scheme):
        # A source may be a merge — `merged3` is built by merging one — but the DESTINATION may never be
        # either source, which would read a cache as its own base and grow the same file on every run.
        command.parser.error(f"{arguments.merged_scheme} is the destination, so it cannot also be a source")

    root = command.rooted(arguments)
    directory = arguments.out or root.processed()
    base = CacheFlags.cache(arguments, directory)
    merger = MergedCache(base, base.under(arguments.extra_scheme), base.under(arguments.merged_scheme))

    merger.labels(
        arguments.labels or root.processed() / "labels" / LABELS,
        arguments.extra_labels or root.processed() / "labels" / EXTRA_LABELS,
        arguments.labels_out or root.processed() / "labels" / f"{arguments.merged_scheme}_labels.csv",
        base_name=arguments.scheme,
        extra_name=arguments.extra_scheme,
    )
    StudyMetadata.merge_written(
        root.processed() / METADATA, (arguments.scheme, arguments.extra_scheme), arguments.merged_scheme
    )
    destination = arguments.labels_out or root.processed() / "labels" / f"{arguments.merged_scheme}_labels.csv"
    if arguments.labels_only:
        # The exclusion applies here too, or a labels-only rebuild onto an existing cache silently puts
        # every black study back into a table the previous build had cleaned.
        merger.drop_unslotted(destination)
        merger.narrowed(destination, arguments.narrow_corpus, arguments.narrow_graded, arguments.narrow_out)
        logger.info("LABELS ONLY — the images were left alone")
        return
    # An hour of sequential copy with no keyboard input, and the idle timer counts INPUT rather than load.
    with Wakelock():
        merger.build()
    merger.drop_unslotted(destination)
    merger.narrowed(destination, arguments.narrow_corpus, arguments.narrow_graded, arguments.narrow_out)
    logger.info("MERGE COMPLETE")


if __name__ == "__main__":
    main()
