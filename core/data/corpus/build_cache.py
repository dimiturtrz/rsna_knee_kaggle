"""Build the decode-once training cache: `python -m core.data.corpus.build_cache [--slices N] [--workers N]`.

A real module rather than a stdin script because the decode runs in a process pool: on Windows the pool
spawns fresh interpreters that re-import the parent module, and a `__main__` read from stdin has no file
to import, so every child dies with BrokenProcessPool.
"""

import logging
import sys
from pathlib import Path

from core import cli
from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.folds import ReportFolds
from core.data.corpus.slots import FOREIGN_SCHEMES
from core.data.corpus.study import Study
from core.data.corpus.study_cache import CorpusSplit
from core.data.corpus.study_metadata import StudyMetadata
from core.wakelock import Wakelock

logger = logging.getLogger(__name__)

METADATA = "canonical_metadata.csv"


def main() -> None:
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument(
        "--print-key",
        action="store_true",
        help="print the cache key this configuration resolves to and exit. The arm script used to "
        "rebuild the key in bash, so every new reading axis silently broke its own completion check.",
    )
    command.parser.add_argument("--workers", type=int, default=20, help="decode processes")
    command.parser.add_argument("--split", default="train_series", help="which split to decode")
    command.parser.add_argument("--holdout-fold", type=int, default=None, help="cache only this fold's holdout studies")
    command.parser.add_argument(
        "--only-studies",
        type=Path,
        default=None,
        help="a text file of study identifiers, one per line: decode only those, in split order. The "
        "generic form of --holdout-fold, for a subset no flag here can name — the 58 gold-annotated "
        "studies being the one that matters, since `foreign_gate` scores a published checkpoint on "
        "exactly them and a whole-corpus cache at 28 slices is 84 GB to answer a 58-study question.",
    )
    command.parser.add_argument("--out", type=Path, default=None, help="cache directory (default: the processed root)")
    command.parser.add_argument(
        "--metadata-only",
        action="store_true",
        help=f"write this corpus's own {METADATA} table and stop, without decoding a pixel. One header "
        "per study against a decode that runs for hours, and the two are separable for the reason "
        "`build_fastmri --labels-only` is: what an axis MEANS is argued about far more often than the "
        "images change, and every other corpus already writes its metadata beside its labels.",
    )
    arguments = command.parsed()

    if arguments.scheme in FOREIGN_SCHEMES:
        # This builder always decodes OUR study list; the scheme only names the output file. So a
        # foreign scheme here writes our corpus under a name that promises another one, and for
        # `merged` the shape matches too — see FOREIGN_SCHEMES for the 59.7 GB that cost.
        command.parser.error(
            f"--scheme {arguments.scheme} names a corpus this builder cannot decode: it would write OUR "
            f"studies under that name. Use core.data.corpus.build_{arguments.scheme}"
        )

    root = command.rooted(arguments)
    raw = root.raw()
    split = CorpusSplit(raw, arguments.split)
    identifiers = split.identifiers()

    if (arguments.holdout_fold is not None or arguments.only_studies is not None) and arguments.out is None:
        # A FOLD RESTRICTION IS NOT IN THE KEY, unlike `--cache-studies`, so a fold cache written into
        # the processed root lands on the full corpus's own filename and destroys it — 71.6 GB that
        # cost two hours, differing only in one axis length. It cannot be named for itself either:
        # `train.py` has no fold-restriction flag, so a suffix here would make the arrays it already
        # reads unaddressable. So the directory is what separates them, and it is now required.
        command.parser.error("a study restriction writes a partial corpus under the FULL cache's name: pass --out")

    if arguments.only_studies is not None:
        identifiers = split.named(arguments.only_studies).chosen
        logger.info(f"restricted to {arguments.only_studies}: {len(identifiers)} studies")

    if arguments.holdout_fold is not None:
        # A whole-corpus cache at 32 slices would be 95 GB to answer a question about one fold's
        # holdout. Restricting to the studies actually being scored keeps it at 20 GB.
        folds = ReportFolds()
        _, holdout = folds.holdout(folds.assign(str(raw / "train.csv"), identifiers), arguments.holdout_fold)
        identifiers = [identifiers[index] for index in holdout]
        logger.info(f"restricted to fold {arguments.holdout_fold}'s holdout: {len(identifiers)} studies")

    if arguments.metadata_only:
        planes = split.planes()
        entries = [Study.from_directory(split.source / identifier, planes).metadata() for identifier in identifiers]
        StudyMetadata.write(entries, root.processed() / "metadata" / METADATA)
        logger.info("CANONICAL METADATA COMPLETE (no decode requested)")
        return

    cache = CacheFlags.cache(arguments, arguments.out or root.processed())
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
    logger.info(f"building {cache.shape(len(cache.enrolled(identifiers)))} uint8 -> {cache.images_path}")
    # Held awake for the decode alone, not for the argument parsing or the key print: a full-corpus
    # build runs for hours unattended, and the idle timer counts INPUT rather than load (knee-wyzj).
    with Wakelock():
        cache.build(split.source, identifiers, split.planes(), workers=arguments.workers)
    logger.info("CACHE BUILD COMPLETE")


if __name__ == "__main__":
    main()
