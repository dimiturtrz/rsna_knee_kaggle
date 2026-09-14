"""Ingest the fastMRI knee corpus: `python -m core.data.corpus.build_fastmri --root PATH`.

A third corpus for pretraining, written into the cache shape the trainer already reads. See
`fastmri_corpus` for what its labels can and cannot say — most of what they carry is absence — and
`slots.FASTMRI_SCHEME` for why the scheme name is a safety requirement rather than a label.

`--labels-only` writes the table without touching the volumes. The label mapping is where the argument
lives and the decode is where the hours go, so the two are separable on purpose: the table can be read,
argued with and rewritten while the images stay where they are.
"""

import logging
from pathlib import Path

from core import cli
from core.data.corpus.cache_flags import CacheFlags
from core.data.corpus.fastmri_boxes import BOX_TABLE, STUDY_TABLE, FastmriBoxes
from core.data.corpus.fastmri_corpus import READER_CONFIDENCE, FastmriCorpus
from core.data.corpus.slots import FASTMRI_SCHEME
from core.data.corpus.study_metadata import StudyMetadata
from core.wakelock import Wakelock

logger = logging.getLogger(__name__)

LABELS = "fastmri_findings.csv"
METADATA = "fastmri_metadata.csv"
ANNOTATIONS = "labels/Annotations"


def main() -> None:
    command = cli.Command(__doc__, logger)
    CacheFlags.add_to(command.parser)
    command.parser.add_argument("--root", type=Path, required=True, help="the directory holding singlecoil_* volumes")
    command.parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=f"the fastMRI+ Annotations directory (default: <root>/{ANNOTATIONS})",
    )
    command.parser.add_argument(
        "--reader-confidence",
        type=float,
        default=READER_CONFIDENCE,
        help="weight for every row of this corpus. A STATED PRIOR, not a measurement — these are "
        "single-reader, single-view reads the corpus itself calls indicative, against Rijeka's "
        "double-blind 1.0. See fastmri_corpus.READER_CONFIDENCE before quoting it.",
    )
    command.parser.add_argument(
        "--labels-out",
        type=Path,
        default=None,
        help=(
            f"where the table is written (default: {LABELS}). A run that changes what a term READS must "
            "change this too, for the reason `build_merged` gives: the default name is what the members "
            "on disk were fitted against, and overwriting it leaves them unreproducible with every gate "
            "still green."
        ),
    )
    command.parser.add_argument("--labels-only", action="store_true", help="write the table, skip the decode")
    command.parser.add_argument(
        "--boxes",
        action="store_true",
        help=f"write only the fastMRI+ box tables ({BOX_TABLE}, {STUDY_TABLE}) in this cache's frame, and exit",
    )
    command.parser.add_argument("--out", type=Path, default=None, help="cache directory (default: the processed root)")
    arguments = command.parsed()

    if arguments.scheme != FASTMRI_SCHEME:
        # THE ONE ERROR THAT COSTS 71.6 GB, refused here for the reason `build_rijeka` refuses it:
        # `Routing.suffix()` is empty for the canonical scheme, so a build under it writes to the arrays
        # every member on disk was fitted on and destroys them with a corpus differing only in axis 0.
        command.parser.error(f"--scheme {FASTMRI_SCHEME} is required: any other scheme writes onto our own corpus")

    root = command.rooted(arguments)
    cache = CacheFlags.cache(arguments, arguments.out or root.processed())
    corpus = FastmriCorpus(
        arguments.root,
        arguments.annotations or arguments.root / ANNOTATIONS,
        cache,
        confidence=arguments.reader_confidence,
    )
    if arguments.boxes:
        FastmriBoxes(corpus.annotations, cache).write(corpus.volumes(), root.processed() / "labels")
        logger.info("FASTMRI BOXES COMPLETE")
        return

    corpus.labels(arguments.labels_out or root.processed() / "labels" / LABELS)
    StudyMetadata.write(corpus.metadata(), root.processed() / "metadata" / METADATA)
    if arguments.labels_only:
        logger.info("FASTMRI LABELS COMPLETE (no decode requested)")
        return

    logger.info(f"building {cache.shape(len(corpus.volumes()))} uint8 -> {cache.images_path}")
    # The decode runs unattended for minutes and the idle timer counts INPUT rather than load (knee-wyzj).
    with Wakelock():
        corpus.build()
    logger.info("FASTMRI BUILD COMPLETE")


if __name__ == "__main__":
    main()
