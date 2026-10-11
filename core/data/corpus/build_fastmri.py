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
from core.data.corpus.corpus_build import CorpusBuild
from core.data.corpus.fastmri_boxes import BOX_TABLE, STUDY_TABLE, FastmriBoxes
from core.data.corpus.fastmri_corpus import READER_CONFIDENCE, FastmriCorpus
from core.data.corpus.slots import FASTMRI_SCHEME

logger = logging.getLogger(__name__)

LABELS = "fastmri_paired_findings.csv"
"""The knee-level table, named apart from the per-file `fastmri_findings.csv` it replaces.

Not a rename: the file-level table is what the members on disk were fitted against, and its rows are
keyed by an exam identifier this build no longer emits. Overwriting it would leave those members
unreproducible with every gate still green — the rule `--labels-out` states, applied to a change of
IDENTITY rather than of reading.
"""
ANNOTATIONS = "labels/Annotations"
BUILD = CorpusBuild(FASTMRI_SCHEME, "FASTMRI", metadata="fastmri_metadata.csv")


def main() -> None:
    command = cli.Command(__doc__, logger)
    parser = BUILD.opening(command)
    parser.add_argument("--root", type=Path, required=True, help="the directory holding singlecoil_* volumes")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help=f"the fastMRI+ Annotations directory (default: <root>/{ANNOTATIONS})",
    )
    parser.add_argument(
        "--reader-confidence",
        type=float,
        default=READER_CONFIDENCE,
        help="weight for every row of this corpus. A STATED PRIOR, not a measurement — these are "
        "single-reader, single-view reads the corpus itself calls indicative, against Rijeka's "
        "double-blind 1.0. See fastmri_corpus.READER_CONFIDENCE before quoting it.",
    )
    parser.add_argument(
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
    parser.add_argument("--labels-only", action="store_true", help="write the table, skip the decode")
    parser.add_argument(
        "--boxes",
        action="store_true",
        help=f"write only the fastMRI+ box tables ({BOX_TABLE}, {STUDY_TABLE}) in this cache's frame, and exit",
    )
    site = BUILD.opened(command)
    arguments = site.arguments
    corpus = FastmriCorpus(
        arguments.root,
        arguments.annotations or arguments.root / ANNOTATIONS,
        site.cache,
        confidence=arguments.reader_confidence,
    )
    labels = site.root.processed() / "labels"
    if arguments.boxes:
        FastmriBoxes(corpus.annotations, site.cache).write(corpus.volumes(), labels)
        logger.info("FASTMRI BOXES COMPLETE")
        return

    corpus.labels(arguments.labels_out or labels / LABELS)
    BUILD.run(site, corpus, corpus.studies, decode=not arguments.labels_only)


if __name__ == "__main__":
    main()
