"""Ingest the KMAR-50K clean side: `python -m core.data.corpus.build_kmar --archive PATH`.

A FOURTH corpus, written into the cache shape the trainer already reads. See `kmar_studies` for how a
subject's acquisitions are grouped and routed, and `slots.KMAR_SCHEME` for why the scheme name is a
safety requirement rather than a label.

IT IS A DIRECTORY, NOT AN ARCHIVE. Rijeka, MRNet and fastMRI each ship one zip this project reads in
place; KMAR-50K ships two extracted part trees, so `--archive` takes the root that holds them. The flag
keeps its name because every other builder spells it that way and a fourth spelling for the same
argument is how a runbook comes to be wrong.
"""

import logging
from pathlib import Path

from core import cli
from core.data.corpus.corpus_build import CorpusBuild
from core.data.corpus.kmar_studies import KmarStudies
from core.data.corpus.slots import KMAR_SCHEME

logger = logging.getLogger(__name__)

BUILD = CorpusBuild(KMAR_SCHEME, "KMAR", metadata="kmar_metadata.csv")


def main() -> None:
    command = cli.Command(__doc__, logger)
    parser = BUILD.opening(command)
    parser.add_argument("--archive", type=Path, required=True, help="the KMAR-50K root holding the part directories")
    site = BUILD.opened(command)
    corpus = KmarStudies(site.arguments.archive, site.cache)
    # NO LABELS TABLE IS WRITTEN, unlike every other builder here. This corpus records no finding of any
    # kind, and an empty table would read like a corpus that was labelled and came back negative.
    BUILD.run(site, corpus, corpus.studies)


if __name__ == "__main__":
    main()
