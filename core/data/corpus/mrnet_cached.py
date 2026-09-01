"""MRNet's ACL table recovered from a cache that is ALREADY BUILT, without re-reading the archive.

`MRNetCorpus` reads its pixels, its member list and its label tables through one `zipfile` handle, so an
EXTRACTED download — three plane directories and no zip — cannot produce the table even when the decode
itself succeeded from some other copy. That is not hypothetical: this cache was built on 2026-08-29 and
then sat unusable for three days, because `labels/mrnet_acl.csv` was never written and the archive it
would have been written from is not on the disk any more.

The recorded index is the better denominator in any case. `MRNetCorpus.volumes` joins the label tables
against the archive's members precisely so no rate is ever reported over exams that never downloaded —
and the index IS that join, already resolved and already written down by the build that decoded the
pixels. Reading it back is therefore not a second opinion about which exams exist; it is the first one.
"""

import logging
from pathlib import Path

from core.data.corpus.mrnet_corpus import MRNET_LEVEL, PREFIX, MRNetCorpus
from core.data.corpus.mrnet_labels import SPLITS, MRNetLabels
from core.data.corpus.study_cache import StudyCache

logger = logging.getLogger(__name__)


class CachedMRNet:
    """The exams a built MRNet cache holds, joined to the ACL verdicts shipped beside it."""

    def __init__(self, cache: StudyCache, tables: MRNetLabels) -> None:
        self.cache = cache
        self.tables = tables

    def verdicts(self) -> dict[str, int]:
        """Every labelled case across both splits, keyed by the case number the identifiers carry.

        Both splits are training data here (`mrnet_labels.SPLITS`), so they are one table and the split
        a case came from is not recorded anywhere downstream.
        """
        return {case: int(verdict) for split in SPLITS for case, verdict in self.tables.table(split).items()}

    def volumes(self) -> list[tuple[str, float]]:
        """Each identifier the cache holds, with its ACL level, in the cache's own order.

        BOTH COUNTS ARE LOGGED because they are allowed to differ and the difference is the finding: a
        cached exam the tables do not name would otherwise reach the merged corpus as a silently
        unsupervised row, and a row absent from this table is one the trainer weights at 0.0 without
        saying so.
        """
        identifiers = self.cache.recorded()
        verdicts = self.verdicts()
        rows = [
            (identifier, MRNET_LEVEL[verdicts[identifier.removeprefix(PREFIX)]])
            for identifier in identifiers
            if identifier.removeprefix(PREFIX) in verdicts
        ]
        logger.info(f"{len(rows)} of {len(identifiers)} cached exams carry a label, from {len(verdicts)} published")
        return rows

    def labels(self, destination: Path) -> None:
        """The same table `MRNetCorpus.labels` writes, over the same exams, from the cache instead."""
        MRNetCorpus.write_labels(self.volumes(), destination)
