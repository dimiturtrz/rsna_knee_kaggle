"""Two decoded corpora as ONE cache, because the trainer reads exactly one.

`train.py` calls `cache.load()` once and gets a single `(corpus, identifiers)` pair, so two corpora on
disk cannot both reach one run however compatible they are. That is the whole gap between the kneeMRI
ingest and an arm that uses it, and it is a data-preparation job: no card, no slot, no new training code.

WHAT MAKES THE CONCATENATION LEGAL is three mechanisms that were already there, none of them written for
this (see `rijeka_corpus` for the audit):

  * `ReportFolds.assign` hashes `reports.get(identifier, identifier)`, so a foreign study folds on its
    own identifier rather than failing to be found.
  * `Targets.load` weights a target it was NOT told about at zero — an absent COLUMN by `Targets.pair`,
    and an absent CELL by `said = ~np.isnan(value)`. So one table carrying both corpora supervises each
    study on exactly the findings its own corpus read, and on nothing else.
  * `train.py` reindexes `train.csv` onto our identifiers and never reads the resulting NaN rows unless
    annotations are on, which they are not by default.

THE FOREIGN CORPUS KEEPS ITS SLOT POSITION rather than being appended to the axis. A kneeMRI exam is
sagittal PD fat-suppressed, which is `SLOTS[0]`'s predicate, so it lands in slot 0 and its other five
slots stay absent in the mask — the same shape a knee of ours that was never acquired axially has. The
head's softmax renormalises over present slots already; nothing downstream learns that these rows are
foreign, and nothing needs to.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float, UInt8

from core.data.corpus.study_cache import StudyCache

logger = logging.getLogger(__name__)

COPY_STUDIES = 32
"""Studies copied per block. At six slots, twelve slices and 336 px a study is 8.1 MB, so a block is
260 MB — large enough that the copy is sequential I/O and small enough to hold twice over in RAM."""


class MergedCache:
    """Writes one cache holding `base`'s studies followed by `extra`'s, into `merged`'s own files.

    The three caches are separate objects rather than derived from one another because each names a real
    set of files on disk, and the destination's key MUST differ from both sources' — see
    `slots.MERGED_SCHEME`.
    """

    def __init__(self, base: StudyCache, extra: StudyCache, merged: StudyCache) -> None:
        self.base = base
        self.extra = extra
        self.merged = merged

    def positions(self) -> list[int]:
        """Where each of `extra`'s slots lands in the merged slot axis, matched by slot NAME.

        By name and not by index: a foreign corpus's slot list is a SUBSET of the canonical one, and
        matching positionally would be correct only for a prefix and silently wrong for anything else.
        """
        axis = self.merged.scheme_slots
        return [axis.index(name) for name in self.extra.scheme_slots]

    def check(self) -> None:
        """Refuse a merge that would write onto a source, or concatenate arrays of different geometry."""
        for source in (self.base, self.extra):
            if source.images_path == self.merged.images_path:
                raise ValueError(f"the merge destination is a source: {self.merged.images_path}")
            if (source.slices, source.image_size) != (self.merged.slices, self.merged.image_size):
                raise ValueError(f"{source.key} is not the merged geometry {self.merged.key}")
        if self.base.scheme_slots != self.merged.scheme_slots:
            raise ValueError(f"{self.base.key} does not share the merged slot axis")

    def build(self) -> None:
        """Write the merged images, index and — last — mask, exactly as a decode would.

        The mask goes last for the reason every other builder here writes it last: its presence is what
        says the arrays beside it are complete, so a merge that dies half way leaves a cache that reads
        as unbuilt rather than as a corpus with black studies in it.
        """
        self.check()
        base_studies, extra_studies = self._index(self.base), self._index(self.extra)
        self._disjoint(base_studies, extra_studies)
        total = len(base_studies) + len(extra_studies)
        self.merged.prepare()
        images = np.lib.format.open_memmap(
            self.merged.images_path, mode="w+", dtype=np.uint8, shape=self.merged.shape(total)
        )
        self._copy(images, 0, np.load(self.base.images_path, mmap_mode="r"), slots=list(range(self.merged.slots)))
        self._copy(images, len(base_studies), np.load(self.extra.images_path, mmap_mode="r"), slots=self.positions())
        images.flush()
        self.merged.record(base_studies + extra_studies)
        mask = self._mask(len(base_studies), len(extra_studies))
        self.merged.report(mask)
        np.save(self.merged.mask_path, mask)
        logger.info(f"wrote {self.merged.shape(total)} uint8 -> {self.merged.images_path}")

    def labels(
        self,
        base: Path,
        extra: Path,
        destination: Path,
        *,
        base_name: str = "base",
        extra_name: str = "extra",
    ) -> None:
        """One labels table for both corpora, as the union of their columns.

        A study is scored on the findings ITS OWN table names and left NaN — weight 0.0, score 0.5 — on
        every other. Filling those cells with 0.5 instead would teach the model that every foreign knee
        is exactly average on eleven findings nobody looked at.

        The `corpus` column is added for Scheme B mixed sampling: it names which corpus each row came
        from without relying on identifier prefixes, which differ between MRNet (`mrnet-`), Rijeka
        (filename stems) and fastMRI (file names). Downstream loaders ignore columns they do not use,
        so adding it is safe for existing members.
        """
        base_table = pd.read_csv(base)
        extra_table = pd.read_csv(extra)
        base_table["corpus"] = base_name
        extra_table["corpus"] = extra_name
        merged = pd.concat([base_table, extra_table], ignore_index=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(destination, index=False)
        logger.info(f"{len(merged)} rows, {len(merged.columns)} columns -> {destination}")

    def _mask(self, base_studies: int, extra_studies: int) -> Float[np.ndarray, "studies slots"]:
        """The merged presence mask: base's own rows, then extra's flags at their matched positions."""
        mask = np.zeros((base_studies + extra_studies, self.merged.slots), dtype=np.float32)
        mask[:base_studies] = np.load(self.base.mask_path)
        mask[base_studies:, self.positions()] = np.load(self.extra.mask_path)
        return mask

    def _copy(
        self,
        images: UInt8[np.ndarray, "studies slots slices height width"],
        start: int,
        source: UInt8[np.ndarray, "source slots slices height width"],
        slots: list[int],
    ) -> None:
        """Block-copy one source's studies into the merged array at `start`, slot by matched slot.

        Blocked because the base corpus is 35.8 GB: a whole-array assignment between two memory maps
        materialises it, and the machine has less RAM than that to spare beside a running trainer.
        """
        for first in range(0, len(source), COPY_STUDIES):
            block = source[first : first + COPY_STUDIES]
            images[start + first : start + first + len(block), slots] = block
            logger.info(f"copied {first + len(block)}/{len(source)} into offset {start}")

    @staticmethod
    def _index(cache: StudyCache) -> list[str]:
        """A source cache's study order, which is also the assertion that it was finished."""
        if not cache.mask_path.exists():
            raise FileNotFoundError(f"{cache.key} has no mask, so it is a build that stopped rather than a corpus")
        return [str(identifier) for identifier in json.loads(cache.index_path.read_text(encoding="utf-8"))]

    @staticmethod
    def _disjoint(base: list[str], extra: list[str]) -> None:
        """Refuse identifiers present in both corpora — a duplicate row is a study weighted twice."""
        shared = set(base) & set(extra)
        if shared:
            raise ValueError(f"{len(shared)} identifiers are in both corpora, e.g. {sorted(shared)[:3]}")
