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

CORPUS = "corpus"
"""The origin column both this producer and `Targets.load` name. One spelling, because a table written
under a different one is not a broken read — it is a silently absent axis, which is how the column came
to be consumed and never written."""

SIBLING = "__"
"""What separates a finding from its own auxiliary column (`ACL__conf`, `ACL__verdict`). One finding's
columns are a family, and `narrow` keeps or blanks the whole family together."""

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

    def positions(self) -> tuple[list[int], list[int]]:
        """Which of `extra`'s slots the merged axis carries, and where each lands — matched by slot NAME.

        By name and not by index: a foreign corpus's slot list is a SUBSET of the canonical one, and
        matching positionally would be correct only for a prefix and silently wrong for anything else.

        A slot the destination axis does not name is DROPPED, not refused. `slots.FASTMRI_SLOTS` prices
        that drop in writing before any merge runs — `COR_FLUID_NOFS` is not among the canonical six, so
        a canonical merge carries the fat-suppressed half of that corpus and leaves the other behind.
        This method used to raise instead, which made the merge that docstring describes impossible to
        run: `ValueError: 'COR_FLUID_NOFS' is not in list`, on first invocation, 2026-09-10. A cost
        stated in one module and refused by another is the same defect either way, and the module that
        argued the cost is the one that was right.
        """
        axis = self.merged.scheme_slots
        carried = [(source, axis.index(slot)) for source, slot in enumerate(self.extra.scheme_slots) if slot in axis]
        if not carried:
            raise ValueError(f"{self.merged.key} names none of {self.extra.key}'s slots, so the merge carries nothing")
        return [source for source, _ in carried], [destination for _, destination in carried]

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
        whole = list(range(self.merged.slots))
        self._copy(images, 0, np.load(self.base.images_path, mmap_mode="r"), sources=whole, slots=whole)
        sources, slots = self.positions()
        self._copy(
            images, len(base_studies), np.load(self.extra.images_path, mmap_mode="r"), sources=sources, slots=slots
        )
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

        A SOURCE THAT ALREADY NAMES ITS CORPORA KEEPS THOSE NAMES. `merged3` is built by merging
        `merged`, whose rows come from two corpora, so stamping the scheme name over them would answer
        "which corpus is this study from" with the name of a merge — losing exactly the distinction the
        column exists to record, and losing it silently, on the only rows a third merge adds nothing to.
        """
        base_table = self._tagged(pd.read_csv(base), base_name)
        extra_table = self._tagged(pd.read_csv(extra), extra_name)
        merged = pd.concat([base_table, extra_table], ignore_index=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(destination, index=False)
        logger.info(f"{len(merged)} rows, {len(merged.columns)} columns -> {destination}")

    def drop_unslotted(self, destination: Path, *, identifier: str = "StudyInstanceUID") -> int:
        """Remove label rows for merged studies whose mask is entirely empty. Returns how many went.

        A slot the merged axis does not name is dropped by `positions`, and a foreign study that filled
        ONLY such a slot therefore arrives as a black volume — every pixel zero, every mask flag zero —
        carrying its full label row. `slots.FASTMRI_SLOTS` prices the dropped slot; it does not price
        this, and this is the part that would corrupt a run rather than merely shrink it. Half of
        fastMRI is acquired without fat suppression, so the first `mergedf` build put **584 black
        studies** into 5,579, each supervised negative on nine of our twelve findings. Nothing
        downstream excludes them: the trainer joins labels to the cache index by identifier and reads
        whatever row it finds, so the lesson available from those rows is that an empty image means
        absent — on exactly the findings this corpus was brought in to supervise.

        Dropping the LABEL row and not the cache row is deliberate. The array stays as decoded, so the
        merge remains reproducible and a later extended-scheme build can carry those studies properly;
        what changes is only that nothing is asked to learn from them.

        An UNBUILT destination drops nothing rather than refusing: a labels-only run over a cache that
        does not exist yet has no mask to read, and there is nothing black in a table whose arrays have
        not been written.
        """
        mask_path = self.merged.mask_path
        if not mask_path.exists():
            logger.info(f"{self.merged.key} has no mask yet, so no study can be known black — nothing dropped")
            return 0
        mask = np.load(mask_path)
        studies = self._index(self.merged)
        empty = {study for study, flags in zip(studies, mask, strict=True) if not flags.any()}
        table = pd.read_csv(destination)
        kept = table[~table[identifier].astype(str).isin(empty)]
        dropped = len(table) - len(kept)
        kept.to_csv(destination, index=False)
        logger.info(f"dropped {dropped} label rows for studies with no slot at all -> {len(kept)} rows")
        return dropped

    @staticmethod
    def narrow(source: Path, destination: Path, corpus: str, graded: tuple[str, ...]) -> int:
        """Blank one corpus's cells outside the findings it actually GRADES. Returns cells blanked.

        A foreign corpus does not read our vocabulary; it reads its own and is mapped onto ours. fastMRI
        grades ACL, MCL and effusion one-to-one, and for the other nine of our findings it supplies
        NEGATIVES ONLY — its meniscus and cartilage findings are unsided while ours are per-compartment,
        so an absent tear there becomes twelve confident zeros here. Merging it therefore shifts the
        prior toward absence on exactly the findings it cannot locate, which `fastmri_corpus` names as
        the hazard of the whole corpus before any arm ran.

        NaN and not a small weight, because NaN is the mechanism the trainer already has:
        `Targets.load` reads `said = ~np.isnan(value)` and leaves an unsaid cell at weight 0.0. So this
        needs no flag, no new loss term and no trainer change — the control and the masked arm differ by
        one `--labels` path, which is what keeps the comparison a one-variable A/B.

        It blanks CELLS and never drops the ROW: those studies still supervise the three findings their
        readers actually looked at, and still contribute their images to every shared trunk parameter.
        """
        table = pd.read_csv(source)
        named = table[CORPUS].astype(str) if CORPUS in table.columns else pd.Series("", index=table.index)
        rows = named == corpus
        # A finding owns a family of columns — `ACL`, `ACL__conf`, `ACL__verdict` — and they are kept or
        # blanked together: a confidence left behind for a score that is now NaN is a weight for a cell
        # nobody scores, which is the shape of the bug this whole method exists to avoid.
        columns = [
            name for name in table.columns if name.split(SIBLING)[0] not in graded and table[name].dtype.kind == "f"
        ]
        blanked = int(table.loc[rows, columns].notna().to_numpy().sum())
        table.loc[rows, columns] = np.nan
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
        logger.info(f"blanked {blanked} cells over {int(rows.sum())} {corpus} rows, keeping {graded} -> {destination}")
        return blanked

    @classmethod
    def narrowed(cls, destination: Path, corpus: str | None, graded_text: str, out: Path | None) -> Path | None:
        """The optional second table, written from a FINISHED first one so both name the same studies.

        A no-op when no corpus was named, which is what lets both of `build_merged`'s exits call it
        unconditionally — the labels-only path and the full path must not diverge on which tables a run
        produces, and a caller-side `if` in two places is how they would.
        """
        if corpus is None:
            return None
        graded = tuple(name.strip() for name in graded_text.split(",") if name.strip())
        if not graded:
            # Blanking every column would leave rows carrying images and no supervision at all, which is
            # a different experiment from the one this flag exists for, and not one to run silently.
            raise ValueError(
                "a narrowed corpus needs its graded findings: one that grades nothing is a row with no lesson"
            )
        written = out or destination.with_name(f"{destination.stem}_{corpus}only.csv")
        cls.narrow(destination, written, corpus, graded)
        return written

    @staticmethod
    def _tagged(table: pd.DataFrame, name: str) -> pd.DataFrame:
        """One source's table with a `corpus` column, filled with `name` only where it has none."""
        if CORPUS not in table.columns:
            table[CORPUS] = name
            return table
        table[CORPUS] = table[CORPUS].fillna(name).replace("", name)
        return table

    def _mask(self, base_studies: int, extra_studies: int) -> Float[np.ndarray, "studies slots"]:
        """The merged presence mask: base's own rows, then extra's flags at their matched positions."""
        mask = np.zeros((base_studies + extra_studies, self.merged.slots), dtype=np.float32)
        mask[:base_studies] = np.load(self.base.mask_path)
        sources, slots = self.positions()
        mask[base_studies:, slots] = np.load(self.extra.mask_path)[:, sources]
        return mask

    def _copy(
        self,
        images: UInt8[np.ndarray, "studies slots slices height width"],
        start: int,
        source: UInt8[np.ndarray, "source slots slices height width"],
        sources: list[int],
        slots: list[int],
    ) -> None:
        """Block-copy one source's studies into the merged array at `start`, slot by matched slot.

        `sources` and `slots` are the same length and pair up: a source slot the merged axis does not
        name appears in neither, so it is read from nowhere and written nowhere.

        Blocked because the base corpus is 35.8 GB: a whole-array assignment between two memory maps
        materialises it, and the machine has less RAM than that to spare beside a running trainer.
        """
        for first in range(0, len(source), COPY_STUDIES):
            block = source[first : first + COPY_STUDIES]
            images[start + first : start + first + len(block), slots] = block[:, sources]
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
