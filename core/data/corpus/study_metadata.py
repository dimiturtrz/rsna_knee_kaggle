"""One study's acquisition metadata, in the shape every corpus can answer.

THE MERGE KEEPS LABELS AND THROWS AWAY EVERYTHING ELSE, which is the half of "process to a common
format, then merge with maximum metadata" that was never built here. `MergedCache.labels` already
stamps a `corpus` column and fills it correctly — `merged3_tagged_labels.csv` reads canonical 4406,
mrnet 1248, rijeka 736 — so origin is not the gap. Everything an exam was acquired WITH is: plane,
sequence, fat suppression, field strength, spacing, echo and repetition time, age, sex, scanner. Each
corpus holds some of that already and drops all of it at the merge.

ABSENCE IS A VALUE HERE, NOT A HOLE, and that is the whole design. Every axis defaults to `None` and a
corpus overrides only the ones it can answer, so adding an axis never breaks a corpus that has never
heard of it, and a corpus that cannot answer says so rather than being silently missing from the frame.
Rijeka is sagittal PD fat-suppressed by construction and knows its spacing to 0.5 mm but has no patient
age; SKM-TEA ships all forty-four columns; fastMRI knows its plane and its geometry and nothing about
the patient. A schema that demanded every field would be answerable by exactly one of them.

WHY IT IS A FLAT ROW AND NOT A TREE. The consumer is a pandas frame joined to the labels table on the
study identifier, and everything downstream that would use it — stratified sampling, a domain column,
an auxiliary head, a per-corpus score breakdown — wants columns. A nested shape would be re-flattened at
every one of those call sites.

NO FIELD IS DERIVED *HERE*, and a corpus that fills one from a derivation names that derivation at the
point of record. `acquisition.py` recovers fluid sensitivity and fat suppression from TR/TE because the
shipped flags disagree with the pixels on a fifth of the corpus; this module records what a corpus can
say and leaves that recovery where it lives. The distinction that matters is not stated-versus-derived —
Rijeka's in-plane spacing is derived from its matrix and a field of view it does not publish, and it is
still the number its whole geometry runs on — it is whether the reader of a filled cell can find out
which. A field quietly filled by a guess is indistinguishable from one read off a header, which is the
failure the `Fluid_Sensitive` measurement is already a monument to.
"""

import logging
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import ClassVar

import pandas as pd

logger = logging.getLogger(__name__)

IDENTIFIER = "StudyInstanceUID"
"""The join key, spelled as the labels tables spell it. A metadata frame is only ever useful merged onto
labels, so a second spelling would be a silent empty join rather than an error."""

CORPUS = "corpus"
"""The origin column, spelled as `merged_cache.CORPUS` spells it. Duplicated deliberately: importing it
would point this module at a cache builder it otherwise has no reason to know, and the two are checked
against each other by this module's mirror test."""


SET_JOIN = "+"
"""How a flat cell names SEVERAL answers. A row is one STUDY and a study is several series, so `plane`
and `sequence` often have no single value — `Axial+Sagittal` is the true answer at this granularity, and
the per-series truth stays in the routing, which is the artifact that needs it. THE ONE HOME for this
separator: two corpora spell a set this way and a second spelling would split the same column in two."""


@dataclass(frozen=True)
class StudyMetadata:
    """What one study was acquired with, as far as its own corpus can say.

    Only `identifier` and `corpus` are required — a corpus that knows nothing else still produces a
    valid row, and that row is the honest answer rather than an absent one.
    """

    identifier: str
    corpus: str
    plane: str | None = None
    sequence: str | None = None
    fat_suppressed: bool | None = None
    fluid_sensitive: bool | None = None
    field_strength_t: float | None = None
    pixel_spacing_mm: float | None = None
    slice_spacing_mm: float | None = None
    matrix_px: int | None = None
    field_of_view_mm: float | None = None
    echo_time_ms: float | None = None
    repetition_time_ms: float | None = None
    age_years: float | None = None
    sex: str | None = None
    scanner: str | None = None

    KEYS: ClassVar[tuple[str, str]] = (IDENTIFIER, CORPUS)
    """The two columns every row carries. Named so a consumer can split the join keys from the axes
    without re-listing the dataclass."""

    @classmethod
    def axes(cls) -> tuple[str, ...]:
        """Every optional axis, in declaration order — the columns a corpus may or may not answer."""
        return tuple(field.name for field in fields(cls) if field.name not in ("identifier", "corpus"))

    def row(self) -> dict[str, object]:
        """This study as one frame row, with the join key under the name the labels tables use."""
        values = asdict(self)
        values[IDENTIFIER] = values.pop("identifier")
        values[CORPUS] = values.pop("corpus")
        return values

    @classmethod
    def frame(cls, entries: list["StudyMetadata"]) -> pd.DataFrame:
        """Many studies as one frame, with every declared column present even when nothing filled it.

        An empty corpus still contributes its columns. A frame whose columns depend on which studies
        happened to answer is a frame that changes shape between runs, and a downstream join against it
        would fail on the corpus rather than on the schema.
        """
        columns = [IDENTIFIER, CORPUS, *cls.axes()]
        table = pd.DataFrame([entry.row() for entry in entries], columns=columns)
        answered = {column: int(table[column].notna().sum()) for column in cls.axes()}
        logger.info(f"{len(table)} studies, answered: {[k for k, v in answered.items() if v]}")
        return table

    @classmethod
    def write(cls, entries: list["StudyMetadata"], destination: Path) -> None:
        """This corpus's metadata as one CSV, beside the labels table its own builder writes.

        ONE FILE PER CORPUS, WHICH IS THE HALF OF THE FORMAT THAT WAS MISSING. Every corpus here already
        writes its own labels table and `build_merged` concatenates two of those; metadata now travels the
        same road, so a corpus is processed on its own and merged afterwards rather than being knowable
        only through whatever survived a concatenation.
        """
        table = cls.frame(entries)
        destination.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination, index=False)
        logger.info(f"{len(table)} rows -> {destination}")

    @classmethod
    def merged(cls, sources: list[Path]) -> pd.DataFrame:
        """Several corpora's metadata as one frame, every source coerced to the DECLARED columns.

        Coerced rather than concatenated raw, and that is the whole mechanism. A corpus written before an
        axis existed has no column for it and pandas would produce a frame whose shape depends on which
        files happened to be passed; reindexing onto `axes()` gives the same columns every time and fills
        the ones a corpus never wrote with absence — which is what a corpus that cannot answer says
        anyway. A column a source carries and this schema does not is DROPPED, so an axis reaches a
        consumer only by being declared here.
        """
        columns = [IDENTIFIER, CORPUS, *cls.axes()]
        tables = [pd.read_csv(source).reindex(columns=columns) for source in sources]
        merged = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=columns)
        logger.info(f"{len(merged)} studies from {len(sources)} corpora: {merged[CORPUS].value_counts().to_dict()}")
        return merged

    @classmethod
    def merge_written(cls, directory: Path, sources: tuple[str, ...], merged_name: str) -> None:
        """Concatenate the named corpora's own metadata CSVs into the destination's, coerced to one schema.

        A SOURCE THAT WROTE NO METADATA IS SKIPPED WITH A COUNT, not an error. The canonical corpus has no
        `metadata()` yet, so refusing here would make the union labels table — which does not need metadata
        to be correct — unbuildable for the one merge that matters most. The log line says how many corpora
        actually contributed, which is the number a reader of the frame needs.
        """
        paths = [directory / f"{name}_metadata.csv" for name in sources]
        present = [path for path in paths if path.exists()]
        for path in paths:
            if path not in present:
                logger.warning(f"no metadata for {path.stem.removesuffix('_metadata')} — {path} is absent")
        if not present:
            logger.warning("METADATA SKIPPED — no source wrote one")
            return
        destination = directory / f"{merged_name}_metadata.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        cls.merged(present).to_csv(destination, index=False)
        logger.info(f"metadata from {len(present)} of {len(paths)} corpora -> {destination}")

    @classmethod
    def joined(cls, labels: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
        """Labels with their metadata alongside, left-joined so no labelled study is ever dropped.

        A study with no metadata row keeps its labels and reads absent on every axis, which is the same
        thing a corpus that cannot answer an axis says. A study with metadata and no labels is not a
        training row and does not appear.
        """
        overlap = (set(labels.columns) & set(metadata.columns)) - {IDENTIFIER}
        return labels.merge(metadata.drop(columns=sorted(overlap)), on=IDENTIFIER, how="left")
