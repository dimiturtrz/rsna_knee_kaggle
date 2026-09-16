"""The program six corpus audits were each written as a copy of, written once.

WHAT THEY SHARE, MEASURED 2026-09-16. `band_coverage`, `crop_coverage`, `crop_alignment`,
`crop_laterality`, `slot_geometry` and `native_depth` each open a pool over a spread sample of studies,
route every study through `SlotPipeline`, catch the same four exceptions, log the same "did not decode"
line, and drop the study. The only thing that differed between the six was ONE `SlotReader` keyword —
`band=FULL_BAND`, `crop_mm=UNCROPPED`, or nothing — and two of them had additionally re-spelled
`CorpusSplit.sample`'s every-k-th slice by hand, under a comment claiming it matched the tool that used
the method. That is the drift `CorpusSample` was written to end, re-introduced by copying.

SO THE AXIS THAT VARIES IS A CONFIG, and it is `Reading`. What does NOT vary is the routing, the draw
and the refusal, and those live here. What stays in each module is the REDUCER — tissue mass along the
axis, off-centre offsets, patient-space frames, acquired depth — which is the question the module exists
to ask and is not duplicated anywhere.

NOT A FRAMEWORK, for `core.cli`'s reason. There is no registry mapping a name to a reducer and no
declarative audit file, because a reducer is a hundred lines of numpy that only Python can spell; a TOML
naming one would buy sweepability where nothing sweeps and cost a string-to-class lookup. The layered
TOML stack earns its keep for TRAINING, where a value is swept, stamped into a checkpoint and priced by
the ledger months later. An audit prints and exits.

WHICH IS WHY `Reading` PRINTS ITSELF. These tools reported coverage fractions for a year without ever
stating the geometry they were read at, so a pasted number could not be re-derived — the same defect as
a score quoted without its cut. `Reading.line` goes under the denominator sentence, where the rest of
the reading's terms already are.
"""

import argparse
import logging
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core import cli
from core.data.corpus.acquisition import Acquisition
from core.data.corpus.crop_centre import CROP_MM
from core.data.corpus.slot_reader import IMAGE_SIZE, SPREAD, Band, SlotReader
from core.data.corpus.slots import Routing
from core.data.corpus.study import SlotPipeline, Study
from core.data.corpus.study_cache import CorpusSample, CorpusSplit

logger = logging.getLogger(__name__)

SAMPLE = 400
"""Studies drawn by default, so the six audits describe the same corpus slice and compose."""

WORKERS = 20
"""Decode processes. Header-only audits override it downward; nothing here is memory-bound."""

CHUNK = 4
"""Studies handed to a worker at a time — a decode is seconds, so the pool is not dispatch-bound."""

HEADERS_ONLY = 0
"""The slice count an audit asks for when it reads no pixels at all: routing and headers, then stop."""

UNCROPPED = 1.0e9
"""A crop wider than any acquisition, so `SlotReader.cropped` decodes and hands back the FULL frame.

Three of the six audits read at it, for the same reason each stated separately: the denominator is the
tissue in the WHOLE acquisition, and a stack already cropped cannot supply it — whatever the crop
discarded is exactly what is being counted. It lives here rather than in `crop_coverage` because
`Reading.line` has to RECOGNISE it: printed as a number it reads `crop 1e+09 mm`, which is a geometry no
scanner produced and no reader would re-derive.
"""

FAILURES = (OSError, ValueError, RuntimeError, IndexError)
"""What a corpus of 4,407 scanned studies raises at a reader, written once.

Each of the six audits caught these four and each explained the catch in its own words. It is a TUPLE
rather than a bare `except Exception` for ruff's BLE001 reason and for a better one: a `KeyError` out of
the routing scheme is this project's bug and must crash, while a truncated DICOM is the corpus being
itself.
"""


@dataclass(frozen=True)
class Reading:
    """The geometry an audit reads at — `SlotReader`'s constructor as PLAIN DATA.

    A `SlotReader` is not data: it carries a `FileMemo` and a bound spacing strategy, and the pool would
    have to pickle both. Every audit therefore built its reader inside the worker, and the config was
    expressed as the literal keyword it passed there — unreachable from the report, which is why none of
    them ever printed it.
    """

    crop_mm: float = CROP_MM
    image_size: int = IMAGE_SIZE
    band: Band = field(default_factory=Band)
    spacing: str = SPREAD

    def reader(self) -> SlotReader:
        """This geometry as the reader itself, built worker-side where it does not have to travel."""
        return SlotReader(crop_mm=self.crop_mm, image_size=self.image_size, band=self.band, spacing=self.spacing)

    def line(self) -> str:
        """What a number in this report was read at, in the wording the denominator sentence uses."""
        extent = "uncropped" if self.crop_mm >= UNCROPPED else f"crop {self.crop_mm:g} mm"
        return (
            f"  read at: {extent}, band {self.band.low:g}-{self.band.high:g}, "
            f"{self.image_size} px, {self.spacing} spacing"
        )


@dataclass(frozen=True)
class AuditRequest:
    """One study's job, as the pool receives it: flat, and carrying its own geometry.

    Flat because a process pool pickles its arguments, and carrying the `Reading` because a worker that
    read the geometry from a module constant could not be asked the same question twice with two
    configs — which is exactly what `crop_coverage` wants.
    """

    identifier: str
    source: str
    planes: dict[str, str]
    reading: Reading = field(default_factory=Reading)
    slices: int = HEADERS_ONLY

    def routed(self) -> "RoutedStudy":
        """Every slot this study routes to, or an EMPTY reading — never a zero, never a raise.

        A study that will not decode is SILENT about whatever is being measured. Counting it as zero
        discarded tissue, or as a zero-slice acquisition, would charge the corpus for the DICOM library:
        each of the six audits said so in its own words and each dropped the study, which is the one
        behaviour that must not be re-decided per module.
        """
        reader = self.reading.reader()
        pipeline = SlotPipeline(scheme=Routing().built(), reader=reader)
        try:
            study = Study.from_directory(Path(self.source) / self.identifier, self.planes, pipeline)
            routed = study.pipeline.route(study.acquisitions, study.planes)
        except FAILURES as failure:
            logger.info(f"  {self.identifier} did not decode: {type(failure).__name__}: {failure}")
            return RoutedStudy(self.identifier, {}, reader, None)
        return RoutedStudy(self.identifier, routed, reader, study)


@dataclass(frozen=True)
class RoutedStudy:
    """One study's routed slots and the reader that produced them — what a reducer is handed.

    The READER travels with the slots because decoding is the reducer's call, not this module's: three
    of the six audits want pixels (`reader.cropped`), two want headers alone, and one wants the `Study`
    itself for its laterality. Handing back all three and deciding nothing is what lets one routing
    serve six questions.
    """

    identifier: str
    slots: dict[str, Acquisition]
    reader: SlotReader
    study: Study | None

    def only(self, slots: tuple[str, ...]) -> "RoutedStudy":
        """The same routing narrowed to named slots, so a reducer DECODES only what it can measure.

        `crop_laterality` declines sagittal series — medial-lateral is through-plane there — and it
        declined them before decoding, which is most of its runtime. Narrowing after the decode would
        also widen the drop: a sagittal series that will not read would take the study's coronal
        measurement with it.
        """
        return replace(self, slots={name: one for name, one in self.slots.items() if name in slots})

    def side(self) -> str | None:
        """Which knee this is, or None where it cannot be told — including where ASKING raises.

        The laterality rule reads headers, so it raises on the same truncated files routing does, and it
        sat inside `crop_laterality`'s single try. None is what that module already does with an
        unresolved side: decline the series rather than enter it with a coin-flip sign.
        """
        if self.study is None:
            return None
        try:
            return self.study.side()
        except FAILURES as failure:
            logger.info(f"  {self.identifier} has no side: {type(failure).__name__}: {failure}")
            return None

    def decoded(self, slices: int) -> list[tuple[str, Acquisition, NDArray[np.float32] | None]]:
        """Every routed slot with its cropped stack, or NOTHING if any series in the study fails.

        Whole-study rather than per-series, which is what all three pixel audits did before this module
        existed: their decode sat inside the same `try` as the routing. Dropping only the failing series
        would be defensible and is a DIFFERENT measurement — the denominators in every report those
        tools have produced are whole studies — so it stays a deliberate change rather than one made by
        moving code.
        """
        try:
            return [(slot, one, self.reader.cropped(one, slices)) for slot, one in self.slots.items()]
        except FAILURES as failure:
            logger.info(f"  {self.identifier} did not decode: {type(failure).__name__}: {failure}")
            return []


@dataclass(frozen=True)
class CorpusAudit:
    """The draw, the pool and the refusal every corpus audit performs identically.

    Constructed by a module beside its reducer, so the file still reads as one command: what geometry it
    reads at, what unit its denominator counts, and how many studies it draws.
    """

    reading: Reading = field(default_factory=Reading)
    unit: str = "series"
    slices: int = HEADERS_ONLY
    sample: int = SAMPLE
    workers: int = WORKERS

    def flags(self, command: cli.Command) -> None:
        """The two arguments all six spelled out: how many studies, and how many processes."""
        command.parser.add_argument("--sample", type=int, default=self.sample, help="studies to draw")
        command.parser.add_argument("--workers", type=int, default=self.workers, help="decode processes")

    def drawn(
        self,
        command: cli.Command,
        arguments: argparse.Namespace,
        reading: Reading | None = None,
        studies: Path | None = None,
    ) -> tuple[CorpusSample, list[AuditRequest]]:
        """The every-k-th sample and its jobs, through `CorpusSplit` — the ONE home for the draw.

        `reading` overrides the declared geometry for a module whose own flags change it, so a band or a
        crop given on the command line reaches the workers instead of being silently dropped.

        `studies` names a file of identifiers to measure INSTEAD of a sample, which is what lets a
        per-study reading be joined against a holdout record — a join needs the same studies on both
        sides, not a sample of them. Both draws are `CorpusSplit`'s; neither is spelled here.
        """
        split = CorpusSplit(command.rooted(arguments).raw())
        planes = split.planes()
        sample = split.named(studies) if studies else split.sample(arguments.sample)
        chosen = reading or self.reading
        requests = [
            AuditRequest(identifier=name, source=str(split.source), planes=planes, reading=chosen, slices=self.slices)
            for name in sample.chosen
        ]
        return sample, requests

    def with_slices(self, slices: int) -> "CorpusAudit":
        """The same audit reading `slices` per series — the one knob a module's own flag moves."""
        return replace(self, slices=slices)

    def mapped[T](
        self, worker: Callable[[AuditRequest], list[T]], requests: list[AuditRequest], workers: int
    ) -> list[T]:
        """Every request through the pool, flattened — a worker returning nothing contributes nothing."""
        measured: list[T] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for got in pool.map(worker, requests, chunksize=CHUNK):
                measured.extend(got)
        return measured

    def reported(self, command: cli.Command, lines: list[str], sample: CorpusSample, measured: int) -> None:
        """The report with its geometry under the denominator sentence, then the empty-decode refusal.

        The `Reading` goes SECOND rather than first because the two lines answer one question together —
        what was measured, and at what — and the denominator is the half a reader is already looking for.
        """
        command.report([lines[0], self.reading.line(), *lines[1:]] if lines else [self.reading.line()])
        sample.denominator(measured, self.unit).refuse_empty()
