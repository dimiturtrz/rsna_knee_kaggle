"""The audit harness, exercised where it decides rather than where it decodes.

A unit test may touch nothing it did not create, so no test here reaches a corpus. That is not a
limitation for this module: the behaviour it centralises is the REFUSAL — a study that will not decode
is silent rather than zero — and an absent directory under `tmp_path` produces exactly that failure,
which is why all six audits already tested their workers that way.
"""

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import override

from core import cli
from core.data.corpus.audit import UNCROPPED, AuditRequest, CorpusAudit, Reading, RoutedStudy
from core.data.corpus.crop_centre import CROP_MM
from core.data.corpus.slot_reader import Band
from core.data.corpus.study_cache import CorpusSample
from core.paths import DataRoot


class _Command(cli.Command):
    """The real `cli.Command` with its two outside edges redirected: the data root, and the sink.

    A SUBCLASS rather than a stand-in, so the harness is exercised through the type it declares. Only
    `rooted` is overridden, because it resolves `paths.yaml` — which names the machine's corpus, and a
    unit test may touch nothing it did not create.
    """

    def __init__(self, root: Path | None = None) -> None:
        super().__init__(__doc__, logging.getLogger(__name__))
        self._root = root
        self.reported: list[str] = []

    @override
    def rooted(self, arguments: argparse.Namespace) -> DataRoot:
        assert self._root is not None, f"this command was given no root, for {arguments}"
        return DataRoot(self._root)

    @override
    def report(self, lines: Iterable[str]) -> None:
        self.reported = list(lines)


def _routed(tmp_path: Path) -> RoutedStudy:
    """The routing of a study that is not there — the shape every audit's failure path produces."""
    return AuditRequest(identifier="absent", source=str(tmp_path), planes={}).routed()


def test_reader() -> None:
    """The geometry reaches the reader it builds, which is the whole reason it travels as data."""
    reader = Reading(crop_mm=UNCROPPED, band=Band(low=0.0, high=1.0)).reader()

    assert reader.crop_mm == UNCROPPED
    assert (reader.band.low, reader.band.high) == (0.0, 1.0)


def test_line() -> None:
    """What a number was read at, with the uncropped sentinel named rather than printed as 1e+09."""
    assert Reading().line().strip().startswith(f"read at: crop {CROP_MM:g} mm")
    assert "uncropped" in Reading(crop_mm=UNCROPPED).line()
    assert "band 0-1" in Reading(band=Band(low=0.0, high=1.0)).line()


def test_routed(tmp_path: Path) -> None:
    """A study that will not open routes to NOTHING rather than raising into the pool."""
    routed = _routed(tmp_path)

    assert routed.slots == {}
    assert routed.study is None
    assert routed.identifier == "absent"


def test_only(tmp_path: Path) -> None:
    """Narrowing keeps the named slots and nothing else, and survives an empty routing."""
    routed = _routed(tmp_path)

    assert _routed(tmp_path).only(("COR_T1",)).slots == {}
    assert routed.only(()).reader is routed.reader, "narrowing the slots must not rebuild the reader"


def test_side(tmp_path: Path) -> None:
    """A study that never opened has no side, and says so rather than guessing one."""
    assert _routed(tmp_path).side() is None


def test_decoded(tmp_path: Path) -> None:
    """Nothing routed decodes to nothing — the silence every reducer filters on."""
    assert _routed(tmp_path).decoded(12) == []


def test_flags() -> None:
    """The two flags every audit spelled by hand, defaulted from the audit's own declaration."""
    command = _Command()
    CorpusAudit(sample=7, workers=3).flags(command)

    parsed = command.parser.parse_args([])

    assert (parsed.sample, parsed.workers) == (7, 3)


def test_with_slices() -> None:
    """The one knob a module's own flag moves, and it moves a COPY."""
    audit = CorpusAudit(slices=4)
    deeper = audit.with_slices(24)

    assert (deeper.slices, audit.slices) == (24, 4)
    assert deeper.reading == audit.reading


def _rooted(tmp_path: Path) -> _Command:
    """A command whose root holds the two shipped tables: four studies, one series each."""
    raw = DataRoot(tmp_path).raw()
    raw.mkdir(parents=True)
    identifiers = [f"study{index}" for index in range(4)]
    (raw / "train.csv").write_text("StudyInstanceUID\n" + "\n".join(identifiers) + "\n", encoding="utf-8")
    (raw / "train_series.csv").write_text(
        "StudyInstanceUID,SeriesInstanceUID,Anatomical_Plane\n"
        + "\n".join(f"{name},{name}-1,Sagittal" for name in identifiers)
        + "\n",
        encoding="utf-8",
    )
    return _Command(tmp_path)


def test_drawn(tmp_path: Path) -> None:
    """The draw is `CorpusSplit`'s, and every request carries the audit's geometry and depth."""
    audit = CorpusAudit(reading=Reading(crop_mm=UNCROPPED), slices=12)
    command = _rooted(tmp_path)

    sample, requests = audit.drawn(command, argparse.Namespace(sample=2))

    assert isinstance(sample, CorpusSample)
    assert len(requests) == 2, "every k-th of four studies, taken twice"
    assert all(request.slices == 12 for request in requests)
    assert all(request.reading.crop_mm == UNCROPPED for request in requests)
    assert all(request.planes for request in requests), "the series table's planes reach the worker"


def test_mapped(tmp_path: Path) -> None:
    """The pool flattens: a worker returning nothing contributes nothing, not an empty row."""
    requests = [AuditRequest(identifier=name, source=str(tmp_path), planes={}) for name in ("a", "b")]

    # One worker, because spawning is what fails on a loaded machine (`StudyCache._decode_all`), and
    # the flattening this asserts is the parent's arithmetic either way.
    assert CorpusAudit().mapped(_nothing, requests, workers=1) == []
    assert CorpusAudit().mapped(_two, requests, workers=1) == ["a", "a", "b", "b"]


def test_reported() -> None:
    """The geometry goes SECOND, under the denominator sentence the reader is already looking for."""
    audit = CorpusAudit(reading=Reading(crop_mm=UNCROPPED))
    command = _Command()

    audit.reported(command, ["3 series measured", "a row"], CorpusSample(["a"], 4407), measured=3)

    assert command.reported[0] == "3 series measured"
    assert "uncropped" in command.reported[1]
    assert command.reported[2] == "a row"


def _nothing(request: AuditRequest) -> list[str]:
    """A worker that measures nothing, at module scope so the pool can pickle it."""
    assert request.identifier
    return []


def _two(request: AuditRequest) -> list[str]:
    """A worker returning two rows, so flattening is visible rather than inferred from a length."""
    return [request.identifier, request.identifier]
