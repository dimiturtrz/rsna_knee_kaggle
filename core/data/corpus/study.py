"""One study assembled into the bag of slot images the model reads, plus its presence mask."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from jaxtyping import Float, UInt8
from torch import Tensor

from core.data.corpus.acquisition import Acquisition
from core.data.corpus.laterality import Laterality
from core.data.corpus.slot_reader import SlotReader
from core.data.corpus.slots import CANONICAL_SCHEME, Slot, SlotScheme
from core.data.corpus.study_metadata import SET_JOIN, StudyMetadata


@dataclass(frozen=True)
class SlotPipeline:
    """The three collaborators that turn a study's series into slot images.

    Bundled rather than passed one by one because they are only ever varied together: a different
    reader implies a different cache, which implies the routing and laterality it was decoded under.
    """

    scheme: SlotScheme = field(default_factory=SlotScheme)
    reader: SlotReader = field(default_factory=SlotReader)
    laterality: Laterality = field(default_factory=Laterality)

    @property
    def image_size(self) -> int:
        """The resolution the reader will produce, which the caller sizes its zeros to."""
        return self.reader.image_size

    def side_of(self, acquisitions: list[Acquisition]) -> str | None:
        """Which knee the study is, by the laterality rule this pipeline was configured with."""
        return self.laterality.of_study(acquisitions)

    def route(self, acquisitions: list[Acquisition], planes: dict[str, str]) -> dict[str, Acquisition]:
        """One acquisition per slot under this pipeline's slot predicate."""
        return self.scheme.assign(acquisitions, planes)

    def slot_sizes(
        self, acquisition: Acquisition, slices: int, slot: Slot, side: str | None, sizes: "tuple[int, ...]"
    ) -> "dict[int, UInt8[Tensor, 's h w']] | None":
        """One slot read ONCE and delivered at every size asked for.

        The decode and the physical crop do not depend on the grid; only the final resize does. So a
        submission whose members read 392, 336 and 224 px can pay for one read instead of three — which
        is what `predict_groups` currently costs, because it decodes the split once per pixel group.

        Each size is still a single resize of the same cropped stack, so what a member is handed here is
        BYTE-IDENTICAL to what a native read at its own size would produce; no fingerprint and no
        witness moves. Laterality is normalised per size for the same reason it is normalised at all: a
        flip is a property of the grid it is applied to.
        """
        stack = self.reader.cropped(acquisition, slices)
        if stack is None:
            return None
        stack = self.reader.narrowed(stack, acquisition.pixel_spacing, slot.crop_mm)
        return {size: self.laterality.normalise(self.reader.sized(stack, size), slot.plane, side) for size in sizes}


class Study:
    """A study's series, routed into slots, ordered, laterality-normalised and decoded once."""

    def __init__(
        self,
        identifier: str,
        acquisitions: list[Acquisition],
        planes: dict[str, str],
        pipeline: SlotPipeline | None = None,
    ) -> None:
        self.identifier = identifier
        self.acquisitions = acquisitions
        self.planes = planes
        self.pipeline = pipeline or SlotPipeline()

    @classmethod
    def from_directory(cls, directory: Path, planes: dict[str, str], pipeline: SlotPipeline | None = None) -> "Study":
        """Probe every series directory under a study, reading one header each."""
        acquisitions = [
            Acquisition.probe(directory.name, series.name, series)
            for series in sorted(directory.iterdir())
            if series.is_dir()
        ]
        return cls(directory.name, acquisitions, planes, pipeline)

    def side(self) -> str | None:
        """Which knee this is: the tag where present, geometry where not, None inside the dead zone."""
        return self.pipeline.side_of(self.acquisitions)

    def metadata(self) -> StudyMetadata:
        """What this study was acquired with, from its series' HEADERS rather than the shipped columns.

        THE SHIPPED FLAGS ARE NOT WRITTEN HERE, deliberately. `slots.py` measured them: shipped
        `Fat_Suppression` matches the header-recovered value on 99.3% of series and shipped
        `Fluid_Sensitive` on only 78.7%, one axis wearing two names. Putting the 78.7% column into a
        schema beside axes read off headers is the exact failure this module's docstring warns about,
        so the recovered value is the one recorded and the shipped one is not recorded at all.

        PLANE AND SEQUENCE RECORD THE SET, joined as MRNet's do, because a study is several series and
        this schema is one row per study. A study shipping sagittal PD and axial T2 reads
        `Axial+Sagittal` / `PD+T2`, which is the true answer at this granularity; the per-series truth
        stays in the routing, which is the artifact that needs it.

        EVERY OTHER AXIS IS FILLED ONLY ON UNANIMITY. Fat suppression, fluid sensitivity and the two
        spacings are properties of a SERIES, and a study whose series disagree has no single value for
        them — so it says absent rather than picking one. Absence is a value here. Spacing will
        therefore be absent on most multi-plane studies, which is honest: there is no one number.

        THE SCANNER AND THE FIELD STRENGTH GO THROUGH THE SAME GATE, and unlike spacing they are
        expected to survive it: a study is normally one visit on one machine, so its series agree. A
        study whose series disagree was acquired on two scanners and has no single vendor, and reading
        absent there is the answer rather than a gap. Recorded at all because they were declared on
        `StudyMetadata` and never filled by any corpus in the tree, which left the acquisition-shift
        axis looking unavailable when it was only unread (knee-m4jkz).
        """
        return StudyMetadata(
            identifier=self.identifier,
            corpus=CANONICAL_SCHEME,
            plane=self._joined(self.planes.get(acquisition.series) for acquisition in self.acquisitions),
            sequence=self._joined(acquisition.weighting for acquisition in self.acquisitions),
            fat_suppressed=self._agreed(acquisition.fat_suppressed for acquisition in self.acquisitions),
            fluid_sensitive=self._agreed(acquisition.fluid_sensitive for acquisition in self.acquisitions),
            pixel_spacing_mm=self._agreed(acquisition.pixel_spacing for acquisition in self.acquisitions),
            slice_spacing_mm=self._agreed(acquisition.slice_spacing for acquisition in self.acquisitions),
            scanner=self._agreed(acquisition.scanner for acquisition in self.acquisitions),
            field_strength_t=self._agreed(acquisition.field_strength for acquisition in self.acquisitions),
        )

    @staticmethod
    def _joined(values: "Iterable[str | None]") -> str | None:
        """The distinct answers as one flat cell, or absence where no series answered."""
        distinct = sorted({value for value in values if value})
        return SET_JOIN.join(distinct) if distinct else None

    @staticmethod
    def _agreed[T](values: "Iterable[T | None]") -> T | None:
        """The one value every series that answered agrees on, or absence where they do not."""
        distinct = {value for value in values if value is not None}
        return distinct.pop() if len(distinct) == 1 else None

    def bag(self, slices: int) -> tuple[UInt8[Tensor, "n s h w"], Float[Tensor, " n"]]:
        """The per-slot image stacks and the presence mask, at this pipeline's own size."""
        return self.bags(slices, (self.pipeline.image_size,))[self.pipeline.image_size]

    def bags(
        self, slices: int, sizes: "tuple[int, ...]"
    ) -> "dict[int, tuple[UInt8[Tensor, 'n s h w'], Float[Tensor, ' n']]]":
        """The per-slot stacks and presence mask at EVERY requested size, from one read per series.

        A slot that is empty, or whose series will not decode at all, contributes zeros AND a zero in
        the mask — the head's softmax then renormalises over the slots the study actually has, rather
        than attending to a black image that the mask claims was acquired. The mask is a property of the
        study rather than of the grid, so every size gets the same one.
        """
        chosen = self.pipeline.route(self.acquisitions, self.planes)
        side = self.side()
        images: dict[int, list[Tensor]] = {size: [] for size in sizes}
        mask: list[float] = []
        # THE PIPELINE'S slots, not the module's. `route` already honours the scheme, so iterating the
        # canonical six here made the two halves of this method disagree: a nine-slot scheme routed
        # nine series and emitted six bags. The cache is allocated at the scheme's width, so every
        # `images[position] = bag` would raise a shape error, land in decode_one's except branch, and be
        # reported as an absent study — 4,407 of them, leaving a fully zero cache whose mask says
        # nothing was acquired. A build that "finished" and a trainer that reads zeros.
        for slot in self.pipeline.scheme.slots:
            acquisition = chosen.get(slot.name)
            read = None if acquisition is None else self.pipeline.slot_sizes(acquisition, slices, slot, side, sizes)
            for size in sizes:
                images[size].append(torch.zeros(slices, size, size, dtype=torch.uint8) if read is None else read[size])
            mask.append(0.0 if read is None else 1.0)
        presence = torch.tensor(mask, dtype=torch.float32)
        return {size: (torch.stack(stacks), presence) for size, stacks in images.items()}
