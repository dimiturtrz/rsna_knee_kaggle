"""One study assembled into the bag of slot images the model reads, plus its presence mask."""

from dataclasses import dataclass, field
from pathlib import Path

import torch
from jaxtyping import Float, UInt8
from torch import Tensor

from kneemri.acquisition import Acquisition
from kneemri.laterality import Laterality
from kneemri.slot_reader import SlotReader
from kneemri.slots import SlotScheme


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

    def slot_images(
        self, acquisition: Acquisition, slices: int, plane: str, side: str | None
    ) -> UInt8[Tensor, "s h w"] | None:
        """Read one slot and normalise its laterality — the two steps are never useful apart."""
        stack = self.reader.read(acquisition, slices)
        return None if stack is None else self.laterality.normalise(stack, plane, side)


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

    def bag(self, slices: int) -> tuple[UInt8[Tensor, "n s h w"], Float[Tensor, " n"]]:
        """The per-slot image stacks and the presence mask, in the fixed slot order.

        A slot that is empty, or whose series will not decode at all, contributes zeros AND a zero in
        the mask — the head's softmax then renormalises over the slots the study actually has, rather
        than attending to a black image that the mask claims was acquired.
        """
        chosen = self.pipeline.route(self.acquisitions, self.planes)
        side = self.side()
        size = self.pipeline.image_size
        images: list[Tensor] = []
        mask: list[float] = []
        # THE PIPELINE'S slots, not the module's. `route` already honours the scheme, so iterating the
        # canonical six here made the two halves of this method disagree: a nine-slot scheme routed
        # nine series and emitted six bags. The cache is allocated at the scheme's width, so every
        # `images[position] = bag` would raise a shape error, land in decode_one's except branch, and be
        # reported as an absent study — 4,407 of them, leaving a fully zero cache whose mask says
        # nothing was acquired. A build that "finished" and a trainer that reads zeros.
        for slot in self.pipeline.scheme.slots:
            acquisition = chosen.get(slot.name)
            stack = None if acquisition is None else self.pipeline.slot_images(acquisition, slices, slot.plane, side)
            if stack is None:
                images.append(torch.zeros(slices, size, size, dtype=torch.uint8))
                mask.append(0.0)
                continue
            images.append(stack)
            mask.append(1.0)
        return torch.stack(images), torch.tensor(mask, dtype=torch.float32)
