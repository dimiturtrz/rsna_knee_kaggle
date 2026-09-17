"""Blank the patch a member leans on, drawn from that member's own occlusion map for that study.

The map is a fact about the MODEL, not about the anatomy, which is what makes this usable while the
same maps are too centre-piled to locate a pathology: an augmentation only needs to know where the
evidence the head currently uses sits, and forcing the head off that evidence is the whole mechanism.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from numpy.typing import NDArray
from torch import Tensor

logger = logging.getLogger(__name__)

STUDIES = "studies"
MAPS = "maps"


class PatchErasing:
    """One rate of single-cell erasure, each study's cell drawn in proportion to its own patch map."""

    def __init__(self, importance: Float[NDArray[np.float64], "studies g g"], rate: float) -> None:
        self.importance = importance
        self.rate = rate

    @classmethod
    def loaded(cls, maps: Path, identifiers: list[str], rate: float) -> "PatchErasing":
        """Read a patch sweep and align it to THIS corpus's study order.

        The join is by identifier because a sweep records one split in that split's order; a trainer
        whose corpus was cached differently would otherwise erase study A's cell out of study B.

        Labels are averaged into one map per study. A per-label draw would need the study's labels at
        augmentation time, and erasing the cell of a label the study does not have is not the
        augmentation this is: what is being removed is the evidence the head reads from THIS image.

        A study the sweep does not name is given the mean map, so a partial sweep degrades toward
        erasing a typical cell instead of dropping the study out of the augmentation. A sweep naming
        NONE of them is refused instead: that is a key mismatch rather than partial coverage, and
        degrading it leaves every study on the mean map — a generic cutout wearing this treatment's
        name, which reads as a measured null on the axis nobody actually tested.
        """
        held = np.load(maps)
        names = [str(name) for name in held[STUDIES]]
        per_study = held[MAPS].mean(axis=1).astype(np.float64)
        mean: Float[NDArray[np.float64], "g g"] = per_study.mean(axis=0)
        recorded: dict[str, Float[NDArray[np.float64], "g g"]] = dict(zip(names, per_study, strict=True))
        missing = [name for name in identifiers if name not in recorded]
        if len(missing) == len(identifiers):
            joined = f"{maps} names none of the {len(identifiers)} studies; it holds {names[:3]} - a key mismatch"
            raise ValueError(joined)
        if missing:
            absent = "%d of %d studies are absent from %s; they erase at the mean map"
            logger.warning(absent, len(missing), len(identifiers), maps)
        return cls(np.stack([recorded.get(name, mean) for name in identifiers]), rate)

    def erased(
        self,
        images: UInt8[Tensor, "b n g h w"],
        positions: Int[NDArray[np.int64], " b"],
        generator: torch.Generator,
    ) -> UInt8[Tensor, "b n g h w"]:
        """Blank one patch of each selected study, across every slot and slice of it.

        Across the whole study rather than per slice: the map was measured by occluding a cell in every
        slice at once, so erasing one slice would remove a fraction of what the map priced.
        """
        if self.rate <= 0:
            return images
        weights = torch.as_tensor(self.importance[positions], dtype=torch.float32, device=images.device)
        flat = weights.flatten(start_dim=1)
        drawn = torch.multinomial(flat.clamp_min(torch.finfo(flat.dtype).tiny), 1, generator=generator).squeeze(1)
        selected = torch.rand(len(positions), generator=generator, device=images.device) < self.rate
        grid = self.importance.shape[-1]
        edges = torch.from_numpy(np.linspace(0, images.shape[-1], grid + 1).astype(np.int64)).to(images.device)
        span = torch.arange(images.shape[-1], device=images.device)
        rows, columns = drawn // grid, drawn % grid
        within_rows = (span >= edges[rows].unsqueeze(1)) & (span < edges[rows + 1].unsqueeze(1))
        within_columns = (span >= edges[columns].unsqueeze(1)) & (span < edges[columns + 1].unsqueeze(1))
        blanked = within_rows.unsqueeze(2) & within_columns.unsqueeze(1) & selected.view(-1, 1, 1)
        return images * (~blanked).view(len(positions), 1, 1, *blanked.shape[1:]).to(images.dtype)
