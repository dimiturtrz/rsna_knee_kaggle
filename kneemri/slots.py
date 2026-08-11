"""Routing a study's series into the six acquisition slots the head attends over.

A knee is read in three planes because the structures run in different directions — cruciates obliquely
and best sagittally, collateral ligaments and the meniscal body coronally, patellar cartilage and the
retinacula axially. Crossing plane with the two acquisition axes gives six slots chosen so each of the
twelve findings has at least one sequence that shows it well.

Measured here on 150 random series: the shipped `Fat_Suppression` column matches recovered fat
suppression on 99.3% of them, but the shipped `Fluid_Sensitive` column matches recovered fluid
sensitivity on only 78.7% — and the non-fat-suppressed pool is 40 T1 against 32 PD/T2. Routing on the
shipped flags alone would therefore put fluid-weighted series in the T1 slots for a fifth of the corpus.
"""

from dataclasses import dataclass
from enum import StrEnum

from kneemri.acquisition import Acquisition


class Plane(StrEnum):
    """The three acquisition planes, spelled once.

    A StrEnum rather than a bare string because these are matched by VALUE in two places that fail
    quietly when the spelling is wrong: routing a series into a slot, and the anatomy table, where a
    mistyped plane produces a flat prior row for that finding rather than an error. The values are the
    ones `train_series.csv` carries in `Anatomical_Plane`, so comparison against raw ingest data still
    works without converting it.
    """

    SAGITTAL = "Sagittal"
    CORONAL = "Coronal"
    AXIAL = "Axial"


@dataclass(frozen=True)
class Slot:
    """One slot's predicate: a plane, a weighting axis, and a fat-suppression state."""

    name: str
    plane: str
    fluid_sensitive: bool
    fat_suppressed: bool

    def accepts(self, acquisition: Acquisition, plane: str) -> bool:
        """Whether this acquisition satisfies the slot's predicate exactly."""
        return (
            plane == self.plane
            and acquisition.fat_suppressed == self.fat_suppressed
            and acquisition.fluid_sensitive == self.fluid_sensitive
        )


SLOTS: tuple[Slot, ...] = (
    Slot("SAG_FLUID_FS", Plane.SAGITTAL, fluid_sensitive=True, fat_suppressed=True),
    Slot("COR_FLUID_FS", Plane.CORONAL, fluid_sensitive=True, fat_suppressed=True),
    Slot("AX_FLUID_FS", Plane.AXIAL, fluid_sensitive=True, fat_suppressed=True),
    Slot("SAG_FLUID_NOFS", Plane.SAGITTAL, fluid_sensitive=True, fat_suppressed=False),
    Slot("COR_T1", Plane.CORONAL, fluid_sensitive=False, fat_suppressed=False),
    Slot("SAG_T1", Plane.SAGITTAL, fluid_sensitive=False, fat_suppressed=False),
)
"""The order is load-bearing: it is the slot axis of every cached array and of the head's slot embedding."""


class SlotScheme:
    """Assigns each study's series to slots, leaving a slot empty when nothing matches its predicate."""

    def __init__(self, slots: tuple[Slot, ...] = SLOTS) -> None:
        self.slots = slots

    def assign(self, acquisitions: list[Acquisition], planes: dict[str, str]) -> dict[str, Acquisition]:
        """One acquisition per slot, ties broken toward the stack with the most slices.

        A denser stack samples the joint more finely and leaves the band sampler margin to work with.

        A slot with no matching series stays EMPTY and no substitute is admitted from a neighbouring
        predicate. Relaxing the weighting to fill a T1 slot would draw from the same pool
        `SAG_FLUID_NOFS` selects from, putting one series in two slots and leaving the T1 slot holding
        fluid-weighted images; the presence mask would then assert a sequence that was never acquired,
        and the per-diagnosis softmax would split its attention across two identical slots. The mask
        exists to say a slot is absent, which is what an absent slot is.
        """
        chosen: dict[str, Acquisition] = {}
        for slot in self.slots:
            candidates = [
                acquisition
                for acquisition in acquisitions
                if slot.accepts(acquisition, planes.get(acquisition.series, ""))
            ]
            if candidates:
                chosen[slot.name] = max(candidates, key=lambda acquisition: acquisition.slice_count)
        return chosen

    def presence(self, chosen: dict[str, Acquisition]) -> list[float]:
        """The mask the head's softmax renormalises over: 1.0 where a slot was filled, 0.0 where not."""
        return [1.0 if slot.name in chosen else 0.0 for slot in self.slots]
