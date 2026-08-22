"""Which knee was scanned, and how to map it onto a left-knee convention.

Five of the twelve targets are side-defined — both menisci, medial and lateral tibiofemoral OA, and the
medial collateral ligament. Medial and lateral are defined against the body's midline, so which side of
the *image* they fall on depends on which knee was scanned. Unnormalised, those five labels are being
asked to learn from an axis the model cannot observe.

Measured on our corpus: the `Laterality` tag is readable on 49.3% of series — and on some it is present
but empty, which a `hasattr` check counts as present. The image centre's x is computable on 100%. So the
geometric rule is not a refinement; it is the difference between normalising half the corpus and
normalising all of it.
"""

import numpy as np
import torch
from jaxtyping import UInt8
from torch import Tensor

from core.data.corpus.acquisition import Acquisition
from core.data.corpus.slots import Plane

MIDLINE_DEAD_ZONE_MM = 20.0
"""Inside this distance of the midline the sign of x is no better than chance, so the side is left
unresolved rather than guessed — an unresolved study is simply not mirrored."""


class Laterality:
    """Resolves a study's side from the tag where it exists and from geometry where it does not."""

    def __init__(self, dead_zone_mm: float = MIDLINE_DEAD_ZONE_MM) -> None:
        self.dead_zone_mm = dead_zone_mm

    def of_study(self, acquisitions: list[Acquisition]) -> str | None:
        """'L', 'R', or None when neither the tag nor geometry resolves it.

        The median over the study's series is thresholded rather than any single series: one header is
        read per series, and on a sagittal stack that slice can sit anywhere across the joint.
        """
        tagged = [acquisition.laterality_tag for acquisition in acquisitions if acquisition.laterality_tag]
        if tagged:
            return tagged[0]
        centres = [acquisition.centre_x for acquisition in acquisitions if acquisition.centre_x is not None]
        if not centres:
            return None
        median = float(np.median(centres))
        if abs(median) < self.dead_zone_mm:
            return None
        # DICOM patient coordinates are LPS: +x is the patient's left.
        return "R" if median < 0 else "L"

    def normalise(self, images: UInt8[Tensor, "s h w"], plane: str, side: str | None) -> UInt8[Tensor, "s h w"]:
        """Map a right knee onto the left-knee convention; leave left and unresolved studies alone.

        The correction differs by plane. Coronally and axially the medial-lateral direction lies in the
        image plane, so mirroring the last axis maps one knee onto the other. Sagittally it is the
        SLICE axis: each slice is unchanged by mirroring, and what differs is the direction in which the
        stack traverses the joint — so the slice order is reversed instead.
        """
        if side != "R":
            return images
        if plane in (Plane.CORONAL, Plane.AXIAL):
            return torch.flip(images, dims=[-1])
        return torch.flip(images, dims=[0])
