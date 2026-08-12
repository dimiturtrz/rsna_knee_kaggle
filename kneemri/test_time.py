"""Test-time views on the axis a whole-stack member has not already spent.

Overlap TTA buys +0.0146 for a windowed member and exactly nothing for a whole-stack one, because
averaging ten overlapping three-slice windows is just another way of showing the model every slice —
the same mechanism whole-stack already applies in feature space (`knee-eeo`). Slice coverage is spent.

Rigid jitter is not spent. A different sampling grid gives the encoder a genuinely different image of
the same anatomy, and the errors it makes on one are not the errors it makes on another.

EXPECT LESS THAN +0.0146, and say so before measuring. Training already augments with rotation and
shift of exactly this size, so the model has been taught to be invariant to it — which is precisely the
reason it should help LESS than an axis the model was never shown. An honest prior here is a few
thousandths, not a repeat of the slice-coverage windfall.

NO FLIPS, ever, for the same reason training has none: a horizontal flip undoes the laterality
normalisation that five of the twelve targets depend on, turning a medial meniscus into a lateral one.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from jaxtyping import Float, UInt8
from torch import Tensor, nn

DEGREES = (0.0, -6.0, 6.0)
SHIFTS = ((0.0, 0.0),)
"""Rotation is the cheap axis: three views is three forward passes, and shift is left at none until
rotation has earned the cost. Both stay inside the training augmentation's own widths (8 degrees,
0.05 of the frame) — a view the model never saw in training is a distribution shift, not an average."""


@dataclass(frozen=True)
class SpatialViews:
    """Deterministic rigid views of a batch, averaged over at inference.

    Deterministic rather than sampled: a submission that returns a different answer each run cannot be
    debugged, and averaging over a FIXED set is what makes two runs comparable.
    """

    degrees: tuple[float, ...] = DEGREES
    shifts: tuple[tuple[float, float], ...] = SHIFTS

    def __len__(self) -> int:
        """How many forward passes this multiplies the cost by."""
        return len(self.degrees) * len(self.shifts)

    def apply(self, images: UInt8[Tensor, "b n s h w"]) -> Iterator[UInt8[Tensor, "b n s h w"]]:
        """Each view of the batch, the identity view first so a single-view run is a no-op."""
        for degrees in self.degrees:
            for shift_y, shift_x in self.shifts:
                if degrees == 0.0 and shift_y == 0.0 and shift_x == 0.0:
                    yield images
                    continue
                yield self._warp(images, degrees, shift_y, shift_x)

    @staticmethod
    def _warp(
        images: UInt8[Tensor, "b n s h w"], degrees: float, shift_y: float, shift_x: float
    ) -> UInt8[Tensor, "b n s h w"]:
        """One rigid view, applied to every slice of every slot at once.

        The whole bag is warped as a single flattened batch rather than per slot: a slot is not an
        independent image here, it is one view of one knee, and warping slots differently would break
        the correspondence the head attends over.
        """
        batch, slots, depth, height, width = images.shape
        flat: Float[Tensor, "n 1 h w"] = images.reshape(-1, 1, height, width).float()
        angle = torch.tensor(degrees * torch.pi / 180.0, device=images.device)
        cos, sin = torch.cos(angle), torch.sin(angle)
        theta = torch.tensor(
            [[cos, -sin, shift_x], [sin, cos, shift_y]], device=images.device, dtype=torch.float32
        ).expand(flat.shape[0], 2, 3)
        grid = nn.functional.affine_grid(theta, list(flat.shape), align_corners=False)
        warped = nn.functional.grid_sample(flat, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return warped.round().clamp(0, 255).to(torch.uint8).reshape(batch, slots, depth, height, width)
