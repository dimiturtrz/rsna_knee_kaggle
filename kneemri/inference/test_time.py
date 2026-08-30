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

WARP_CHUNK = 512
"""How many single-channel images one `grid_sample` call may hold.

TWO ENTRIES CAME BACK COMPLETE WITH A BLANK PUBLIC SCORE and this is the only explanation that fits
both (knee-c9q4). The warp ran on the WHOLE chunk at once, so a 48-study batch (6 slots x 12 slices =
3,456 images at 336 px) allocated the float copy, the sampling grid and the result together — about
6.2 GB of transient on a 16 GB T4, on top of the model and the decoded batch. `Ensemble.forwarded`
halves its rows on `torch.OutOfMemoryError`, but it wraps the FORWARD; an out-of-memory here is raised
before the forward is reached and kills the kernel, and a hidden rerun that dies writes no submission
at all — which is exactly what a COMPLETE entry with no score looks like from outside.

The visible set is three studies, 216 images, a fifteenth of the memory. That is why every one of these
runs looked clean in a log and cost a slot anyway: the arm was only ever tested at a size that fits.

512 bounds the transient at ~230 MB regardless of chunk size. Views are averaged over, so this changes
no arithmetic — `test_warp_is_indifferent_to_the_chunk_size` is what says so."""

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

    @property
    def identity(self) -> bool:
        """Whether this averages over nothing — the single unwarped view, so a read is the plain model.

        A LENGTH of one does not answer that question: `SpatialViews(degrees=(6.0,))` is also one pass,
        and it is a different function. The distinction matters to whoever asks whether an array
        computed under some other policy may stand in for this one.
        """
        return self.degrees == (0.0,) and self.shifts == ((0.0, 0.0),)

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
        flat: UInt8[Tensor, "n 1 h w"] = images.reshape(-1, 1, height, width)
        angle = torch.tensor(degrees * torch.pi / 180.0, device=images.device)
        cos, sin = torch.cos(angle), torch.sin(angle)
        theta = torch.tensor([[[cos, -sin, shift_x], [sin, cos, shift_y]]], device=images.device, dtype=torch.float32)
        # ONE GRID FOR EVERY IMAGE, because the transform does not depend on the image. Built at the
        # batch's own length it was a (n, h, w, 2) float32 tensor — 3.1 GB for the 3,456 images a
        # 48-study chunk holds, which is the largest single allocation this serving path ever made.
        grid = nn.functional.affine_grid(theta, [1, 1, height, width], align_corners=False)
        warped = torch.empty_like(flat)
        for begin in range(0, flat.shape[0], WARP_CHUNK):
            piece: Float[Tensor, "c 1 h w"] = flat[begin : begin + WARP_CHUNK].float()
            sampled = nn.functional.grid_sample(
                piece,
                grid.expand(piece.shape[0], height, width, 2),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            warped[begin : begin + WARP_CHUNK] = sampled.round().clamp(0, 255).to(torch.uint8)
        return warped.reshape(batch, slots, depth, height, width)
