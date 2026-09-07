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

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import ClassVar

import torch
from jaxtyping import Float, UInt8
from torch import Tensor, nn

from core.findings import TARGETS

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


TOP_TWO = 2
"""How many windows `top2` averages. Named because a bare 2 in a `topk` is the whole policy."""


@dataclass(frozen=True)
class WindowPooling:
    """How a member's overlapping windows become one answer, chosen PER TARGET.

    THE TWO TEST-TIME AXES ARE NOT THE SAME AXIS, and this path averaged over both together until
    2026-09-07 (`Ensemble.predict_member` accumulated one running total over the cross product of views
    and window starts). A VIEW is a re-photograph of the same anatomy, so averaging is the whole point
    — the errors made on one grid are not the errors made on another. A WINDOW is DIFFERENT ANATOMY,
    and averaging over it asserts that every finding is spread evenly through the knee. Five of the
    twelve are not: a fracture, a contusion, a meniscal tear and a Baker's cyst each occupy a few
    slices, and a mean over fourteen windows divides their evidence by fourteen.

    That is also the standing explanation for `knee-eeo` — overlap TTA measured flat on whole-stack
    members — being a fact about the MEAN rather than about the windows. The default here reproduces
    that mean exactly (every window is read under every view, so mean-then-mean equals mean over the
    product), which is what lets the pooling be measured against it without a second variable.
    """

    @staticmethod
    def _mean_over_windows(windows: Float[Tensor, "b w t"]) -> Float[Tensor, "b t"]:
        """Every window counts equally — what this path has always done, on every target."""
        return windows.mean(dim=1)

    @staticmethod
    def _max_over_windows(windows: Float[Tensor, "b w t"]) -> Float[Tensor, "b t"]:
        """The most convinced window decides. For a finding that lives in ONE place: a fracture visible
        in one of fourteen windows enters a mean at a fourteenth of its strength, and the study is
        positive."""
        return windows.amax(dim=1)

    @staticmethod
    def _top_two_over_windows(windows: Float[Tensor, "b w t"]) -> Float[Tensor, "b t"]:
        """The two most convinced windows, averaged. A ligament runs THROUGH slices rather than sitting
        in one, so a single window is thin evidence and the whole stack is dilution; two is the
        compromise."""
        return windows.topk(min(TOP_TWO, windows.shape[1]), dim=1).values.mean(dim=1)

    @staticmethod
    def _centre_window(windows: Float[Tensor, "b w t"]) -> Float[Tensor, "b t"]:
        """The middle window alone — the reading a member gets with no overlap at all, kept as the
        control arm: it says what the extra windows are worth without changing anything else."""
        return windows[:, windows.shape[1] // 2, :]

    REDUCERS: ClassVar[dict[str, Callable[[Float[Tensor, "b w t"]], Float[Tensor, "b t"]]]] = {
        "mean": _mean_over_windows,
        "max": _max_over_windows,
        "top2": _top_two_over_windows,
        "centre": _centre_window,
    }
    """The four ways a stack of window readings becomes one reading, by name."""

    per_target: tuple[str, ...] = tuple("mean" for _ in TARGETS)

    def __post_init__(self) -> None:
        """Refuse a policy that does not name one reducer per target, spelled the way `REDUCERS` is."""
        if len(self.per_target) != len(TARGETS):
            message = f"pooling names {len(self.per_target)} reducers for {len(TARGETS)} targets"
            raise ValueError(message)
        unknown = sorted(set(self.per_target) - set(self.REDUCERS))
        if unknown:
            message = f"unknown window reducers {unknown}; known are {sorted(self.REDUCERS)}"
            raise ValueError(message)

    @classmethod
    def uniform(cls, reducer: str) -> "WindowPooling":
        """One reducer for all twelve — the arm that asks whether the AXIS matters before the split does."""
        return cls(per_target=tuple(reducer for _ in TARGETS))

    @classmethod
    def focal(cls) -> "WindowPooling":
        """Max on the findings that occupy a few slices, top2 on the ligaments, mean on the diffuse ones.

        The split is anatomical, not fitted: effusion, synovitis and the three OA grades are properties
        of the whole joint and a mean is the right reader of them. It matches the one published kernel
        that pools per target (`external/dinov3folds`, lines 993-994) — which offers no measured delta
        for the choice, so this is a hypothesis the 909 has to price, not a borrowed result.
        """
        focal = {"Medial Meniscus", "Lateral Meniscus", "Baker's", "Contusion", "Fracture"}
        ligament = {"ACL", "MCL"}
        return cls(
            per_target=tuple(
                "max" if target in focal else "top2" if target in ligament else "mean" for target in TARGETS
            )
        )

    @property
    def identity(self) -> bool:
        """Whether this is the plain mean, and so may stand in for an array computed before it existed."""
        return set(self.per_target) == {"mean"}

    def pooled(self, windows: Float[Tensor, "b w t"]) -> Float[Tensor, "b t"]:
        """One reading per study per target, out of one reading per window.

        Every reducer runs over the whole batch and the column it owns is taken out of the result. That
        is twelve times the arithmetic of a gather, on a tensor of a few thousand floats — and it keeps
        the reducers total functions of the stack, which is what makes each one testable on its own.
        """
        columns = [self.REDUCERS[name](windows)[:, index] for index, name in enumerate(self.per_target)]
        return torch.stack(columns, dim=1)
