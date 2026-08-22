"""What a VOLUMETRIC trunk changes about a read, and the guard that proves the change landed.

A tubelet trunk consumes the depth window itself: it spans `tubelet` slices per token and emits several
depth positions, which no reshaping of a 2D stem can do at any window size. `Conv2d(3, D, k=14, s=14)`
over slices (i-1, i, i+1) is mathematically IDENTICAL to `Conv3d(1, D, k=(3,14,14), s=(1,14,14))`, so
channel stacking at window 3 collapses the window to one linear projection per output dimension and one
depth position — measured flat on this data as `through_dense` (-0.0020 against its control). The residue
that null leaves standing is a window WIDER than 3 and a stem emitting SEVERAL depth positions, and this
is the module that reaches it.

Its own module rather than three branches in `model.py`, which had crossed the god-file threshold: the
tubelet decides the pooling, the prefix, the tensor layout and the guard, and those four facts are one
fact about the trunk.
"""

from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from kneemri.models.slice_pooling import SLICE_POOLINGS
from kneemri.models.token_pooling import MEAN


@dataclass(frozen=True)
class VolumeRead:
    """The three numbers a volumetric trunk fixes about the tensor it is handed."""

    tubelet: int
    """How many SLICES one token spans. Non-zero is what makes a trunk volumetric at all."""

    patch: int
    native: int
    """The spatial size the weights were FITTED at. Not a suggestion — see `check_tokens`."""

    @staticmethod
    def of(config: object, pool: str, slice_pooling: str) -> "VolumeRead | None":
        """The read this backbone's own config describes, or None for an ordinary 2D trunk.

        The CONFIG is asked and not our registry, because the loaded weights are the only authority on
        what they emit. Both refusals live here rather than in the forward pass: a 5-D tensor reaching a
        video model as an unknown keyword names transformers in the traceback rather than the recipe
        that asked for it.
        """
        tubelet = int(getattr(config, "tubelet_size", 0) or 0)
        if not tubelet:
            return None
        if pool != MEAN:
            message = f"a tubelet trunk emits no class token, so {pool!r} cannot be pooled; use {MEAN!r}"
            raise ValueError(message)
        if SLICE_POOLINGS[slice_pooling].per_slice:
            # A per-slice pooling encodes each slice ALONE, which is the one thing this trunk exists not
            # to do: hand it one frame and it is a 2D encoder with extra steps.
            message = (
                f"{slice_pooling!r} encodes each slice alone, which spends a tubelet trunk's only "
                f"advantage; a volumetric read needs a whole-stack pooling"
            )
            raise ValueError(message)
        return VolumeRead(tubelet, int(config.patch_size), int(config.crop_size))  # type: ignore[attr-defined]

    def frames(self, x: Float[Tensor, "n c p q"], videos: int) -> Float[Tensor, "n f c p q"]:
        """Fold a flat stack of greyscale-replicated slices back into frames of one video each.

        Built flat by the caller and folded here so the resize and the normalisation lines are reused
        exactly rather than growing 5-D copies of themselves.
        """
        return x.reshape(videos, x.shape[0] // videos, *x.shape[1:])

    def probe_frames(self, group: int) -> int:
        """The nearest frame count at or above `group` that this trunk can actually read.

        Only the FINGERPRINT asks for this. Its bag is a synthetic identity probe, so its depth is a
        constant of the question (`GROUP` = 3) rather than of the run — and three slices is one and a
        half tubelets, which `check_tokens` refuses. The probe died there after a whole training epoch
        had already succeeded, which reads in a traceback as "the training window is 3" and is not.

        Rounding up rather than stamping the run's own window, because the window is not stored anywhere
        the VERIFIER can reach: `Member.group` is the config's windowed width, it feeds `pixel_group`,
        and a whole-stack member sharing pixels with an existing one must keep sharing its decode. Both
        sides derive this from the loaded trunk instead, so writer and verifier cannot drift, and a 2D
        trunk has no `VolumeRead` at all — every fingerprint already on disk is untouched.
        """
        return group + (-group % self.tubelet)

    def check_tokens(self, x: Float[Tensor, "n f c h w"], tokens: Float[Tensor, "n t d"]) -> None:
        """Assert the trunk read what we think it read — against its CONFIG, not against the input.

        THE FIRST VERSION OF THIS GUARD WAS TAUTOLOGICAL AND A NEGATIVE TEST IS WHAT FOUND IT. It
        predicted the token count from `x`'s own shape and compared it to what the trunk returned — but
        the trunk derives its grid from that same tensor, so the two agreed at 224 and at 240 exactly as
        they agreed at 256, and the check could not fail. A guard that cannot fail is worse than no
        guard, because it reads in review as coverage.

        The quantity that CAN be wrong is the read size against the trunk's native crop. Depth is
        genuinely free here — 3D RoPE derives position ids from the run-time frame count and there is no
        learned position table, so a 12-frame read of a 64-frame-pretrained model is not an
        extrapolation. The SPATIAL grid is not free the same way: the weights were fitted at
        `native // patch` and a different read silently hands them a grid they never saw. Measured on
        V-JEPA 2 at 4 frames: 256 -> 512 tokens, 240 -> 450, 224 -> 392, no error at any of them. So the
        native crop is asserted, and the token count alongside it as the cheap plumbing witness that the
        frame axis reached the trunk as frames rather than as anything else.
        """
        frames, _, height, width = x.shape[1:]
        expected = (self.native // self.patch) ** 2 * (frames // self.tubelet)
        if (height, width) != (self.native, self.native) or frames % self.tubelet or tokens.shape[1] != expected:
            message = (
                f"this trunk was fitted at {self.native}x{self.native} on patch {self.patch} with "
                f"tubelet {self.tubelet}, so a {height}x{width} read over {frames} frames predicts "
                f"{expected} tokens and the trunk emitted {tokens.shape[1]}"
            )
            raise ValueError(message)
