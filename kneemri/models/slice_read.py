"""How a study's BYTES become the tensor the encoder is handed, and the facts the trunk fixes about it.

One value rather than four attributes of `StudyModel`, because they are one fact read four times. The
trunk's config decides whether the read is volumetric (`VolumeRead`), whether its position grid must be
resampled (`GridFit`), and how many prefix tokens sit in front of the patches; the design decides whether
a slice reaches the three input channels as copies of itself or as its through-plane neighbours
(`DepthWindow`). All four are settled at construction, none of them changes afterwards, and every one of
them is consumed by the same step — the reshape, the resize and the normalisation that turn `uint8`
pixels into encoder input.

A PLAIN VALUE AND NOT AN `nn.Module`. It registers no parameter and no buffer, so no state-dict key moves
and every member already on disk loads unchanged — the same argument `EncoderSurgery` is a plain
attribute for. The normalisation constants stay REGISTERED ON THE MODEL and are handed in, because they
ride in the state dict: a member keeps the constants it was actually fitted under whatever the registry
now says.

Its own module because `model.py` had crossed the god-module threshold (fan-in 21 x fan-out 9), and this
is the cluster that could leave: three of the model's nine intra-package imports were `volume_read`,
`grid_fit` and `depth_window`, and no other responsibility of the model touches any of them.
"""

from dataclasses import dataclass

from jaxtyping import Float, UInt8
from torch import Tensor, nn

from kneemri.models.depth_window import DepthWindow
from kneemri.models.grid_fit import GridFit
from kneemri.models.volume_read import VolumeRead

BYTE_SCALE = 255.0
"""What a `uint8` pixel is divided by. Part of the function the weights were fitted to, not a knob."""


@dataclass(frozen=True)
class SliceRead:
    """Everything settled before the first pixel moves: what the trunk takes, and how it is fed."""

    volume: VolumeRead | None
    """The tubelet facts of a VOLUMETRIC trunk, or None for an ordinary 2D one."""

    grid: GridFit
    """Whether the trunk must be told the read size is not the size its position grid was fitted at."""

    prefix_tokens: int
    """How many tokens sit in front of the patches — a class token, plus DINOv3's four registers.

    Getting this wrong is silent: the head accepts the width and the score comes out plausible, which is
    the failure this field exists to prevent."""

    through_plane: bool
    """Whether a slice reaches the encoder with its two depth NEIGHBOURS in the other two channels."""

    @classmethod
    def of(cls, backbone: nn.Module, pool: str, slice_pooling: str, *, through_plane: bool) -> "SliceRead":
        """The read this backbone's own config and this design between them describe.

        The BACKBONE's config is asked and not our registry, because the loaded weights are the only
        authority on what they emit. DINOv2 has one class token and no registers; DINOv3 adds four
        registers, which must be excluded from the patch pooling. A tubelet trunk has no class token at
        all — the first token is a patch.

        Primitives rather than the `StudyDesign` value that holds them, because the design lives with
        the model and this module is one of the model's imports: taking the type would be a cycle.
        """
        config = getattr(backbone, "config", None)
        volume = VolumeRead.of(config, pool, slice_pooling)
        prefix = 0 if volume else 1 + int(getattr(config, "num_register_tokens", 0) or 0)
        return cls(volume, GridFit.of(backbone), prefix, through_plane)

    def extra(self) -> dict[str, bool]:
        """The extra keywords a forward call needs — see `GridFit`, which reads them off the trunk."""
        return self.grid.extra()

    def probe_frames(self, group: int) -> int:
        """The depth the FINGERPRINT's synthetic bag must be widened to for this trunk to read it.

        Unchanged on a 2D trunk, which is what keeps every fingerprint already on disk untouched — see
        `VolumeRead.probe_frames` for why the rounding exists at all.
        """
        return self.volume.probe_frames(group) if self.volume else group

    def check_tokens(self, x: Float[Tensor, "n *pixels"], tokens: Float[Tensor, "n t d"]) -> None:
        """Assert a volumetric trunk read what we think it read, and say nothing about a 2D one.

        The no-op half is the point: the caller states the check once rather than branching on which
        kind of trunk it holds, and `VolumeRead.check_tokens` keeps the only version of the assertion.
        """
        if self.volume:
            self.volume.check_tokens(x, tokens)

    def pixels(
        self,
        imgs: UInt8[Tensor, "b s g p q"],
        mean: Float[Tensor, "1 c 1 1"],
        std: Float[Tensor, "1 c 1 1"],
        img_size: int | None = None,
        *,
        per_slice: bool,
    ) -> Float[Tensor, "n *pixels"]:
        """The encoder's input for a whole bag of slot images — bytes scaled, resized and normalised.

        "channels": three consecutive slices become the encoder's three channels — one forward per slot.
        A per-slice pooling instead encodes each slice ALONE, replicated across the three channels the
        encoder expects, and the slot's embedding is pooled from the results afterwards.

        The difference is not a preference. Channel stacking couples the group's PHYSICAL THICKNESS to
        the slice count, which is measurably why raising inference slices from 12 to 32 cost -0.018
        here while the recipe that pools per-slice can do it freely. It also means a finding is only
        visible if it falls inside a sampled three-slice group.

        THROUGH-PLANE READING CHANGES ONLY WHAT THE THREE CHANNELS HOLD — (i-1, i, i+1) rather than
        three copies of i. The slice axis, the pooling, the token count, the parameter count and the
        FLOP count are all exactly the control's. See `DepthWindow`.

        A VOLUMETRIC TRUNK TAKES THE WINDOW AS FRAMES, NOT CHANNELS, and that is the whole difference it
        buys — see `VolumeRead` for why no reshaping of a 2D stem reaches it. Each slice is greyscale
        replicated to the three channels the trunk was pretrained on, exactly as the per-slice path
        builds it, and only the depth axis is new; it is folded in AFTER the resize and the
        normalisation so both lines are reused rather than growing 5-D copies of themselves.
        """
        batch, slots, group = imgs.shape[:3]
        if per_slice and self.through_plane:
            x = DepthWindow.channels(imgs).reshape(batch * slots * group, 3, *imgs.shape[3:]).float().div(BYTE_SCALE)
        elif per_slice or self.volume:
            x = imgs.reshape(batch * slots * group, 1, *imgs.shape[3:]).expand(-1, 3, -1, -1)
            x = x.float().div(BYTE_SCALE)
        else:
            x = imgs.reshape(batch * slots, *imgs.shape[2:]).float().div_(BYTE_SCALE)
        if img_size is not None and img_size != x.shape[-1]:
            # The cache is held at the highest resolution any configuration needs and the rest
            # downsample from it, so every configuration sees the same pixels through a different
            # sampling grid rather than a different crop.
            x = nn.functional.interpolate(x, size=(img_size, img_size), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        return self.volume.frames(x, batch * slots) if self.volume else x
