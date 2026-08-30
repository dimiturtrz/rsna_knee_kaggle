"""Whether a trunk must be TOLD that the read is not the size its position grid was fitted at.

A vision transformer carries one learned position embedding per patch of a FIXED grid, so reading it at
another size needs those embeddings resampled. DINOv2 and DINOv3 do that silently inside their own
forward, which is why every trunk this project had read at 336 px without anyone naming the question.
An AugReg ViT does not: `google/vit-base-patch16-224-in21k` at 336 px raises

    ValueError: Input image size (336*336) doesn't match model (224*224).

before a single block runs. `transformers` exposes the resampling as a forward keyword rather than as a
config field, so it is a fact about the CALL and cannot be declared in the encoder table.

READ OFF THE TRUNK, NOT OFF OUR REGISTRY, for the reason `KneeModel` states about every other fact it
derives: the loaded weights are the only authority on what they accept. A registry flag would be a
second place for a family to be described, and it would be wrong for exactly the trunk that arrives
mounted from a Kaggle dataset under a path rather than a name.

PASSING IT WHENEVER IT IS ACCEPTED CHANGES NO MEMBER ALREADY ON DISK. Of the four families here only ViT
and Swin take the keyword at all, and `microsoft/swin-base-patch4-window12-384-in22k` sets
`use_absolute_embeddings=False` — it positions by relative-position bias tables, so there is no absolute
grid to resample and the keyword is inert. DINOv2, DINOv3 and ConvNeXt do not accept it and are not
handed it. So no fingerprint moves, which is the property that made a signature reading preferable to a
size comparison: `image_size` is declared by trunks that enforce it and by trunks that pad to it, and
telling those two apart would have needed the family test this module exists to avoid.
"""

import inspect
from dataclasses import dataclass

from torch import nn

INTERPOLATE = "interpolate_pos_encoding"


@dataclass(frozen=True)
class GridFit:
    """The one keyword a trunk needs when the read size is not the size its grid was fitted at."""

    interpolates: bool
    """Whether this trunk's forward will resample its position grid when asked to."""

    @staticmethod
    def of(backbone: nn.Module) -> "GridFit":
        """What the trunk's own forward signature says it accepts.

        A signature reading rather than a family name: the wrapper in `trunk_tokens` forwards a fixed
        keyword set, a mounted trunk arrives without its repository name, and a stand-in under test is
        any callable at all. Each of those answers this question correctly by being asked.
        """
        return GridFit(INTERPOLATE in inspect.signature(backbone.forward).parameters)

    def extra(self) -> dict[str, bool]:
        """The keywords to add to a forward call — empty for a trunk that resamples on its own."""
        return {INTERPOLATE: True} if self.interpolates else {}
