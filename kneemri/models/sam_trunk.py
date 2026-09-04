"""A trunk published as a segmentation model, of which only the vision encoder is wanted.

This is the third answer to "where do the weights come from", and it is distinct from both existing ones.
`Encoder.from_hub` calls `AutoModel.from_pretrained` and hands back the whole model, which is right for
every trunk published AS a feature extractor. `StateDictWeights` states an architecture the checkpoint
cannot name. MedSAM is neither: it is a complete HuggingFace model directory that rebuilds itself
perfectly — into a `SamModel`, three quarters of which is a prompt encoder and a mask decoder this
project has no use for. The weights wanted are one sub-module of it.

THE GRID IS THE REAL WORK, not the sub-module. SAM's vision encoder carries ABSOLUTE position embeddings
as a learned (1, 64, 64, 768) parameter, sized for the 1024 px read it was pretrained at, and its patch
embedding asserts the input matches — `Input image size (336*336) doesn't match model (1024*1024)`. So
unlike DINOv2, which interpolates its own position embeddings on the way through, a SAM encoder must be
RE-GRIDDED before it is trained, once, at build. Doing it per forward is not an option: `pos_embed` is a
learnable parameter, and resizing it inside `forward` would hand the optimiser a different tensor than
the one it was given and discard what training had put there.

A 1024 px read is why the re-grid is not optional rather than a nicety. At patch 16 that is 4096 tokens a
slice, and a 20-slice study would run 81,920 of them — an order of magnitude past the 8,820 the shipped
recipe runs. Re-gridded to 336 the same study is 8,820 tokens, which is the operating point every other
trunk here is measured at.
"""

from dataclasses import dataclass
from pathlib import Path

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn


@dataclass(frozen=True)
class SamVisionWeights:
    """How to reach the vision encoder inside a released SAM, and the grid to re-scale it to.

    `read_size` is a FIELD rather than an argument to `build` because `Encoder.load` takes only a source,
    and threading a size through it would put a per-trunk parameter on the one seam whose purpose is that
    every trunk loads the same way. It is a property of the ENTRY for the same reason `swinbase` pins its
    only legal read in a comment: the weights are only correct at one geometry, and which one is a fact
    about the registry row, not about the caller.
    """

    repository: str
    """The HuggingFace model directory holding the full SAM, used when no mounted source is given."""

    read_size: int
    """The read this entry is re-gridded to, and the ONLY size an arm on this trunk may use.

    Not a default that a sweep may vary. The position embeddings are resized to `read_size // patch` at
    build and then TRAINED at that grid, so a checkpoint fitted here is a function of this number; a run
    that read anything else would interpolate a second time, off an already-interpolated table, and hand
    back a plausible answer computed on a geometry no stage of training saw.
    """

    patch: int = 16
    """SAM's patch stride. Stated rather than read off the config so `grid` can be derived before a
    fetch, which is what lets a bad `read_size` be refused without touching the network."""

    @property
    def grid(self) -> int:
        """Tokens a side after re-gridding — `read_size` cut into `patch`-sized squares."""
        return self.read_size // self.patch

    @staticmethod
    def regridded(embed: Float[Tensor, "n g g d"], grid: int) -> Float[Tensor, "n h h d"]:
        """The position embedding table resampled to a `grid` x `grid` layout, channels last throughout.

        Bicubic, and over the two SPATIAL axes only, which is why the tensor is permuted either side of
        the call: `F.interpolate` reads (n, c, h, w) and SAM stores (n, h, w, c). Resampling the stored
        layout directly would interpolate across the 768 feature channels as though they were image
        columns — a silent corruption that changes no shape and so raises nothing.

        Separated from `build` because it is the whole of the geometry and none of the network. It is
        what a unit test can exercise, and the `small` gate forbids one that fetches weights.
        """
        spatial = embed.permute(0, 3, 1, 2)
        resized = F.interpolate(spatial, size=(grid, grid), mode="bicubic", align_corners=False)
        return resized.permute(0, 2, 3, 1).contiguous()

    def build(self, source: str | Path | None = None) -> nn.Module:  # devtools-ignore: test-mirror
        """The re-gridded vision encoder, wearing the calling contract the head's adapter expects.

        No mirror test: reaching it means fetching a 375 MB checkpoint, which is a network call the
        `small` gate forbids in a unit test. The geometry it delegates to IS tested; what is left here is
        the fetch and three assignments, covered where every other entry is — the integration suite that
        loads each registry row against its own snapshot.

        BOTH copies of the size are set. `config.image_size` is what the encoder reports and
        `patch_embed.image_size` is what actually raises, and they are separate attributes: setting only
        the config leaves the assertion reading 1024 and the run dies at the first batch rather than at
        construction.
        """
        # Imported here rather than at module scope because a Kaggle kernel can only reach transformers
        # once the kit is on `sys.path`, and this module is imported before that happens.
        from transformers import SamModel  # noqa: PLC0415 - see above

        encoder = SamModel.from_pretrained(source or self.repository).vision_encoder
        embed = encoder.pos_embed
        if embed is None:
            # Not a type-checker appeasement: SAM's config has `use_abs_pos`, and a checkpoint built with
            # it False carries no table to re-grid. Such a trunk would run at any size and silently be a
            # different model than the one this entry is measured as, so it is refused rather than served.
            message = f"{self.repository} carries no pos_embed — nothing to re-grid to {self.grid}x{self.grid}"
            raise ValueError(message)
        embed.data = self.regridded(embed.data, self.grid)
        encoder.config.image_size = self.read_size
        encoder.patch_embed.image_size = [self.read_size, self.read_size]
        return encoder


MEDSAM_VIT_B = SamVisionWeights(
    # The mirror, NOT `wanglab/medsam-vit-b` from the MedSAM README — that id 404s against the HF API
    # (checked, not assumed). This one is apache-2.0 and ungated, read from the API rather than the model
    # card, so it is redistributable to a Kaggle dataset when a member earns the seat.
    repository="flaviagiammarino/medsam-vit-base",
    read_size=336,
)
"""MedSAM's ViT-B image encoder: SAM fine-tuned on 1M+ medical images across modalities, MSK MRI included.

Verified against the weights rather than described from the card — 89.7M parameters in the encoder alone,
`pos_embed` (1, 64, 64, 768), and a 336 px read comes out (n, 256, 21, 21) after the neck. Width 256 and a
feature MAP is the same output contract `cnxbase`, `swinbase` and `radnet121` already satisfy, so
`TrunkAsTokens.from_map` wears it unchanged.

THE OBJECTIVE IS THE DIVERSITY CLAIM. Every trunk in this registry was pretrained either without labels
(DINOv2/v3, ConvNeXtV2, V-JEPA2, RadioDino) or on image-level classes (Swin, ViT-IN21k, RadImageNet). This
one was trained to SEGMENT — a dense, spatially-localised objective — on medical images. That is an axis
of pretraining the project has never varied, and it is varied here at the same 336 px operating point
every other trunk is measured at.
"""
