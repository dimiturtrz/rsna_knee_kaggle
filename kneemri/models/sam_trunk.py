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
from typing import override

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn


class TokensInsteadOfNeck(nn.Module):
    """SAM's 256-channel projection, replaced by the 768-wide tokens the blocks actually produce.

    THE NECK IS A DECODER'S INPUT, not a representation. SAM's vision encoder runs twelve ViT-B blocks
    at 768 wide and then a 1x1 conv, LayerNorm, 3x3 conv, LayerNorm stack that compresses to 256
    channels for the mask decoder — a module this project never builds. Reading a classifier off it
    costs a 3x compression fitted to a task we do not run, and the published evidence against it is
    direct: SAM's neck output linear-probes at 11.1% ImageNet top-1 where DINOv2 reads 77.4%
    (SAMantics, arXiv:2411.15288), and the one medical model that states a read-out point drops this
    exact neck to use patch embeddings instead (FODA-PG, arXiv:2409.03947).

    THE OBJECTION WORTH ANSWERING was that MedSAM's fine-tune updated every image-encoder parameter,
    the neck included, and published no layer-wise breakdown — so a neck-concentrated adaptation would
    make this change throw away the medical part of a medical checkpoint and leave a plain SAM under
    MedSAM's name. `kneemri.models.sam_drift` measured it against `facebook/sam-vit-base`, which is the
    release MedSAM started from: all twelve blocks moved between 0.85 and 1.72 of their own initial
    norm, and the neck moved 0.68 — the least-moved region of the fifteen bar the position table. The
    adaptation is in the trunk. Dropping the neck keeps it.

    Kept as a module substituted for the neck rather than as a different forward path because
    `SamVisionEncoder.forward` calls `self.neck` unconditionally: the read-out point is a property of
    the assembled trunk, so it belongs where the trunk is assembled and not in a branch every caller
    has to remember to take.
    """

    @override
    @staticmethod
    def forward(hidden_states: Float[Tensor, "n h w d"]) -> Float[Tensor, "n d h w"]:
        """The blocks' output as a feature MAP, which is the contract `TrunkAsTokens.from_map` reads.

        A permutation and nothing else. SAM carries its spatial states channels-LAST and the real neck
        permutes on the way in, so returning them untouched would hand the head a (n, h, w, 768) tensor
        that flattens into 21 tokens of 21x768 — a shape error nowhere, and a grid read as a sequence.
        """
        return hidden_states.permute(0, 3, 1, 2).contiguous()


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
        # The read-out point, and the reason is on `TokensInsteadOfNeck`. Substituted rather than skipped
        # because `SamVisionEncoder.forward` calls `self.neck` unconditionally, and the discarded module
        # is 1.3M parameters an optimiser would otherwise be handed for a decoder nothing here builds.
        # pyrefly: ignore[bad-assignment] - `neck` is declared `SamVisionNeck`, and the substitute is
        # deliberately not one: it is the read-out point, and narrowing the attribute to the class being
        # replaced is what this line exists to stop honouring.
        encoder.neck = TokensInsteadOfNeck()
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
`pos_embed` (1, 64, 64, 768), and a 336 px read comes out (n, 21, 21, 768) from the blocks. `build`
replaces the neck with `TokensInsteadOfNeck`, so what the head receives is (n, 768, 21, 21): a feature
MAP, the same output contract `cnxbase`, `swinbase` and `radnet121` already satisfy, so
`TrunkAsTokens.from_map` wears it unchanged — at the 768 width every other ViT-B here is read at, which
also removes read-out width as a confound against the `base_dense` and `v3base_dense` controls.

THE OBJECTIVE IS THE DIVERSITY CLAIM. Every trunk in this registry was pretrained either without labels
(DINOv2/v3, ConvNeXtV2, V-JEPA2, RadioDino) or on image-level classes (Swin, ViT-IN21k, RadImageNet). This
one was trained to SEGMENT — a dense, spatially-localised objective — on medical images. That is an axis
of pretraining the project has never varied, and it is varied here at the same 336 px operating point
every other trunk is measured at.
"""
