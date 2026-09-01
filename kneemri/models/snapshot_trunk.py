"""A trunk published as a HuggingFace model directory that lives on this disk rather than on the hub.

`Encoder.from_hub` covers the hub case and `StateDictWeights` covers the bare-`OrderedDict` case. This is
the third: a release whose weights are a checkpoint of somebody else's training script, converted ONCE
into a `config.json` plus `model.safetensors` and kept under the data root's `reference/` kind. The
conversion is the expensive and error-prone half, and it is not something a run should redo.

WHY IT IS NOT A LITERAL PATH ON THE REGISTRY. The registry is committed and the snapshot is not — the data
root is machine-local by construction (`paths.yaml`, gitignored). So the entry names the DIRECTORY under
`reference/` and the root is resolved at load time, which is the same seam every cache in this project
already goes through.
"""

from dataclasses import dataclass
from pathlib import Path

from torch import nn

from core.cli import PATHS
from core.paths import REFERENCE, DataRoot


@dataclass(frozen=True)
class LocalSnapshot:
    """Where a converted HuggingFace snapshot sits under the data root's `reference/` kind."""

    directory: str
    """The snapshot directory relative to `reference/` — it holds a `config.json` and its weights."""

    def resolve(self, source: str | Path | None) -> Path:
        """The snapshot directory, from a mounted path or from this machine's `reference/` kind.

        A kernel passes its mount and has no data root at all; a local run passes nothing. Same shape as
        `StateDictWeights.resolve`, and for the same reason — the two halves of "where are the weights"
        are the kernel's answer and this machine's, never a single constant.
        """
        if source is not None:
            return Path(source)
        return DataRoot.from_config(PATHS).kind(REFERENCE) / self.directory

    def build(self, source: str | Path | None = None) -> nn.Module:  # devtools-ignore: test-mirror
        """The trunk the snapshot's own `config.json` names.

        No mirror test: it loads a gigabyte of real weights off a machine-local path, which a unit test
        may neither reach nor fabricate. It is exercised by `registry_to_checkpoints`, the integration
        suite that already holds every registry entry to its own snapshot.
        """
        # Imported here rather than at module scope because a Kaggle kernel can only reach transformers
        # once the kit is on `sys.path`, and this module is imported before that happens.
        from transformers import AutoModel  # noqa: PLC0415 - see above

        return AutoModel.from_pretrained(self.resolve(source))


ORTHOFOUNDATION_L = LocalSnapshot("orthofoundation/hf_snapshot")
"""OrthoFoundation-L: DINOv3-L continued with DINO-style SSL on 1.25M knee images, MIT licence.

357k radiographs plus 894k knee-MRI slices (arXiv:2601.18250), and architecturally identical to
`facebook/dinov3-vitl16-pretrain-lvd1689m` — 24 blocks, 1024 wide, 16 heads, patch 16, 4 register
tokens — which is what makes `v3large` an EXACT control rather than an adjacent one. The released
`OrthoFoudation-L.pth` is a bare training checkpoint; the snapshot under `reference/` is that file
converted to a transformers directory (qkv split, `mlp.fc1/fc2` to `up_proj/down_proj`, `ls.gamma` to
`layer_scale.lambda1`, `storage_tokens` to `register_tokens`).

THE ONLY IN-DOMAIN CONTINUATION THIS PROJECT CAN TEST WITHOUT SPENDING SSL COMPUTE. Ilse et al. 2025 put
the DINO-family crossing point at ~100k in-domain images and the on-disk unlabelled knee substrate here at
~72k slices, so running the continuation ourselves is priced BELOW the published threshold. This release
is 1.25M images of the same continuation, already done.
"""
