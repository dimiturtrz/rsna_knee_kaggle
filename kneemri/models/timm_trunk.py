"""A trunk whose weights are keyed the way timm keys them, under a config naming a transformers class.

This is the FOURTH answer to "where do the weights come from", and it is the one that arrived as a
defect rather than as a design. `Encoder.from_hub` is right for a checkpoint whose `config.json` names
the class that rebuilds it. `StateDictWeights` is right for a bare `OrderedDict` that names nothing.
`SamVisionWeights` is right for a model directory holding more than the wanted sub-module. RadioDino is
a fifth case that LOOKS like the first: it ships a `config.json` that says `ViTModel` beside weights
written by timm, so `AutoModel.from_pretrained` builds the right architecture, matches not one key, and
hands back a correctly-shaped ViT-B/16 with every parameter at its random initialisation.

That is what happened. `radiodino_corpus` trained three epochs of a from-scratch trunk (0.6501, 0.6714,
0.6813) against a pre-registration about MEDICAL PRETRAINING, and its lowest branch — REFUTES, under
0.80 — would have closed the medical-image row of the grid and retired MRI-CORE on a loader bug. It was
caught by reading the log, not by a gate; `refuse_unloaded` is the gate, and this module is the fix.

The architecture is STATED here for the same reason `StateDictWeights` states one: timm's key names
carry a shape convention (fused `attn.qkv`, `patch_embed.proj`, a `cls_token` parameter) and not a class,
so nothing in the file can be asked which model they came off. A strict load is what holds the statement
to the weights.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import override

from jaxtyping import Float
from torch import Tensor, nn

from kneemri.models.trunk_tokens import TokenConfig, TokenOutput


class TimmTrunk(nn.Module):
    """A timm vision transformer wearing the HuggingFace calling contract the head reads.

    NOT `TrunkAsTokens`. That wrapper exists for trunks short of the contract's PREFIX — a ConvNeXt map
    and a Swin sequence both arrive with no class token, and it supplies one from the spatial mean. A
    timm ViT is short of a different half: `forward_features` already returns (n, 1 + patches, d) with a
    real class token in front, and it is the CALL that does not match, not the layout. Routing it
    through `from_sequence` would prepend a second prefix and push every patch token one slot along,
    which changes no shape and so raises nothing.
    """

    config: TokenConfig
    """Annotated at class level for the reason `TrunkAsTokens.config` is: `nn.Module.__getattr__` types
    every attribute it serves as `Tensor | Module`, so a reader of `backbone.config.hidden_size` is
    otherwise reading a Tensor as far as the type checker is concerned."""

    def __init__(self, trunk: nn.Module, dim: int) -> None:
        super().__init__()
        self.trunk = trunk
        self.config = TokenConfig(dim)

    @property
    def is_gradient_checkpointing(self) -> bool:
        """What the trunk says, because the trunk is what checkpoints — the readback `switch_coverage`
        proves `--recompute-activations` landed with."""
        return bool(getattr(self.trunk, "grad_checkpointing", False))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, object]) -> None:
        """Forward transformers' checkpointing hook onto timm's differently-spelled one.

        The kwargs are transformers' vocabulary (`use_reentrant`) and timm's switch takes none, so they
        are dropped rather than passed. Named in the signature anyway because `EncoderSurgery` calls
        this by keyword on whatever it holds as a backbone, and the whole point of the contract is that
        the caller does not know which of the wrappers it has.
        """
        _ = gradient_checkpointing_kwargs
        self.trunk.set_grad_checkpointing(enable=True)  # pyrefly: ignore[not-callable]

    @override
    def forward(self, pixel_values: Float[Tensor, "n c h w"]) -> TokenOutput:
        return TokenOutput(self.trunk.forward_features(pixel_values))  # pyrefly: ignore[not-callable]


@dataclass(frozen=True)
class TimmHubWeights:
    """The timm architecture a HuggingFace-hosted checkpoint's keys came off, and where the file is."""

    architecture: str
    """The timm model name, stated because the checkpoint's own `config.json` names a different class."""

    dim: int
    """The trunk's feature width, which the head reads off `config.hidden_size`."""

    filename: str = "model.safetensors"
    """The weights file, named because a mount is a DIRECTORY and a kernel has no internet to ask."""

    repository: str = ""
    """The HuggingFace repository holding `filename`, used only when no local source is given."""

    def resolve(self, source: str | Path | None) -> Path:
        """The weights file on disk, from a mounted directory or by fetching `repository` once.

        `source` may be the file itself or the directory holding it: a Kaggle dataset mounts as a
        directory and the caller does not otherwise know the filename.
        """
        if source is not None:
            path = Path(source)
            return path if path.is_file() else path / self.filename
        # Imported here rather than at module scope because a Kaggle kernel can only reach the
        # HuggingFace stack once the kit is on `sys.path`, and this module is imported before that.
        from huggingface_hub import hf_hub_download  # noqa: PLC0415 - see above

        return Path(hf_hub_download(repo_id=self.repository, filename=self.filename))

    def build(self, source: str | Path | None = None) -> nn.Module:  # devtools-ignore: test-mirror
        """The loaded trunk, wearing the calling contract the head's adapter expects.

        No mirror test, for the reason `SamVisionWeights.build` has none: reaching it means fetching a
        330 MB checkpoint, and the `small` gate forbids a unit test that touches the network. What is
        separable IS tested — `resolve` names the file, `TimmTrunk` wears the contract — and the load
        itself is covered where every other registry row is, by the integration suite.

        Strict, and that is the whole point of the module. A non-strict load here is precisely what
        `AutoModel.from_pretrained` did: it produced a plausible ViT that trained, logged a rising
        curve, and answered a question about nothing.
        """
        # Imported here rather than at module scope for the reason `resolve` names — a kernel reaches
        # neither library until the kit is on `sys.path`.
        import timm  # noqa: PLC0415 - see above
        from safetensors.torch import load_file  # noqa: PLC0415 - see above

        # `dynamic_img_size` and NOT `img_size`. Building at the read size resizes the position table
        # before the load, so a 197-row checkpoint meets a 442-row parameter and the strict load fails —
        # measured, not assumed. Left at the pretrained grid, timm interpolates a COPY per forward and
        # the learnable table stays 14x14, which is what every HF ViT in this registry already does
        # through `interpolate_pos_encoding`. It is the opposite of `SamVisionWeights`, whose table is
        # re-gridded ONCE at build — SAM's is absolute and its patch embedding asserts the input matches,
        # so there is nothing to interpolate per forward and the parameter itself has to move.
        trunk = timm.create_model(self.architecture, pretrained=False, num_classes=0, dynamic_img_size=True)
        trunk.load_state_dict(load_file(str(self.resolve(source))), strict=True)
        return TimmTrunk(trunk, self.dim)
