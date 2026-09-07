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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict, override

from jaxtyping import Float
from torch import Tensor, nn

from kneemri.models.trunk_tokens import TokenConfig, TokenOutput, TrunkAsTokens


class TimmOptions(TypedDict, total=False):
    """The build keywords `timm.create_model` may be given here, named so the unpack can be checked.

    `total=False` because the hierarchical case asks for none of them. It exists as a TypedDict rather
    than a `dict[str, bool]` for the reason `TimmHubWeights.options` states: unpacking a value-typed
    mapping into a function with named `Path | str | None` parameters is checked key-by-key against
    every one of them, and fails.
    """

    dynamic_img_size: bool


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
        proves `--recompute-activations` landed with.

        timm puts the flag where the recomputation happens, which is not one place: a flat ViT holds it
        on the model, and a staged trunk like MaxxVit holds one per stage and nothing at the root. So the
        question is asked of every module and answered by any of them — reading only the root reported
        False for a CoAtNet whose stages had all been switched on, which `switch_coverage` then refused.
        """
        return any(bool(getattr(module, "grad_checkpointing", False)) for module in self.modules())

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

    drop_path: float = 0.0
    """Stochastic depth, as the PRETRAINING recipe of the trunk being replicated states it.

    A build keyword and not a training flag, because it is a property of the ARCHITECTURE timm assembles:
    the rate is baked into each block at construction, so a run that sets it afterwards has already built
    the wrong trunk. Zero is the identity for every architecture in this registry, which is why it is
    passed unconditionally rather than through `options` — a keyword no family refuses needs no family
    test.
    """

    classes: int = 0
    """How wide a classifier the CHECKPOINT carries, which is not a thing this project ever runs.

    Zero builds no head at all, which is right for a self-supervised checkpoint that shipped without one.
    A supervised release ships the head it was trained with — CoAtNet's IN-1k finetune carries
    `head.fc.{weight,bias}` — and a headless build meets them as `Unexpected key(s) in state_dict`. The
    answer is to build the head and leave it unused rather than to drop the keys: `forward_features`
    never reaches it, and a strict load is the only thing holding the stated architecture to the file.
    """

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
        trunk = timm.create_model(
            self.architecture,
            pretrained=False,
            num_classes=self.classes,
            drop_path_rate=self.drop_path,
            **self.options(),
        )
        trunk.load_state_dict(load_file(str(self.resolve(source))), strict=True)
        return TimmTrunk(trunk, self.dim)

    def options(self) -> TimmOptions:
        """The build keywords this family needs, which is where the two timm cases differ.

        `dynamic_img_size` is a ViT keyword: it makes the position table interpolate per forward instead
        of being re-gridded into the parameter, which is what lets a 224-grid checkpoint read 336. A
        hierarchical trunk has no such table to interpolate and timm refuses the keyword outright, so it
        is asked for here rather than passed unconditionally.

        A `TypedDict` and not a `dict[str, bool]`: `timm.create_model` declares named parameters
        (`checkpoint_path`, `cache_dir`, `pretrained_cfg`) that a value-typed mapping is checked against
        one by one when unpacked, so the loose type makes the CALL fail to type rather than the mapping.
        """
        return {"dynamic_img_size": True}


@dataclass(frozen=True)
class TimmChunkedWeights(TimmHubWeights):
    """A DINOv2 TRAINING checkpoint, whose blocks are stored in chunks, rebuilt as a flat timm ViT.

    The sixth answer to "where do the weights come from", and it is the one a release gives when it ships
    the state of its own pretraining run rather than a finished model. MRI-CORE is that: a `.pth` holding
    one key, `teacher`, under which sit a `backbone.*` trunk and a `dino_head.*` projector that exists only
    to compute the SSL loss and is discarded here.

    THE CHUNKING IS THE PART THAT CANNOT BE GUESSED. DINOv2 builds its blocks as `block_chunks=4` groups,
    so the twelve blocks are keyed `blocks.<group>.<block>` — and the second index is the GLOBAL block
    number, not an offset inside the group, because each chunk is padded at the front with `nn.Identity`.
    The pairs on this file are (0,0) (0,1) (0,2) (1,3) ... (3,11), verified against the file. A reader who
    assumed `group * 3 + offset` gets a mapping that is right for the first three blocks and wrong for the
    other nine, which a strict load catches and a lenient one would not.

    Everything else is the parent's argument: the architecture is STATED because timm key names carry a
    shape convention and not a class, and `strict=True` is what holds the statement to the file.
    """

    directory: str = ""
    """Where the checkpoint sits under the data root's `reference/` kind, for the reason `LocalSnapshot`
    names one: the registry is committed and the weights are machine-local, so the path is resolved at
    load time rather than written down."""

    section: str = ""
    """The top-level key holding the trunk, since a training checkpoint stores several networks."""

    prefix: str = ""
    """The trunk's key prefix under `section`, stripped so what remains is timm's own naming."""

    dropped: tuple[str, ...] = ()
    """Keys the pretraining objective needed and a feature extractor does not."""

    @override
    def resolve(self, source: str | Path | None) -> Path:
        """The checkpoint file, from a mount or from this machine's `reference/` kind.

        The parent fetches from HuggingFace when given nothing; these weights are published on Google
        Drive alone, so there is no repository to fall back to and the local root is the only answer.
        """
        if source is not None:
            path = Path(source)
            return path if path.is_file() else path / self.filename
        # Imported here rather than at module scope because a Kaggle kernel has no data root at all, and
        # `core.cli` reads a config file this module is imported long before.
        from core.cli import PATHS  # noqa: PLC0415 - see above
        from core.paths import REFERENCE, DataRoot  # noqa: PLC0415 - see above

        return DataRoot.from_config(PATHS).kind(REFERENCE) / self.directory / self.filename

    def flattened(self, stored: dict[str, dict[str, Tensor]]) -> dict[str, Tensor]:
        """The trunk's parameters under timm's flat names, with the SSL machinery left behind."""
        trunk: dict[str, Tensor] = {}
        for key, value in stored[self.section].items():
            if not key.startswith(self.prefix):
                continue
            name = key.removeprefix(self.prefix)
            if name in self.dropped:
                continue
            trunk[re.sub(r"^blocks\.\d+\.(\d+)\.", r"blocks.\1.", name)] = value
        return trunk

    @override
    def build(self, source: str | Path | None = None) -> nn.Module:  # devtools-ignore: test-mirror
        """The loaded trunk wearing the head's calling contract. No mirror test, for the reason the parent
        `build` has none: it reads 436 MB off a machine-local path a unit test may neither reach nor fake."""
        # Imported here rather than at module scope for the reason `resolve` names — a kernel reaches
        # neither library until the kit is on `sys.path`.
        import timm  # noqa: PLC0415 - see above
        import torch  # noqa: PLC0415 - see above

        stored = torch.load(self.resolve(source), map_location="cpu", weights_only=True)
        trunk = timm.create_model(
            self.architecture,
            pretrained=False,
            num_classes=self.classes,
            drop_path_rate=self.drop_path,
            **self.options(),
        )
        trunk.load_state_dict(self.flattened(stored), strict=True)
        return TimmTrunk(trunk, self.dim)


MRICORE_VITB = TimmChunkedWeights(
    "vit_base_patch16_224",
    768,
    filename="MRI_CORE_vitb.pth",
    directory="mricore",
    section="teacher",
    prefix="backbone.",
    # The masked-image-modelling placeholder. It is a parameter of the pretraining objective, never of a
    # forward pass, and timm's ViT has no slot for it — so it is dropped rather than mapped.
    dropped=("mask_token",),
)
"""MRI-CORE: a ViT-B/16 pretrained with DINOv2 on 6.1M MRI slices across 18 body locations, Apache 2.0.

Verified against the file rather than described from its README — 159 keys under `teacher`, of which 151
are `backbone.*` and 8 are the discarded `dino_head.*`; the remainder flatten to a strict load into
`timm.create_model('vit_base_patch16_224')` with 0 missing, 0 extra and 0 shape mismatch, 85.80M
parameters, and a 336 px read comes out (n, 442, 768).
"""


@dataclass(frozen=True)
class TimmMapWeights(TimmHubWeights):
    """A timm trunk whose `forward_features` emits a MAP, wearing the contract in two layers.

    The composition is the point. `TimmTrunk` fixes the CALL — timm answers `forward_features`, the head
    calls `backbone(pixel_values=)` — and `TrunkAsTokens.from_map` fixes the LAYOUT, flattening (n, d, h,
    w) and supplying the prefix slot the head's `tokens[:, 1:]` assumes. A CoAtNet is short of both
    halves, and stacking the two wrappers is what keeps either from growing a second case: gradient
    checkpointing and the readback both delegate straight down the chain.

    It is a hierarchical trunk, so `dynamic_img_size` is refused and the read size is fixed by the
    architecture rather than interpolated. `CoAtNet-rmlp` pins its relative-position MLP to the 12x12
    grid its stage-3 attention was pretrained on: 448 px raises `The size of tensor a (784) must match
    the size of tensor b (576)`, measured, so 384 is the only read and `require_tiling` cannot say so.
    """

    @override
    def options(self) -> TimmOptions:
        """None — the position encoding is relative and baked to a grid, so there is nothing to re-grid."""
        return {}

    @override
    def build(self, source: str | Path | None = None) -> nn.Module:  # devtools-ignore: test-mirror
        """The loaded trunk wearing both halves of the contract. No mirror test, for the reason the base
        `build` has none: reaching it fetches a 290 MB checkpoint and the `small` gate forbids the network."""
        return TrunkAsTokens.from_map(super().build(source), self.dim)
