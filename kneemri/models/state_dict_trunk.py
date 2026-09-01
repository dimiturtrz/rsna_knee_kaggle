"""A trunk whose pretrained weights arrive as a bare state dict rather than a HuggingFace model directory.

`Encoder.load` fetches through `AutoModel.from_pretrained`, which is the right and only seam for the eight
trunks this project has fitted so far: each is published as a model directory with a `config.json` beside
its weights, so the class that rebuilds it is named by the checkpoint itself. RadImageNet is not published
that way. It ships an `OrderedDict` of tensors and nothing else, so the architecture is not recoverable
from the file — it has to be STATED, which is what `StateDictWeights` is for.

This is the second kind of "where do the weights come from", not a second kind of encoder: what comes back
still wears the head's calling contract, still goes through `TrunkAsTokens`, and still answers
`backbone(pixel_values=)`. Held as a strategy on the registry entry for the reason `Encoder.adapter` is —
a family test inside `load` is correct exactly until the second non-conforming family arrives.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn
from torchvision.models import densenet121


@dataclass(frozen=True)
class FeatureMapOutput:
    """What a transformers backbone returns, for a trunk that returns a feature map instead.

    Distinct from `TokenOutput` and deliberately not merged with it: that type's field is a token
    SEQUENCE and this one is a MAP, and the two are told apart only by their jaxtyping shapes. Reusing
    one for the other would put a shape annotation on the wrapper that its contents contradict, which is
    the failure `shape-contracts` exists to catch rather than to launder.
    """

    last_hidden_state: Float[Tensor, "n d h w"]


class StateDictTrunk(nn.Module):
    """A feature extractor built from stated architecture plus loaded weights, wearing the HF calling contract.

    Two things separate this from handing the bare module to `TrunkAsTokens`: a torchvision feature
    extractor is called positionally and returns a raw `Tensor`, where the adapter calls
    `trunk(pixel_values=)` and reads `.last_hidden_state` off the result. Both halves of that mismatch are
    silent — the positional call raises a `TypeError` that names an unexpected keyword, and the missing
    attribute raises at the first forward rather than at construction.
    """

    def __init__(self, features: nn.Module) -> None:
        super().__init__()
        self.features = features

    @staticmethod
    def densenet_features(model: nn.Module) -> nn.Module:
        """The convolutional stack of a torchvision DenseNet, which is all a released trunk carries.

        Named rather than written as a lambda at the declaration below because `trunk_of` is the one
        field whose wrongness a strict load would NOT catch: pointing it at the whole model fails loudly
        on the classifier keys, but pointing it at any other complete sub-module that happens to accept
        the same keys would not.
        """
        return model.get_submodule("features")

    @override
    def forward(self, pixel_values: Float[Tensor, "n c h w"]) -> FeatureMapOutput:
        return FeatureMapOutput(self.features(pixel_values))


@dataclass(frozen=True)
class StateDictWeights:
    """How to rebuild a trunk whose checkpoint states no architecture, and where its weights live.

    Every field here is a thing the file itself cannot answer. A `config.json` would name the class, the
    head, and the preprocessing; a bare state dict names none of them, so each is declared and each is
    then held to the checkpoint by a strict load — `load_state_dict(strict=True)` is the assertion that
    the stated architecture is the one the weights came off, and it is why none of this is a guess that
    can survive to a run.
    """

    architecture: Callable[[], nn.Module]
    """Builds the empty module the weights go into — `torchvision.models.densenet121`, uninitialised."""

    trunk_of: Callable[[nn.Module], nn.Module]
    """The sub-module the checkpoint actually covers, since a released trunk usually drops its head."""

    prefix: str
    """The key prefix to strip, for a checkpoint saved from inside a wrapper rather than off the trunk."""

    filename: str
    """The weights file, named because a mount is a DIRECTORY and a kernel has no internet to ask."""

    repository: str = ""
    """The HuggingFace repo holding `filename`, used only when no local source is given."""

    def resolve(self, source: str | Path | None) -> Path:
        """The weights file on disk, from a mounted directory or by fetching `repository`.

        A kernel passes the mount and has no internet; a local run passes nothing and downloads once
        into the HuggingFace cache. `source` may be the file itself or the directory holding it, because
        a Kaggle dataset mounts as a directory and the caller does not otherwise know the filename.
        """
        if source is not None:
            path = Path(source)
            return path if path.is_file() else path / self.filename
        # Imported here rather than at module scope because a Kaggle kernel can only reach the
        # HuggingFace stack once the kit is on `sys.path`, and this module is imported before that.
        from huggingface_hub import hf_hub_download  # noqa: PLC0415 - see above

        return Path(hf_hub_download(repo_id=self.repository, filename=self.filename))

    def build(self, source: str | Path | None = None) -> nn.Module:
        """The loaded trunk, wearing the calling contract the head's adapter expects.

        Strict on purpose. A non-strict load is how a stated architecture that does not match the
        checkpoint produces a randomly initialised trunk that trains, logs a plausible curve, and
        answers a question about nothing — the same failure shape `Normalization` was written for.
        """
        stored = torch.load(self.resolve(source), map_location="cpu", weights_only=True)
        stripped = {key.removeprefix(self.prefix): value for key, value in stored.items()}
        trunk = self.trunk_of(self.architecture())
        trunk.load_state_dict(stripped, strict=True)
        return StateDictTrunk(trunk)


RADIMAGENET_DENSENET121 = StateDictWeights(
    architecture=densenet121,
    trunk_of=StateDictTrunk.densenet_features,
    # The checkpoint was saved from `nn.Sequential(*list(densenet121().children())[:-1])`, so every one
    # of its 725 keys is `backbone.0.<torchvision name>`. Stripped rather than renamed: the names under
    # the prefix ARE torchvision's, which is what makes the strict load meaningful.
    prefix="backbone.0.",
    filename="DenseNet121.pt",
    # The official release is Google-Drive-only, which priced this job at several hours until the mirror
    # was found. MIT, like the upstream repository, so it is redistributable to a Kaggle dataset — which
    # is what a submission will need, since a kernel has no internet.
    repository="Lab-Rasool/RadImageNet",
)
"""RadImageNet's DenseNet-121: 1.35M radiology images, 165 supervised classes, no head released.

Verified against the file rather than described from its README — 725 keys strip to a strict load into
`torchvision.models.densenet121().features` with 0 missing, 0 extra and 0 shape mismatch, 7.04M
parameters, and a 448 px read comes out (n, 1024, 14, 14). Stride 32 and width 1024 is the same output
contract `cnxbase` and `swinbase` already satisfy, so `TrunkAsTokens.from_map` wears it unchanged.
"""
