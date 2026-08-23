"""The token-pooling vocabulary: what each NAME decides about how one image's tokens become one vector.

The third sibling of `slice_pooling` (across the slices of a slot) and `slot_head` (across the slots of a
study). This is the innermost of the three — across the PATCH TOKENS of a single read — and it was the
one still living as a branch inside `model.py`, which crossed the god-file threshold when a volumetric
trunk was wired in. Four call sites speak this vocabulary: the model, the training argument parser, the
frozen-feature trial and the embedding builder.

The reduction is held in the table rather than in a branch for the reason `SlicePooling` states about
its own: a name added to the vocabulary and not to the reduction is a flag that lands in a recorded
config and never reaches the consumer, which is this project's recurring plumbing failure.
"""

import math
from dataclasses import dataclass
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor

CLS_MEAN = "cls_mean"
"""Class token and patch mean — what every member on disk was fitted with."""

CLS_MEAN_FOCAL = "cls_mean_focal"
"""Those two plus the per-channel upper tail, for findings that occupy a small part of the field."""

MEAN = "mean"
"""The patch mean ALONE — the only pooling a trunk without a class token can honestly offer.

Every encoder this project has fitted prepends a class token, so `tokens[:, 0]` was a summary vector and
`CLS_MEAN` could be the default without anyone naming the assumption. A volumetric trunk breaks it:
V-JEPA 2 emits tubelet tokens and nothing else, so `tokens[:, 0]` is the top-left patch of the first
tubelet. Under `CLS_MEAN` that patch would be concatenated to the mean and called a summary, the head
would accept the width, and the run would score like a slightly noisy mean — a silent wrong answer of
exactly the kind `prefix` was introduced to stop. Naming a one-part pooling makes the absence of a class
token a VOCABULARY fact the table enforces, rather than a comment somebody has to hold."""

SIDED = "sided"
"""The class token plus each HALF of the patch grid, so the head can tell medial from lateral.

Every other pooling in this table collapses the token axis to a single vector — `reduce` is typed
`"n t d" -> "n f"` and the focal one discards even the indices `topk` returns. So the head cannot
express WHERE in the field a response sat, and five of our twelve targets are defined by exactly that:
both menisci, both tibiofemoral compartments, and the medial collateral ligament.

WHY A LEFT/RIGHT SPLIT IS A MEDIAL/LATERAL SPLIT HERE, which is the fact the whole pooling rests on:
`Laterality` mirrors every study onto one canonical side inside the cache pipeline (`study.py`), on 100%
of the corpus because the geometric rule covers what the DICOM tag does not. In that frame the medial
compartment is always the same side of the array, so a column split of the patch grid IS the anatomical
split — no new input, no cache rebuild, no label.

WHY IT SUPERSEDES `CLS_MEAN` RATHER THAN COMPETING WITH IT: the two halves partition the patch grid, so
their mean is the patch mean the default already emits. The head can form that sum and can additionally
form the difference. One extra part buys the contrast; nothing is taken away.

WHAT IT IS NOT. Not the side term `side_dense` refuted — that asked the LOSS for the contrast and paid
for it on every label. This changes what the head can SEE and leaves the objective alone. That arm's own
return was the reason to try this one: its compartment contrast rose 0.062 on demand, so the encoder
holds the side and the pooled head is where it is lost."""

FOCAL_FRACTION = 8
"""The focal pooling keeps the top 1/8 of each channel's patch responses."""


@dataclass(frozen=True)
class TokenPooling:
    """What a token-pooling NAME decides: how many `dim`-wide parts it emits, and what emits them.

    One subclass per name rather than one record holding a free function, because the two facts are the
    same fact: `parts` is a promise about the width `reduce` returns, and a subclass is what keeps the
    promise beside the code that has to keep it. `SlicePooling` holds a `Callable` for the opposite
    reason — its aggregators are `nn.Module`s that already exist as classes elsewhere.
    """

    parts: int
    """How many `dim`-wide vectors `reduce` concatenates — the head's input width in trunk units."""

    def reduce(self, tokens: Float[Tensor, "n t d"], prefix: int) -> Float[Tensor, "n f"]:
        """`(tokens, prefix) -> pooled`. The prefix count is the model's, because only the loaded
        backbone's config knows how many non-patch tokens sit in front."""
        message = f"{type(self).__name__} names a pooling and implements no reduction"
        raise NotImplementedError(message)


@dataclass(frozen=True)
class PatchMean(TokenPooling):
    """The patch mean alone, with any prefix tokens dropped first."""

    parts: int = 1

    @override
    def reduce(self, tokens: Float[Tensor, "n t d"], prefix: int) -> Float[Tensor, "n f"]:
        return tokens[:, prefix:].mean(1)


@dataclass(frozen=True)
class ClassAndPatchMean(TokenPooling):
    """The class token concatenated to the patch mean — every published member's pooling."""

    parts: int = 2

    @override
    def reduce(self, tokens: Float[Tensor, "n t d"], prefix: int) -> Float[Tensor, "n f"]:
        return torch.cat([tokens[:, 0], tokens[:, prefix:].mean(1)], dim=1)


@dataclass(frozen=True)
class ClassAndPatchMeanAndTail(TokenPooling):
    """Those two plus the upper tail of each channel over the patch grid.

    Taken per channel rather than by selecting whole patches: a finding occupies a small part of the
    field, so a plain mean over the patches dilutes it, and this keeps the top eighth of each channel's
    responses.
    """

    parts: int = 3

    @override
    def reduce(self, tokens: Float[Tensor, "n t d"], prefix: int) -> Float[Tensor, "n f"]:
        patch = tokens[:, prefix:]
        k = max(1, patch.shape[1] // FOCAL_FRACTION)
        return torch.cat([tokens[:, 0], patch.mean(1), patch.topk(k, dim=1).values.mean(1)], dim=1)


@dataclass(frozen=True)
class ClassAndSidedMean(TokenPooling):
    """The class token plus the mean of each half of the patch grid, split down the midline column.

    The grid is recovered as square from the token count, because that is what every encoder here emits
    at a square input, and a non-square count is refused rather than reshaped into a silent wrong answer.
    An odd column count drops its centre column from both halves: that column IS the midline, so it
    carries no side, and including it in one half would put the midline's response on one compartment.
    """

    parts: int = 3

    @override
    def reduce(self, tokens: Float[Tensor, "n t d"], prefix: int) -> Float[Tensor, "n f"]:
        patch = tokens[:, prefix:]
        side = math.isqrt(patch.shape[1])
        if side * side != patch.shape[1]:
            message = f"{patch.shape[1]} patch tokens are not a square grid, so there is no midline to split on"
            raise ValueError(message)
        grid = patch.unflatten(1, (side, side))
        half = side // 2
        return torch.cat([tokens[:, 0], grid[:, :, :half].mean((1, 2)), grid[:, :, side - half :].mean((1, 2))], dim=1)


TOKEN_POOLINGS: dict[str, TokenPooling] = {
    CLS_MEAN: ClassAndPatchMean(),
    CLS_MEAN_FOCAL: ClassAndPatchMeanAndTail(),
    MEAN: PatchMean(),
    SIDED: ClassAndSidedMean(),
}
"""The vocabulary, mapped to how the token axis is reduced."""

POOL_PARTS: dict[str, int] = {name: pooling.parts for name, pooling in TOKEN_POOLINGS.items()}
"""How many `dim`-wide parts each pooling concatenates, which fixes the head's input width.

Derived rather than written, so the width the head is BUILT for and the width the forward pass PRODUCES
cannot drift apart. The two strings had five more copies — the design's default, the branch in the model,
two argv defaults and the manifest's — and the manifest's was a LITERAL, so a run's own pooling could not
reach the field the serve path rebuilds the design from."""
