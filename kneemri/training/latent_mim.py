"""Masked image modelling against a TEACHER'S FEATURES rather than pixels, for adapting to knee MRI.

WHY THE TARGET IS THE VARIABLE. `SpectralMae` regresses pixels at the masked positions, and its arm
(`mae_unitfix`) ran -0.0885 against a byte-identical control after the pretrain itself converged
(0.9122 -> 0.5570 hidden-patch MSE). So the objective FIT and the fine-tune still lost, which points at
what is being fitted rather than at whether it fitted. Two lines of evidence say the same thing:

- CAPI (Darcet et al., arXiv 2502.08769) measures exactly this axis on frozen probes, ViT-L on IN1k:
  MAE 79.4, iBOT 80.9, CAPI 82.9; ADE20K k-NN mIoU 21.5 / 26.0 / 29.2. Their mechanism, verbatim, is
  that pixel targets "provide good initialization for fine-tuning, but yield poor frozen
  representations, possibly because the target is too low-level to capture the task's inherent
  uncertainty".
- OUR OWN pair. `ortholarge` IS `v3large` plus 1.25M knee images under a LATENT self-distillation
  objective, and at matched recipe it is +0.0194 (224px/12 slices) and +0.0140 (whole-stack/20). That is
  the only knee-continuation gain this project has ever measured, and it is on this side of the axis.

Note what is NOT claimed: DINOv2 is already a masked-image model — its objective is DINO + iBOT, and
iBOT masks student patches and predicts the teacher's OUTPUT at those positions (Oquab et al., arXiv
2304.07193 §3). So this does not add a missing local signal. It changes the TARGET of one, from pixels
back to features, on our own slices.

THE TEACHER IS NOT AN EMA AT A PUBLISHED MOMENTUM, and that is the single most load-bearing decision
here. Every published momentum was fitted for hundreds of epochs from RANDOM init, where the teacher is
meant to be a slow copy of a student that starts at noise. Ours starts at DINOv2 and runs ~8 epochs, so
the stock teacher's surviving weight `momentum ** updates` is annihilated: at data2vec's vision value
0.9998 over 25,160 updates (batch 128, 8 epochs, 402,600 slices) the prior is down to 0.0065 of itself,
and at DINOv2's own 0.994 it is gone inside the first epoch. The run would then be distilling the
student into its own moving average with no anchor. So the momentum is DERIVED from the run length to
retain a DECLARED fraction of stock (`LatentMimConfig.retention`), which makes the teacher's drift a
stated parameter instead of an inherited constant. Researcher D250; the arithmetic is `tau ** U`, not a
published measurement.

The endpoints are both wrong, which is why the fraction is a knob and not a maximum: a fully frozen
teacher caps the student at DINOv2 and makes this in-domain distillation through a masking bottleneck,
while a fast teacher throws the prior away. `ortholarge`'s gain came from a teacher that moved.
"""

import copy
from dataclasses import dataclass
from typing import override

import torch
import torch.nn.functional as functional
from jaxtyping import Bool, Float
from torch import Tensor

from kneemri.models.encoder import IMAGENET, Normalization
from kneemri.training.masked_objective import MaskConfig, MaskedObjective

FULL_RETENTION = 1.0
"""A retention of exactly 1.0 means a FROZEN teacher — DINOv2's own choice when its teacher started strong.

Verbatim from the paper's distillation section: "we use a larger model as a frozen teacher, keep a spare
EMA of the student that we use as our final model." The EMA is on the STUDENT side there, as the output
model, not on the teacher. The EMA teacher exists in the from-scratch loop because there is nothing
better to point at; the moment something better existed, the same authors stopped tracking it."""


@dataclass(frozen=True)
class LatentMimConfig(MaskConfig):
    """What the student is shown, and what the teacher's features are averaged and scored as."""

    retention: float = 0.6
    """The fraction of the STOCK teacher that must survive the whole run, from which momentum is derived.

    0.6 rather than a published momentum because no published momentum was fitted at our run length or
    from our init (see the module docstring). It sits mid-range of D250's stated 0.5-0.7 starting band:
    the teacher stays majority-stock through the run and still moves, which is what the `ortholarge`
    precedent says a working in-domain continuation does. It is a derivation from a stated intent, NOT a
    measured optimum, and it is the first thing to sweep if the arm lands flat."""

    teacher_blocks: int = 6
    """How many of the teacher's top blocks are averaged into the target. data2vec's vision setting (K=6),
    which is the top half of our twelve — the same half supervision is already allowed to move."""

    smooth_beta: float = 2.0
    """The smooth-L1 knee, data2vec's vision setting. Below it the loss is quadratic, above it linear, so
    a handful of far-out target dimensions cannot dominate a batch the way a plain MSE lets them."""


class LatentMim(MaskedObjective[LatentMimConfig]):
    """A student encoder, an EMA teacher of it, and a smooth-L1 loss on the masked positions.

    No decoder, no pixels, no spectral term: the ONLY difference from `SpectralMae` is what the masked
    positions are regressed against, which is what makes this a one-variable arm against `mae_unitfix`
    on an identical corpus and schedule.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        config: LatentMimConfig,
        normalization: Normalization = IMAGENET,
    ) -> None:
        super().__init__(backbone, config, normalization)
        self.teacher = copy.deepcopy(backbone).requires_grad_(requires_grad=False).eval()
        self.momentum = FULL_RETENTION
        """Set by `plan` once the run's update count is known. 1.0 until then, so an unplanned run holds
        the teacher frozen rather than silently tracking at whatever a default would have been."""

    @override
    def plan(self, updates: int) -> None:
        """Derive the EMA momentum from the number of updates this run will actually take.

        `momentum ** updates == retention` by construction, so the teacher's stock content at the end of
        the run is the number that was declared rather than whatever a paper's constant happens to leave.
        """
        self.momentum = float(self.config.retention ** (1.0 / max(1, updates)))

    @override
    def after_step(self) -> None:
        """Move the teacher toward the student, AFTER the step that scored against it.

        After, not during: an EMA that moved inside the forward pass would make the target a function of
        the batch being scored on it, which is a supervision leak dressed as a schedule.
        """
        if self.momentum >= FULL_RETENTION:
            return
        with torch.no_grad():
            for teacher, student in zip(self.teacher.parameters(), self.backbone.parameters(), strict=True):
                teacher.lerp_(student.detach(), 1.0 - self.momentum)
            for teacher, student in zip(self.teacher.buffers(), self.backbone.buffers(), strict=True):
                teacher.copy_(student.detach())

    def target(self, images: Float[Tensor, "b 1 h w"], tokens: int) -> Float[Tensor, "b n d"]:
        """The teacher's top-K block average over the UNMASKED image, each block normalised first.

        Parameter-less layer normalisation per block is data2vec's explicit anti-collapse device — it
        stops the model "collapsing into a constant representation" and stops "layers with high norm to
        dominate the target features". That second clause is our own D248 table restated: this trunk's
        block norms run 69.8 to 204.6 with b11 the largest, so an unnormalised average over the top six
        would be mostly b11 by construction rather than by choice.

        The teacher reads the image WHOLE. A target computed from the same masked view the student sees
        would be a target that also has to guess, which is not a target.
        """
        with torch.no_grad():
            hidden = self.teacher(pixel_values=self.normalise(images), output_hidden_states=True).hidden_states
        blocks = hidden[-self.config.teacher_blocks :]
        normalised = torch.stack([functional.layer_norm(block, block.shape[-1:]) for block in blocks])
        return self.patches(normalised.mean(dim=0), tokens)

    def latent_loss(
        self, predicted: Float[Tensor, "b n d"], target: Float[Tensor, "b n d"], mask: Bool[Tensor, "b n"]
    ) -> Float[Tensor, ""]:
        """Smooth L1 over the HIDDEN positions only, against the teacher's features there.

        Visible positions are excluded for the same reason the pixel objective excluded them: predicting
        the teacher where the student can see the patch is a copy, and it carries no information about
        the anatomy the mask removed.
        """
        error = functional.smooth_l1_loss(predicted, target, beta=self.config.smooth_beta, reduction="none")
        return (error.mean(dim=-1) * mask).sum() / mask.sum()

    def spread(self, target: Float[Tensor, "b n d"], mask: Bool[Tensor, "b n"]) -> Float[Tensor, ""]:
        """How much the TARGET varies across the masked positions — the collapse alarm, not a loss term.

        A latent objective can drive its loss to zero by making every target identical, and the loss
        curve of that failure is indistinguishable from the loss curve of success. This is the number
        that tells them apart: it is a property of the teacher alone, so it falls only if the
        representation is actually flattening.
        """
        selected = target[mask]
        return selected.var(dim=0).mean()

    @override
    def term_names(self) -> tuple[str, str]:
        return ("latent", "spread")

    @override
    def forward(
        self, images: Float[Tensor, "b 1 h w"], mask: Bool[Tensor, "b n"]
    ) -> tuple[Float[Tensor, ""], Float[Tensor, ""], Float[Tensor, ""]]:
        """The loss, itself, and the collapse alarm beside it."""
        target = self.target(images, mask.shape[1])
        predicted = self.encode(images, mask)
        latent = self.latent_loss(predicted, target, mask)
        return latent, latent, self.spread(target, mask)
