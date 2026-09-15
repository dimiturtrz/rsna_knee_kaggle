"""Withhold whole acquisition slots from a study: the slot-dropout rule, in the one layer both sides read.

Here rather than in `training.augmentation` because two layers apply it and neither may import the
other: training draws it to FIT a member and evaluation draws it to SERVE one over the masks it was
fitted on (`kneemri.evaluation.slotdrop_tta`, knee-ilc58). A copy in each is how the served mask drifts
from the trained one; `Augmentation.dropped` delegates here, drawing exactly what it drew before.
"""

import torch
from jaxtyping import Float, UInt8
from torch import Tensor


class SlotDropout:
    """One rate of whole-slot withholding over the slots a study actually acquired."""

    def __init__(self, rate: float) -> None:
        self.rate = rate

    def dropped(
        self,
        images: UInt8[Tensor, "b n g h w"],
        presence: Float[Tensor, "b n"],
        generator: torch.Generator,
    ) -> tuple[UInt8[Tensor, "b n g h w"], Float[Tensor, "b n"]]:
        """Withhold present slots at random, moving the PIXELS and the PRESENCE MASK together.

        Both, always. Zeroing pixels while the mask still calls the slot acquired is the exact defect
        `SlotReader._stacked` records paying for once — the head then attends to a slot that carries
        nothing, which teaches it that a present slot may be empty rather than that a slot may be
        absent. Clearing the mask alone would leave real pixels behind a zero flag, which is the value
        `SlotHead` is documented to refuse to trust.

        NO STUDY IS EMPTIED. A draw that would withhold every present slot is cancelled for that study
        rather than clamped to one survivor: cancelling keeps the marginal drop rate per slot equal
        across studies, where keeping a random one would make the effective rate depend on how many
        slots a study happens to have — and slot count correlates with the acquisition protocol, which
        is the covariate this whole change exists to stop the model reading.

        An ABSENT slot is never selected, so the rate is a rate over what was actually acquired.
        """
        if self.rate <= 0:
            return images, presence
        acquired = presence > 0
        draw = torch.rand(presence.shape, generator=generator, device=presence.device)
        dropping = (draw < self.rate) & acquired
        survives = dropping.sum(dim=1) < acquired.sum(dim=1)
        dropping = dropping & survives.unsqueeze(1)
        kept = (~dropping).to(presence.dtype)
        return images * kept.view(*kept.shape, 1, 1, 1).to(images.dtype), presence * kept
