"""How much of the encoder moves, and how fast — the entangled triple, apart from the loop that uses it.

It lived inside `trainer.py` until 2026-09-12, when that file crossed the god-file ceiling and the split
had to be made somewhere. This is the seam that costs nothing: `Adaptation` is a frozen record of four
numbers with no behaviour, it is CONSTRUCTED by five modules that have no other business with `Trainer`
(the flag vocabularies read its defaults to print them), and the trainer holds one rather than being one.
"""

from dataclasses import dataclass

from kneemri.models.encoder_surgery import NO_OFFSET
from kneemri.models.model import NO_LAYER_DECAY

WEIGHT_DECAY = 0.02


@dataclass(frozen=True)
class Adaptation:
    """How much of the encoder moves, and how fast — one object because they cannot be read apart.

    The head is random at initialisation and has everything to learn; the encoder starts from a good
    solution and needs only to be moved off it. That argument sets the 125:1 default ratio, and it was
    inherited from the reference rather than measured.

    Depth and rate are ENTANGLED, which is why they stopped being three constants. Our bracket found
    unfreezing every block worth +0.0008, and a null result at a backbone rate of 8e-6 cannot
    distinguish "more trainable blocks do not help" from "no block was going to move anyway". Asking
    how many blocks should train is only a real question at a rate where a block moves.
    """

    head_lr: float = 1e-3
    backbone_lr: float = 8e-6
    weight_decay: float = WEIGHT_DECAY
    """How hard AdamW pulls every trained weight back toward zero — a member of the same entangled set.

    It was a module constant until 2026-08-31 and no arm had ever varied it. 0.02 came in with the
    reference recipe at a backbone rate of 8e-6; the rate sweep then moved the trunk five-fold to 4e-5
    without re-deriving it, so the decay every current number was measured under is the one chosen for a
    trunk that barely moved. Defaults to that constant, which is what keeps every existing arm identical.
    """

    unfreeze_last: int = 6
    """Six of twelve blocks — inherited from the reference, and only meaningful beside the rate above."""

    unfreeze_offset: int = NO_OFFSET
    """How many blocks at the TOP of the trunk stay frozen, which turns the suffix into a WINDOW.

    Zero for every run before 2026-09-05, and that is the suffix every ledger number was measured
    under. It exists because the block a run should open is decided by the KIND of distribution shift
    (Lee et al., ICLR 2023) and the input-level case — which natural-image pretraining to greyscale MRI
    is — wants EARLY blocks, the one thing `unfreeze_last` alone cannot express. See `BlockWindow`.
    """

    layer_decay: float = NO_LAYER_DECAY
    """How much slower each block below the top one moves — the third member of the entangled triple.

    Defaults to no decay, which is what every ledger number was measured under. It belongs here rather
    than beside the schedule because it is the term that makes the OTHER two separable: at one flat rate
    a depth result and a rate result are the same measurement read twice.
    """
