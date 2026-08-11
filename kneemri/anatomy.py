"""Where each finding is actually read: the plane that shows it, and whether it needs fluid weighting.

`SlotHead` gives every diagnosis its own query over the six acquisition slots, and then asks it to
discover from study-level labels alone that the ACL is a sagittal structure and patellofemoral cartilage
an axial one. That is a lot to learn from supervision which never says *where* in a study the finding
was seen — 4,407 studies, one label vector each, no localisation at all. The mapping is not something we
need to learn: it is how knee MRI is read, and it has been stable for thirty years.

So it is supplied as a starting point rather than a constraint. The affinity below is added to the
attention logits and remains a learnable parameter, so the model can overrule it wherever the data
disagrees; what it cannot do any more is spend its scarce supervision rediscovering anatomy.

Two facts per finding generate the whole 12x6 table, which is why there are two fields here and not
seventy-two numbers:

- **Which planes show the structure.** A cruciate runs obliquely front-to-back and is followed on
  sagittal slices; the collaterals and the meniscal body run side-to-side and are read coronally;
  patellar cartilage and the retinacula face forward and are read axially.
- **Whether it needs a fluid-sensitive sequence.** Marrow oedema, effusion and synovitis are only
  conspicuous when fluid is bright, so a contusion or a fracture is a fluid-weighted finding. Cartilage
  loss and osteophytes are morphology and read as well on T1, so osteoarthritis is not.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from kneemri.slots import Plane, Slot

PRIMARY = 1.0
"""The plane a finding is normally read on."""

SECONDARY = 0.45
"""A plane that shows the structure but is not where the call is made — roughly half the weight."""

MATCHED_SEQUENCE = 1.0
"""The slot's fluid sensitivity is what this finding needs."""

WRONG_SEQUENCE = 0.5
"""It is not: the structure is still visible, the finding is much less conspicuous."""

PRIOR_STRENGTH = 2.0
"""Scales the affinity into attention logits.

At 2.0 a primary matched slot enters the softmax about e^1.5 ~ 4.5 times higher than a secondary
mismatched one, which is a lean rather than a decision — the point is to start the query in the right
place, not to hard-route it.
"""


@dataclass(frozen=True)
class FindingAnatomy:
    """How one finding is read: on which planes, and whether it needs fluid weighting."""

    primary: tuple[Plane, ...]
    secondary: tuple[Plane, ...] = ()
    needs_fluid: bool = True
    needs_fat_sat: bool = False
    """Marrow oedema is invisible without fat suppression, whatever the plane and whatever the
    weighting — so a bone finding on a non-fat-saturated sequence is not merely less conspicuous, it is
    largely absent. Stated separately from `needs_fluid` because most fluid-weighted findings do not
    need it: an effusion is obvious on a non-fat-saturated T2."""

    def affinity(self, slot: Slot) -> float:
        """How well one acquisition slot shows this finding, in [0, 1]."""
        if slot.plane in self.primary:
            plane = PRIMARY
        elif slot.plane in self.secondary:
            plane = SECONDARY
        else:
            return 0.0
        sequence = MATCHED_SEQUENCE if slot.fluid_sensitive == self.needs_fluid else WRONG_SEQUENCE
        if self.needs_fat_sat and not slot.fat_suppressed:
            sequence *= WRONG_SEQUENCE
        return plane * sequence


ANATOMY: dict[str, FindingAnatomy] = {
    "ACL": FindingAnatomy(primary=(Plane.SAGITTAL,), secondary=(Plane.CORONAL,)),
    "MCL": FindingAnatomy(primary=(Plane.CORONAL,)),
    # The meniscal body is coronal and its horns are sagittal, so a tear is genuinely read on both.
    "Medial Meniscus": FindingAnatomy(primary=(Plane.SAGITTAL, Plane.CORONAL)),
    "Lateral Meniscus": FindingAnatomy(primary=(Plane.SAGITTAL, Plane.CORONAL)),
    # Cartilage loss, joint-space narrowing and osteophytes are morphology: T1 shows them well.
    "Medial OA": FindingAnatomy(primary=(Plane.CORONAL,), secondary=(Plane.SAGITTAL,), needs_fluid=False),
    "Lateral OA": FindingAnatomy(primary=(Plane.CORONAL,), secondary=(Plane.SAGITTAL,), needs_fluid=False),
    "PF OA": FindingAnatomy(primary=(Plane.AXIAL,), secondary=(Plane.SAGITTAL,), needs_fluid=False),
    "Effusion": FindingAnatomy(primary=(Plane.AXIAL, Plane.SAGITTAL)),
    "Synovitis": FindingAnatomy(primary=(Plane.AXIAL,), secondary=(Plane.SAGITTAL,)),
    # A Baker's cyst sits posteromedially, and its NECK - the part that identifies it as a cyst rather
    # than a fluid collection - runs between the semimembranosus tendon and the medial head of
    # gastrocnemius, which is an axial relationship. Axial was demoted to secondary in the first version
    # of this table and it was the single worst row in it: the prior cost 0.0589 of AUC on this finding
    # alone, against a mean of -0.0009 across the other eleven, six of which improved.
    "Baker's": FindingAnatomy(primary=(Plane.AXIAL, Plane.SAGITTAL), secondary=(Plane.CORONAL,)),
    # Marrow oedema is invisible without fat suppression and fluid weighting, whatever the plane.
    "Contusion": FindingAnatomy(primary=(Plane.SAGITTAL, Plane.CORONAL), secondary=(Plane.AXIAL,), needs_fat_sat=True),
    "Fracture": FindingAnatomy(primary=(Plane.SAGITTAL, Plane.CORONAL), secondary=(Plane.AXIAL,), needs_fat_sat=True),
}


class SlotAffinity:
    """The finding-by-slot table the head starts its attention from."""

    def __init__(self, anatomy: dict[str, FindingAnatomy]) -> None:
        self.anatomy = anatomy

    def matrix(self, targets: tuple[str, ...], slots: tuple[Slot, ...]) -> Float[Tensor, "o s"]:
        """Attention logits per (finding, slot), before any training.

        A finding absent from the table gets a flat row rather than zeros everywhere, so adding a
        thirteenth target cannot silently mask every slot for it.
        """
        rows = [
            [PRIOR_STRENGTH * self.anatomy[target].affinity(slot) for slot in slots]
            if target in self.anatomy
            else [0.0] * len(slots)
            for target in targets
        ]
        return torch.tensor(rows, dtype=torch.float32)
