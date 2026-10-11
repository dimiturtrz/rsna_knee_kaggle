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
from enum import Enum

import torch
from jaxtyping import Float
from torch import Tensor

from core.data.corpus.slots import SLOTS, Plane, Slot
from core.findings import Finding


class Extent(Enum):
    """How big the abnormality being called physically is — the axis a crop or a resolution trades on.

    Two classes and not a millimetre per finding, because the split is what a reading has to survive:
    a structure whose abnormality is a few voxels across is lost to sampling long before one that
    spans centimetres, and no arm we can run resolves the difference between 2 mm and 3 mm.

    Assigned from the STRUCTURE, never from a measured per-label result. Cartilage sits on the focal
    side because it is 2-4 mm thick and its loss is read at the joint line, even though our own
    annotation study found osteoarthritis label-bottlenecked rather than model-bottlenecked. Fitting
    the classes to that observation would make every subsequent per-label pattern unfalsifiable —
    the grouping would then agree with any result by construction.
    """

    FOCAL = "focal"
    """Millimetre-scale and at the joint centre: a meniscal tear is 1-2 mm, which at 0.387 mm/px is
    three to five pixels — a fraction of one 14 px DINOv2 patch. These are the findings a tighter,
    finer crop can only help."""

    REGIONAL = "regional"
    """Centimetre-scale, and often reaching past the joint centre: an effusion spreads into the
    suprapatellar pouch, a Baker's cyst sits posteriorly, marrow oedema covers a condyle. Resolution
    buys these nothing, and a crop tight enough to help the focal findings should COST them."""


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
    extent: Extent = Extent.REGIONAL
    """How big the thing being called is. Read by the analysis that decides whether a tighter crop
    bought detail, and by nothing in training — `affinity` never sees it, so adding this axis cannot
    move a single logit of the attention prior."""
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
    Finding.ACL: FindingAnatomy(primary=(Plane.SAGITTAL,), secondary=(Plane.CORONAL,), extent=Extent.FOCAL),
    Finding.MCL: FindingAnatomy(primary=(Plane.CORONAL,), extent=Extent.FOCAL),
    # The meniscal body is coronal and its horns are sagittal, so a tear is genuinely read on both.
    Finding.MEDIAL_MENISCUS: FindingAnatomy(primary=(Plane.SAGITTAL, Plane.CORONAL), extent=Extent.FOCAL),
    Finding.LATERAL_MENISCUS: FindingAnatomy(primary=(Plane.SAGITTAL, Plane.CORONAL), extent=Extent.FOCAL),
    # Cartilage loss, joint-space narrowing and osteophytes are morphology: T1 shows them well.
    Finding.MEDIAL_OA: FindingAnatomy(
        primary=(Plane.CORONAL,), secondary=(Plane.SAGITTAL,), needs_fluid=False, extent=Extent.FOCAL
    ),
    Finding.LATERAL_OA: FindingAnatomy(
        primary=(Plane.CORONAL,), secondary=(Plane.SAGITTAL,), needs_fluid=False, extent=Extent.FOCAL
    ),
    Finding.PF_OA: FindingAnatomy(
        primary=(Plane.AXIAL,), secondary=(Plane.SAGITTAL,), needs_fluid=False, extent=Extent.FOCAL
    ),
    Finding.EFFUSION: FindingAnatomy(primary=(Plane.AXIAL, Plane.SAGITTAL), extent=Extent.REGIONAL),
    Finding.SYNOVITIS: FindingAnatomy(primary=(Plane.AXIAL,), secondary=(Plane.SAGITTAL,), extent=Extent.REGIONAL),
    # A Baker's cyst sits posteromedially, and its NECK - the part that identifies it as a cyst rather
    # than a fluid collection - runs between the semimembranosus tendon and the medial head of
    # gastrocnemius, which is an axial relationship. Axial was demoted to secondary in the first version
    # of this table and it was the single worst row in it: the prior cost 0.0589 of AUC on this finding
    # alone, against a mean of -0.0009 across the other eleven, six of which improved.
    Finding.BAKERS: FindingAnatomy(
        primary=(Plane.AXIAL, Plane.SAGITTAL), secondary=(Plane.CORONAL,), extent=Extent.REGIONAL
    ),
    # Marrow oedema is invisible without fat suppression and fluid weighting, whatever the plane.
    Finding.CONTUSION: FindingAnatomy(
        primary=(Plane.SAGITTAL, Plane.CORONAL), secondary=(Plane.AXIAL,), needs_fat_sat=True, extent=Extent.REGIONAL
    ),
    Finding.FRACTURE: FindingAnatomy(
        primary=(Plane.SAGITTAL, Plane.CORONAL), secondary=(Plane.AXIAL,), needs_fat_sat=True, extent=Extent.REGIONAL
    ),
}


@dataclass(frozen=True)
class ScaleProfile:
    """A set of per-label deltas summed up the one axis a crop or a resolution actually trades on.

    The 80 mm arm gates a 99 GB cache rebuild, and its prediction is TWO-SIDED — focal findings up,
    regional ones down — because a uniform lift is a confound and a uniform fall refutes resolution
    outright. Stated that way in prose it is unfalsifiable in practice: twelve deltas read by eye
    always contain a subset that agrees with whichever story is being told. So the split is computed
    from a table written before the arm ran, and the pattern is named rather than narrated.
    """

    focal: float
    regional: float

    @classmethod
    def of(cls, deltas: dict[str, float], anatomy: dict[str, FindingAnatomy] | None = None) -> "ScaleProfile":
        """The mean delta on each side of the scale split; anything not a finding (MEAN) is ignored."""
        table = anatomy if anatomy is not None else ANATOMY
        sides = {
            extent: [
                value for label, value in deltas.items() if table.get(label, None) and table[label].extent is extent
            ]
            for extent in Extent
        }
        return cls(focal=cls._mean(sides[Extent.FOCAL]), regional=cls._mean(sides[Extent.REGIONAL]))

    @staticmethod
    def _mean(values: list[float]) -> float:
        """The average of one side, or nan when the arm scored nothing on it."""
        return sum(values) / len(values) if values else float("nan")

    def verdict(self) -> str:
        """Which of the four patterns this arm produced, said outright.

        Only the first is the mechanism the multi-scale build is predicated on. The other three each
        mean something different and none of them is 'a bit of a win': a uniform move is a story about
        the recipe rather than about the field of view, and an inverted one says the crop removed what
        the findings needed.
        """
        if self.focal > 0 > self.regional:
            return "TRADED detail for context — the two-sided prediction, and the case for multi-scale"
        if self.focal > 0:
            return "UNIFORM LIFT — both sides up, so something other than the field of view moved this"
        if self.regional > 0:
            return "INVERTED — the tighter view helped exactly the findings it should have cost"
        return "REFUTED — nothing gained at either scale, so resolution was not the bottleneck"

    def line(self) -> str:
        """The two means, and NOT the verdict — the split of any arm's deltas, whatever it changed."""
        return f"  by structure scale: focal {self.focal:+.4f}  regional {self.regional:+.4f}"

    def judged(self) -> str:
        """The means WITH the pattern named — only for an arm that moved the field of view.

        Every one of the four verdicts is a sentence about the view ("the tighter view helped exactly
        the findings it should have cost", "resolution was not the bottleneck"), so printing one under
        an arm that changed the schedule or the loss states a conclusion about an axis that arm held
        fixed. `run_comparison` printed it under all six arms of the 2026-08-17 re-read, which told a
        pair-memory arm that resolution was not its bottleneck.
        """
        return f"{self.line()}  -> {self.verdict()}"


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

    def requested(self, targets: tuple[str, ...], n_slot: int, *, wanted: bool) -> Float[Tensor, "o s"] | None:
        """The prior a design ASKED for over `n_slot` slots, or nothing — the refusal, spelled once.

        The table is defined over the canonical slots, so asking for it with a different slot count is
        a mistake rather than something to broadcast quietly: a (12, 6) table against a (12, 2)
        attention map either throws deep inside the head or, worse, does not.

        `StudyModel` and `TrialHead` both made that refusal, four lines apiece, the second admitting in
        a comment that it was the first one's rule. A rule spelled in two constructors is one the proxy
        can drift out of while still sharing the flag's name — which is the drift `TrialHead` exists to
        not have, since a head trial that answers a different question under `--anatomical-prior` is
        worse than no proxy at all.
        """
        if not wanted:
            return None
        if n_slot != len(SLOTS):
            message = f"anatomical prior covers {len(SLOTS)} slots, the head has {n_slot}"
            raise ValueError(message)
        return self.matrix(targets, SLOTS)
