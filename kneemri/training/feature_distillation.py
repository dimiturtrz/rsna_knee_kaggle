"""Distillation at the REPRESENTATION rather than at the verdict — a cosine pulled through a projection.

Beside `terms.py` rather than inside it because it is the one term that owns weights, and so the one
that carries a width pair and a learned map the others have no use for.
"""

from dataclasses import dataclass
from typing import override

import torch
from jaxtyping import Float
from torch import Tensor, nn

from kneemri.training.grades import Predicted
from kneemri.training.terms import FEATURE_TERM, Term

FEATURE_WEIGHT = 1.0
"""The default strength of the cosine term, and 1.0 makes it comparable with cross-entropy rather than
dominant over it. Stage one reaches for a LARGE value on the flag — that is where the label term is
pushed to near-nothing and the twelve heads become a probe — so the default here is the composed case,
not the pretraining one."""


@dataclass(frozen=True)
class Widths:
    """How wide the student's pooled reading is, and how wide the teacher's — the two are unrelated.

    A pair rather than two arguments because they are only ever meaningful together: the projection maps
    one onto the other, and a class holding a `nn.Linear(a, b)` built from mismatched sources fails at
    the first step with a shape error that names neither side. Frozen, and read at construction, because
    a width that changed after the optimiser was built would leave a parameter group pointing at a
    tensor nothing steps — the `attached` docstring's failure exactly.
    """

    student: int
    teacher: int


class FeatureDistillation(Term):
    """A cosine pulling the student's REPRESENTATION onto the teacher's.

    WHY THIS IS NOT `Distillation` WITH A DIFFERENT TARGET. That term's argument — a ranking term beats a
    KL because the metric reads order and discards calibration — is an argument about LOGITS, and it does
    not transfer. There is no ordering to preserve in a `d`-wide representation and no metric reading it;
    what is being taught is a DIRECTION in feature space, so the pairwise machinery is not reused here.

    WHY COSINE AND NOT A DISTANCE. The teacher's features carry a scale this student has no reason to
    reproduce — a different trunk, a different normalisation, a different window count folded in — and an
    L2 target would spend the term's gradient matching that norm. Cosine asks only that the student read
    the study along the same direction.

    WHY A LEARNED PROJECTION AND WHY IT HAS NO BIAS. The widths differ across trunks and must never be
    assumed equal, so the comparison is made after a linear map from student width to teacher width. It
    carries NO BIAS deliberately: a bias can satisfy a cosine target by translating the whole cloud onto
    the teacher's mean direction, which scores well on every study at once while carrying no per-study
    information.

    WHAT STAGE ONE IS FOR. The label term stays present and can be pushed to near-nothing by a large
    weight on this one. The twelve heads are then a probe on a representation rather than the thing being
    fitted, which is what makes the stage a PRETRAIN whose product is an encoder rather than a member.

    THE TEACHER MUST NOT HAVE TRAINED ON THESE STUDIES, and this term cannot check it — the same single
    assumption `Distillation` states.
    """

    key = FEATURE_TERM

    def __init__(self, widths: Widths, weight: float = FEATURE_WEIGHT) -> None:
        super().__init__(weight)
        self.projection = nn.Linear(widths.student, widths.teacher, bias=False)

    def __call__(self, batch: Predicted) -> Float[Tensor, ""]:
        """One minus the cosine between the projected student reading and the teacher's, over the batch.

        The student's context is `(b, o, h)` — one reading per finding — and the mean over the finding
        axis is what is compared. That matches how the teacher's own vector was pooled: its attention
        gate is softmaxed over WINDOWS, giving one vector per finding, and the exporter means over the
        same axis. Comparing the twelve separately would teach the student the teacher's HEAD, which is
        the level `swin_distill_fit` measured at -0.0077.
        """
        embedding = self.arrived(batch.embedding, "student context")
        teacher = self.arrived(batch.teacher_embedding, "teacher embedding")
        projected = self.projection(embedding.float().mean(dim=1))
        return (1.0 - nn.functional.cosine_similarity(projected, teacher.float(), dim=-1)).mean()

    @override
    def attached(self, device: torch.device) -> list[nn.Parameter]:
        """The projection's parameters, moved to the card, for the optimiser to step.

        THE ONE TERM THAT OWNS WEIGHTS, and the reason the hook exists on `Term`. Left unattached the
        projection would sit in no parameter group, stay at its initialisation for the whole arm, and the
        cosine would report a fact about a random map — knee-7gyw's defect, and silent.
        """
        self.projection = self.projection.to(device)
        return list(self.projection.parameters())
