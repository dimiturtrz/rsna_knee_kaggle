"""The representation term: that it is a COSINE, that its projection is steppable, and that it refuses."""

import pytest
import torch

from kneemri.training.feature_distillation import FeatureDistilledObjective, Widths
from kneemri.training.grades import Predicted
from kneemri.training.objective import FEATURE_TERM

FINDINGS = 3
STUDENT_WIDTH = 4
TEACHER_WIDTH = 6
BATCH = 2


def _objective(weight: float = 1.0) -> FeatureDistilledObjective:
    """The objective over a fixed projection, seeded so two constructions in one test are comparable."""
    torch.manual_seed(0)
    return FeatureDistilledObjective(Widths(STUDENT_WIDTH, TEACHER_WIDTH), weight)


def _batch(teacher: torch.Tensor | None = None) -> Predicted:
    """A batch carrying a student context and a teacher reading, both tiny and both deterministic."""
    scores = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    torch.manual_seed(1)
    return Predicted(
        torch.zeros(BATCH, FINDINGS),
        scores,
        torch.ones_like(scores),
        embedding=torch.rand(BATCH, FINDINGS, STUDENT_WIDTH),
        teacher_embedding=torch.rand(BATCH, TEACHER_WIDTH) if teacher is None else teacher,
    )


def test_widths() -> None:
    """The pair is what sizes the projection, so its two numbers must reach it in that order.

    Swapped, the map would run teacher-width to student-width and fail at the first step with a shape
    error that names neither side — which is the confusion the value object exists to make impossible.
    """
    projection = _objective().projection

    assert Widths(STUDENT_WIDTH, TEACHER_WIDTH).student == STUDENT_WIDTH
    assert (projection.in_features, projection.out_features) == (STUDENT_WIDTH, TEACHER_WIDTH)
    assert projection.bias is None, (
        "a bias could satisfy the cosine by translating the whole cloud onto the teacher's mean "
        "direction, scoring well on every study at once while carrying nothing per study"
    )


def test_imitation() -> None:
    """One minus the cosine: zero when the projected student already points the teacher's way.

    Pinned against a teacher CONSTRUCTED from the projection's own output rather than against a
    remembered number, because the thing worth asserting is that the two sides are compared after the
    map and along the direction axis — a term computed before the projection, or reduced over the wrong
    axis, would not go to zero here.
    """
    objective = _objective()
    batch = _batch()
    assert batch.embedding is not None
    projected = objective.projection(batch.embedding.mean(dim=1))

    aligned = objective.imitation(_batch(teacher=projected.detach() * 3.0))
    assert float(aligned) == pytest.approx(0.0, abs=1e-6), "scale is not part of a cosine"

    opposed = objective.imitation(_batch(teacher=-projected.detach()))
    assert float(opposed) == pytest.approx(2.0, abs=1e-6)


def test_imitation_raises_when_either_side_never_arrived() -> None:
    """A missing half must RAISE, not contribute zero — the arm would otherwise be its own control.

    A feature-distilling run whose teacher never joined would descend the plain recipe under this
    class's name, land in the ledger as a distillation arm, and read as a refutation of the mechanism.
    """
    scores = torch.zeros(BATCH, FINDINGS)
    logits = torch.zeros(BATCH, FINDINGS)
    neither = Predicted(logits, scores, torch.ones_like(scores))
    student_only = Predicted(
        logits, scores, torch.ones_like(scores), embedding=torch.zeros(BATCH, FINDINGS, STUDENT_WIDTH)
    )
    teacher_only = Predicted(
        logits, scores, torch.ones_like(scores), teacher_embedding=torch.zeros(BATCH, TEACHER_WIDTH)
    )

    for batch in (neither, student_only, teacher_only):
        with pytest.raises(ValueError, match="plumbing failure"):
            _objective().imitation(batch)


def test_attached() -> None:
    """The projection's weights are handed to the caller that builds the optimiser, and moved first.

    THE ONE TERM IN THIS FAMILY THAT OWNS WEIGHTS. `StudyModel.parameter_groups` collects the head by
    EXCLUSION from the backbone and cannot see a tensor the loss holds, so an unattached projection sits
    in no parameter group and stays at its initialisation for the whole arm — knee-7gyw's defect, and
    silent. Identity is asserted, not shape: a COPY would satisfy a shape check and still not be stepped.
    """
    objective = _objective()
    attached = objective.attached(torch.device("cpu"))

    assert [id(parameter) for parameter in attached] == [id(objective.projection.weight)]
    assert attached[0].requires_grad


def test_settings() -> None:
    """The account reports this term's real strength beside the ranking family's own."""
    assert _objective(weight=20.0).settings()[FEATURE_TERM] == 20.0
    assert _objective().settings()[FEATURE_TERM] == 1.0


def test_call() -> None:
    """The label loss plus the cosine at its own weight, and the weight has to REACH the sum.

    Two weights over one batch: the difference between them must be the cosine scaled by the difference
    in weight, which no arrangement that dropped or double-counted the term can satisfy.
    """
    batch = _batch()
    weak = _objective(weight=1.0)
    strong = _objective(weight=3.0)
    cosine = float(weak.imitation(batch))

    assert float(strong(batch)) - float(weak(batch)) == pytest.approx(2.0 * cosine, abs=1e-5)
    assert cosine > 0.0, "a random projection against a random teacher is not already aligned"
