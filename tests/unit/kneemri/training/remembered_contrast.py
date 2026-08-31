"""Cross-step contrasting: the equality oracle first, then the positive no single batch contains."""

import torch

from kneemri.training.grades import GRADE_LADDER, Predicted
from kneemri.training.label_focus import FOCUS
from kneemri.training.objective import (
    CONTRAST_TERM,
    COUNT_NORMALISED,
    DISSENT_TERM,
    DISTILL_TERM,
    FEATURE_TERM,
    MARGIN_TERM,
    PAIR_POOL,
    PER_LABEL,
    RANK_TERM,
    ContrastiveObjective,
)
from kneemri.training.remembered_contrast import RememberedContrast
from kneemri.training.side_ranking import SIDE_TERM


def _embedded(directions: list[list[float]], positive: list[float]) -> Predicted:
    """One finding's worth of embeddings beside the labels that say which of them belong together."""
    embedding = torch.tensor(directions).unsqueeze(1)
    scores = torch.tensor(positive).reshape(-1, 1)
    logits = torch.zeros_like(scores)
    return Predicted(logits, scores, torch.ones_like(scores), embedding=embedding)


def test_settings():
    """The pool is read off the memory, so a run's `--pair-memory` is checked against what would drop."""
    assert RememberedContrast(0.3, capacity=4).settings() == {
        RANK_TERM: 0.0,
        DISTILL_TERM: 0.0,
        FEATURE_TERM: 0.0,
        DISSENT_TERM: 0.0,
        CONTRAST_TERM: 0.3,
        MARGIN_TERM: 0.0,
        PAIR_POOL: 4,
        PER_LABEL: False,
        COUNT_NORMALISED: False,
        GRADE_LADDER: False,
        SIDE_TERM: 0.0,
        FOCUS: "",
    }


def test_contrast():
    """The empty memory is the equality oracle: with nothing remembered this MUST be the old term.

    Any difference on the first step is the new path changing the loss for a run that has remembered
    nothing yet, which would make every comparison against `contrast_dense`'s measured control a
    comparison of two things at once.
    """
    batch = _embedded([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], [1.0, 1.0, 0.0, 0.0])

    assert torch.equal(RememberedContrast().contrast(batch), ContrastiveObjective().contrast(batch))


def test_contrast_reaches_a_positive_no_single_batch_holds():
    """The whole point: at 25% prevalence a batch of two sees one torn study and forms no term at all.

    Each batch here holds exactly one positive, so within a batch that anchor has no partner, drops out,
    and the plain term is left with the negatives alone. Remembering the earlier step gives the second
    batch's positive a partner it was never batched with — which is the small-batch weakness
    `contrast_dense` pre-registered as the suspect against its own reading.
    """
    early = _embedded([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0])
    late = _embedded([[-1.0, 0.0], [0.0, 1.0]], [1.0, 0.0])
    objective = RememberedContrast()

    objective.contrast(early)
    across = float(objective.contrast(late))
    within = float(ContrastiveObjective().contrast(late))

    assert across > within, "the remembered positive disagrees with the live one and must be charged for"


def test_widened():
    """The live batch FIRST and the memory after it, which is what `supcon`'s self-exclusion rests on."""
    batch = _embedded([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0])
    remembered = _embedded([[0.5, 0.5]], [1.0])

    pool = RememberedContrast().widened(batch, remembered)

    assert pool.embedding is not None
    assert pool.embedding.shape[0] == 3
    assert torch.equal(pool.embedding[:2], batch.embedding), "the anchors lead, so the diagonal is the self-match"
    assert torch.equal(pool.scores[:2], batch.scores)
    assert pool.weights.shape[0] == 3, "every column of a readout has to agree in length"


def test_joined():
    """Both contexts as one, and None the moment either side lacks one — never the live half alone."""
    batch = _embedded([[1.0, 0.0], [0.0, 1.0]], [1.0, 0.0])
    remembered = _embedded([[0.5, 0.5]], [1.0])
    objective = RememberedContrast()

    assert objective.joined(batch, remembered).shape[0] == 3
    bare = Predicted(batch.logits, batch.scores, batch.weights)
    assert objective.joined(batch, bare) is None, "a pool with no context must raise downstream, not narrow"
    assert objective.joined(bare, remembered) is None
