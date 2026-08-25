"""`LabelFocus` narrows the loss to one finding by zeroing every other column's weight."""

import pytest
import torch

from core.findings import TARGETS
from kneemri.training.grades import Predicted
from kneemri.training.label_focus import FOCUS, LabelFocus
from kneemri.training.objective import LabelStructure, Objective, RankedObjective, Shaping

LATERAL_MENISCUS = "Lateral Meniscus"
BATCH = 4


def batch() -> Predicted:
    """A readout whose weights are all one, so a zeroed column is visible as a zero rather than a small number."""
    generator = torch.Generator().manual_seed(2026)
    shape = (BATCH, len(TARGETS))
    return Predicted(
        logits=torch.randn(shape, generator=generator),
        scores=torch.rand(shape, generator=generator),
        weights=torch.ones(shape),
    )


def test_focused() -> None:
    """The focused finding keeps its weight and the other eleven are zeroed, values untouched."""
    focused = LabelFocus(LATERAL_MENISCUS).focused(batch())
    kept = TARGETS.index(LATERAL_MENISCUS)

    assert torch.equal(focused.weights[:, kept], torch.ones(BATCH))
    assert focused.weights.sum() == pytest.approx(float(BATCH))
    assert torch.equal(focused.logits, batch().logits)


def test_focused_null_policy_returns_the_batch() -> None:
    """No finding named means no mask at all — the object itself comes back, so an unfocused arm is unchanged."""
    unfocused = batch()

    assert LabelFocus().focused(unfocused) is unfocused


def test_focused_is_idempotent() -> None:
    """Applying the mask twice equals applying it once, which is what lets both objectives focus at their own entry."""
    policy = LabelFocus(LATERAL_MENISCUS)
    once = policy.focused(batch())

    assert torch.equal(policy.focused(once).weights, once.weights)


def test_settings() -> None:
    """The account names WHICH finding was fitted, so a checkpoint can be checked for the one the arm asked for."""
    assert LabelFocus(LATERAL_MENISCUS).settings() == {FOCUS: LATERAL_MENISCUS}
    assert LabelFocus().settings() == {FOCUS: ""}


def test_unknown_finding_is_rejected() -> None:
    """A mistyped name raises rather than falling through to the null policy and training the plain recipe."""
    with pytest.raises(ValueError, match="not one of the twelve findings"):
        LabelFocus("Lateral Meniscis")


def test_ranking_term_is_narrowed_by_the_same_mask() -> None:
    """The pairwise term sees only the focused finding, which is what makes one mask enough for the whole loss.

    Read as an equality rather than as an inequality: a focused twelve-label batch must descend exactly
    what the same objective descends on a batch where the other eleven were never weighted at all.
    """
    structure = LabelStructure(focus=LabelFocus(LATERAL_MENISCUS))
    focused = RankedObjective(1.0, Shaping(structure=structure))(batch())
    hand_masked = RankedObjective(1.0)(LabelFocus(LATERAL_MENISCUS).focused(batch()))

    assert focused.item() == pytest.approx(hand_masked.item())


def test_focus_changes_the_scalar_it_descends() -> None:
    """Focusing is not a no-op on this data — a guard against a mask that silently never lands."""
    plain = Objective()(batch())
    focused = Objective(Shaping(structure=LabelStructure(focus=LabelFocus(LATERAL_MENISCUS))))(batch())

    assert focused.item() != pytest.approx(plain.item())
