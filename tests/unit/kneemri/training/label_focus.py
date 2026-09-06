"""`LabelFocus` leans the loss on named findings by scaling every other column's weight.

Two policies share the object: the MASK (`unfocused` zero) and the UPWEIGHT (`unfocused` inside the
open unit interval). The upweight is the one the arithmetic does not make idempotent for free — 0.25
applied twice descends 0.0625 — so it is tested through the marker rather than through the values.
"""

import pytest
import torch

from core.findings import TARGETS
from kneemri.training.grades import Predicted
from kneemri.training.label_focus import FOCUS, UNFOCUSED, LabelFocus
from kneemri.training.objective import LabelStructure, Objective, RankedObjective, Shaping

LATERAL_MENISCUS = "Lateral Meniscus"
MCL = "MCL"
BATCH = 4
PARTIAL = 0.25


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
    focused = LabelFocus((LATERAL_MENISCUS,)).focused(batch())
    kept = TARGETS.index(LATERAL_MENISCUS)

    assert torch.equal(focused.weights[:, kept], torch.ones(BATCH))
    assert focused.weights.sum() == pytest.approx(float(BATCH))
    assert torch.equal(focused.logits, batch().logits)


def test_focused_null_policy_returns_the_batch() -> None:
    """No finding named means no mask at all — the object itself comes back, so an unfocused arm is unchanged."""
    unfocused = batch()

    assert LabelFocus().focused(unfocused) is unfocused


def test_focused_is_idempotent() -> None:
    """Applying the policy twice equals applying it once, which is what lets both objectives focus at their own entry.

    Asserted for the UPWEIGHT and not only for the mask. Zero and one square to themselves, so the mask
    was idempotent by arithmetic and a broken marker would pass unnoticed on it; 0.25 squares to 0.0625,
    which is a mechanism landing at the wrong strength with every gate green.
    """
    for policy in (LabelFocus((LATERAL_MENISCUS,)), LabelFocus((LATERAL_MENISCUS,), PARTIAL)):
        once = policy.focused(batch())

        assert torch.equal(policy.focused(once).weights, once.weights)
        assert once.shaped


def test_settings() -> None:
    """The account names WHICH finding was fitted, so a checkpoint can be checked for the one the arm asked for."""
    assert LabelFocus((LATERAL_MENISCUS,)).settings() == {FOCUS: LATERAL_MENISCUS, UNFOCUSED: 0.0}
    assert LabelFocus((MCL, LATERAL_MENISCUS), PARTIAL).settings() == {
        FOCUS: f"{MCL}+{LATERAL_MENISCUS}",
        UNFOCUSED: PARTIAL,
    }
    # The null policy accounts for a 0.0 strength whatever was passed: nothing is being leaned on.
    assert LabelFocus(unfocused=PARTIAL).settings() == {FOCUS: "", UNFOCUSED: 0.0}


def test_unknown_finding_is_rejected() -> None:
    """A mistyped name raises rather than falling through to the null policy and training the plain recipe."""
    with pytest.raises(ValueError, match="not among the twelve findings"):
        LabelFocus(("Lateral Meniscis",))


def test_ranking_term_is_narrowed_by_the_same_mask() -> None:
    """The pairwise term sees only the focused finding, which is what makes one mask enough for the whole loss.

    Read as an equality rather than as an inequality: a focused twelve-label batch must descend exactly
    what the same objective descends on a batch where the other eleven were never weighted at all.
    """
    structure = LabelStructure(focus=LabelFocus((LATERAL_MENISCUS,)))
    focused = RankedObjective(1.0, Shaping(structure=structure))(batch())
    hand_masked = RankedObjective(1.0)(LabelFocus((LATERAL_MENISCUS,)).focused(batch()))

    assert focused.item() == pytest.approx(hand_masked.item())


def test_focus_changes_the_scalar_it_descends() -> None:
    """Focusing is not a no-op on this data — a guard against a mask that silently never lands."""
    plain = Objective()(batch())
    focused = Objective(Shaping(structure=LabelStructure(focus=LabelFocus((LATERAL_MENISCUS,)))))(batch())

    assert focused.item() != pytest.approx(plain.item())


def test_focused_upweight_keeps_all_twelve_columns() -> None:
    """The named findings keep their full weight and the others are turned DOWN, never to zero.

    The distinction the arm rests on: `focus_mcl` starved eleven columns outright and collapsed to a
    0.5332 twelve-column mean, so what is being tested is that every column still carries gradient.
    """
    leaned = LabelFocus((MCL, LATERAL_MENISCUS), PARTIAL).focused(batch())
    named = [TARGETS.index(MCL), TARGETS.index(LATERAL_MENISCUS)]

    assert torch.equal(leaned.weights[:, named], torch.ones((BATCH, len(named))))
    assert (leaned.weights > 0.0).all()
    assert leaned.weights.sum() == pytest.approx(BATCH * (len(named) + PARTIAL * (len(TARGETS) - len(named))))


def test_parse() -> None:
    """A '+'-joined flag value becomes the policy it names, and an absent one the null policy."""
    assert LabelFocus.parse(f"{MCL}+{LATERAL_MENISCUS}", PARTIAL).findings == (MCL, LATERAL_MENISCUS)
    assert LabelFocus.parse(f"{MCL}+{LATERAL_MENISCUS}", PARTIAL).unfocused == PARTIAL
    assert LabelFocus.parse(None).findings == ()
    assert LabelFocus.parse("").findings == ()


def test_a_bare_string_is_rejected() -> None:
    """One finding passed unwrapped raises instead of iterating into characters and reading as unknown."""
    with pytest.raises(TypeError, match="not the string"):
        LabelFocus(LATERAL_MENISCUS)  # type: ignore[arg-type]
