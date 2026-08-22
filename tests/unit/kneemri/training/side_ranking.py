"""The within-study compartment term, on batches built to be exactly side-blind or exactly side-aware."""

import pytest
import torch

from core.findings import COMPARTMENT_PAIRS, TARGETS
from kneemri.training.grades import Predicted
from kneemri.training.side_ranking import PairedSides, SideRanking

STUDIES = 16
COLUMNS = {target: index for index, target in enumerate(TARGETS)}
BLIND_PENALTY = 0.6931
"""softplus(0): what a pair of columns emitting the same logit costs, which is the term's maximum."""


def _discordant() -> tuple[torch.Tensor, torch.Tensor]:
    """Half the studies positive on the lateral side of every pair, half on the medial side."""
    scores = torch.zeros(STUDIES, len(TARGETS))
    for medial, lateral in COMPARTMENT_PAIRS:
        scores[: STUDIES // 2, COLUMNS[lateral]] = 1.0
        scores[STUDIES // 2 :, COLUMNS[medial]] = 1.0
    return scores, torch.ones(STUDIES, len(TARGETS))


def test_paired_sides_term():
    """A pair emitting one signal twice pays the maximum; ordering the sides right pays ~nothing."""
    scores, weights = _discordant()
    blind = torch.zeros(STUDIES, len(TARGETS))
    aware = scores * 6.0 - 3.0

    term = PairedSides(1.0)

    assert float(term.term(Predicted(blind, scores, weights))) == pytest.approx(BLIND_PENALTY, abs=1e-3)
    assert float(term.term(Predicted(aware, scores, weights))) < 0.01
    # The weight scales the term rather than gating it, so an arm can be run at half strength.
    assert float(term.term(Predicted(blind, scores, weights))) == pytest.approx(
        2.0 * float(PairedSides(0.5).term(Predicted(blind, scores, weights)))
    )


def test_paired_sides_term_pushes_each_side_toward_its_own_column():
    """The gradient raises the side the reports grade higher and lowers its partner, symmetrically.

    The symmetry is the claim worth pinning: a term that only ever pushed the lateral column up would
    be a thumb on the scale for the label we happen to be weak on, and would read as an improvement on
    the contrast for a reason that has nothing to do with resolving the side.
    """
    scores, weights = _discordant()
    logits = torch.zeros(STUDIES, len(TARGETS), requires_grad=True)

    PairedSides(1.0).term(Predicted(logits, scores, weights)).backward()

    lateral_only = slice(None, STUDIES // 2)
    gradient = logits.grad
    assert gradient is not None
    for medial, lateral in COMPARTMENT_PAIRS:
        raises = gradient[lateral_only, COLUMNS[lateral]].mean()
        lowers = gradient[lateral_only, COLUMNS[medial]].mean()
        assert raises < 0.0, "descending must raise the column the reports grade higher"
        assert lowers == pytest.approx(-float(raises))
    assert gradient[:, COLUMNS["PF OA"]].abs().max() == 0.0, "a column in no pair is untouched"


def test_paired_sides_term_ignores_studies_the_reports_do_not_separate():
    """A study graded the same on both sides contributes nothing, and needs no mask to do so."""
    weights = torch.ones(STUDIES, len(TARGETS))
    agreed = torch.zeros(STUDIES, len(TARGETS))
    for medial, lateral in COMPARTMENT_PAIRS:
        agreed[:, COLUMNS[medial]] = 1.0
        agreed[:, COLUMNS[lateral]] = 1.0
    blind = torch.zeros(STUDIES, len(TARGETS))

    # Zero rather than the blind penalty: with no separating study the numerator is empty, and the
    # floor in the denominator must not turn that into a division that reports a misordering.
    assert float(PairedSides(1.0).term(Predicted(blind, agreed, weights))) == 0.0


def test_side_ranking_term():
    """The policy every shipped member trains under contributes a hard zero, not a small number."""
    scores, weights = _discordant()
    logits = torch.randn(STUDIES, len(TARGETS), generator=torch.Generator().manual_seed(0))

    assert float(SideRanking().term(Predicted(logits, scores, weights))) == 0.0


def test_paired_sides_settings():
    """The term answers a key naming its weight, so a run can check it landed rather than assume it."""
    assert PairedSides(0.4).settings() == {"side_weight": 0.4}


def test_side_ranking_settings():
    """The null policy answers the same key with a zero, so an arm without the term is legible too."""
    assert SideRanking().settings() == {"side_weight": 0.0}
