import torch

from core.data.corpus.slots import SLOTS
from core.findings import TARGETS
from kneemri.evaluation.attribution_faithfulness import (
    PRESENT,
    Faithfulness,
    FaithfulnessReport,
    SlotFaithfulness,
)
from kneemri.evaluation.slot_probe import SlotProbe
from kneemri.models.slot_head import SlotHead

SEED = 2026
STUDIES = 4
DIM = 4
SAMPLES = 4
SENSITIVITY_SAMPLES = 2
ABSENT_SLOT = 3
"""One slot every fixture study is missing, so a perturbation has something it must not touch."""


def _head() -> SlotHead:
    torch.manual_seed(SEED)
    return SlotHead(DIM, len(SLOTS), len(TARGETS)).eval()


def _features() -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    return torch.randn(STUDIES, len(SLOTS), DIM)


def _mask() -> torch.Tensor:
    mask = torch.ones(STUDIES, len(SLOTS))
    mask[:, ABSENT_SLOT] = 0.0
    return mask


def _probe() -> SlotProbe:
    return SlotProbe(_head(), _features())


def _faithfulness() -> SlotFaithfulness:
    return SlotFaithfulness(samples=SAMPLES, sensitivity_samples=SENSITIVITY_SAMPLES)


def _row(ablation: float = 0.2, attention: float = 0.1) -> Faithfulness:
    return Faithfulness(
        finding=TARGETS[0],
        ablation_infidelity=ablation,
        attention_infidelity=attention,
        ablation_sensitivity=0.3,
        attention_sensitivity=0.4,
    )


def test_ablation() -> None:
    attribution = _faithfulness().ablation(_probe(), _mask(), 0)
    assert attribution.shape == (STUDIES, len(SLOTS))
    assert bool((attribution[:, ABSENT_SLOT] == 0.0).all())


def test_attention() -> None:
    attribution = SlotFaithfulness.attention(_probe(), _mask(), 0)
    assert attribution.shape == (STUDIES, len(SLOTS))
    assert bool((attribution[:, ABSENT_SLOT] < 1e-3).all())


def test_infidelity() -> None:
    probe, faithfulness = _probe(), _faithfulness()
    attribution = SlotFaithfulness.attention(probe, _mask(), 0)
    scored = faithfulness.infidelity(probe, _mask(), attribution, 0)
    assert scored.shape == (STUDIES,)
    assert bool((scored >= 0.0).all())


def test_sensitivity() -> None:
    scored = _faithfulness().sensitivity(_probe(), _mask(), 0, attend=True)
    assert scored.shape == (STUDIES,)
    assert bool((scored >= 0.0).all())


def test_measure() -> None:
    measured = _faithfulness().measure(_probe(), _mask(), 0)
    assert measured.finding == TARGETS[0]
    assert measured.ablation_infidelity >= 0.0
    assert measured.attention_sensitivity >= 0.0


def test_perturb() -> None:
    mask = _mask()
    perturbation, perturbed = _faithfulness().perturb(mask)
    assert torch.allclose(perturbation, mask - perturbed)
    assert bool((perturbed <= mask).all()), "an absent slot must never be switched ON by a perturbation"
    assert bool((perturbed.sum(dim=1) >= 1.0).all()), "a study must never be emptied of every slot"


def test_perturb_never_empties_a_single_slot_study() -> None:
    """The rescue is the case that matters: at DROP_RATE a one-slot study is emptied ~30% of the time."""
    torch.manual_seed(SEED)
    lone = torch.zeros(64, len(SLOTS))
    lone[:, 0] = 1.0
    _, perturbed = _faithfulness().perturb(lone)
    assert bool((perturbed[:, 0] >= PRESENT).all())


def test_line() -> None:
    line = _row().line()
    assert TARGETS[0] in line
    assert "infidelity" in line


def test_lines() -> None:
    lines = FaithfulnessReport([_row(), _row(ablation=0.05)]).lines()
    assert any("at least as faithful as ablation on 1 of 2" in line for line in lines)
    assert any("INTERACT" in line for line in lines)
