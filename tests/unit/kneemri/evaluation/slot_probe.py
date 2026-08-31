import torch

from core.data.corpus.slots import SLOTS
from core.findings import TARGETS
from kneemri.evaluation.slot_probe import SlotProbe
from kneemri.models.slot_head import SlotHead

SEED = 2026
STUDIES = 4
DIM = 4
ABSENT_SLOT = 3
"""One slot every fixture study is missing, so the masked softmax has something to renormalise away."""


def _features() -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    return torch.randn(STUDIES, len(SLOTS), DIM)


def _mask() -> torch.Tensor:
    mask = torch.ones(STUDIES, len(SLOTS))
    mask[:, ABSENT_SLOT] = 0.0
    return mask


def _probe() -> SlotProbe:
    torch.manual_seed(SEED)
    return SlotProbe(SlotHead(DIM, len(SLOTS), len(TARGETS)).eval(), _features())


def test_probabilities() -> None:
    probabilities = _probe().probabilities(_mask())
    assert probabilities.shape == (STUDIES, len(TARGETS))
    assert bool(((probabilities > 0.0) & (probabilities < 1.0)).all())


def test_probabilities_tile_the_cached_features_for_a_stacked_batch() -> None:
    """captum stacks perturbations into one batch; the trunk features must follow, not broadcast."""
    probe = _probe()
    stacked = probe.probabilities(_mask().repeat(3, 1))
    assert stacked.shape == (3 * STUDIES, len(TARGETS))
    assert torch.allclose(stacked[:STUDIES], probe.probabilities(_mask()))


def test_weights() -> None:
    weights = _probe().weights(_mask())
    assert weights.shape == (STUDIES, len(TARGETS), len(SLOTS))
    assert torch.allclose(weights.sum(-1), torch.ones(STUDIES, len(TARGETS)), atol=1e-4)
    assert bool((weights[:, :, ABSENT_SLOT] < 1e-3).all()), "an absent slot must leave the softmax"


def test_tiled() -> None:
    features = _features()
    assert SlotProbe.tiled(features, STUDIES) is features
    assert SlotProbe.tiled(features, 2 * STUDIES).shape == (2 * STUDIES, len(SLOTS), DIM)
