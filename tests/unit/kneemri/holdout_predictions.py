"""Loading a member from a checkpoint and reading it over test-time windows, without a network.

The backbone is injected precisely so these can run: an encoder fetched inside `load` would make both
methods require a download, and a method that cannot be constructed in a test is a method whose contract
nothing checks.
"""

from pathlib import Path
from typing import override

import torch

from kneemri.holdout_predictions import GROUP, HoldoutPredictions, Window
from kneemri.model import TARGETS, StudyDesign, StudyModel

DIM = 12
"""At most SIZE: the double reads one feature per image ROW, so a wider feature than the image has
rows cannot be filled."""
SLOTS = 6
SLICES = 6
SIZE = 14
SEED = 2026


class _FakeBackbone(torch.nn.Module):
    """A stand-in encoder whose tokens vary across the feature dimension.

    Varying per dimension is the load-bearing part. An earlier double returned the same scalar in every
    feature, which the head's opening LayerNorm removes exactly, so its output did not depend on its
    pixels and no test using it could detect anything.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(DIM, DIM) for _ in range(2)])

    @override
    def forward(self, pixel_values: torch.Tensor) -> object:
        rows = pixel_values.mean(dim=(1, 3))[:, :DIM]
        tokens = rows[:, None, :].expand(-1, 5, -1) * torch.linspace(0.5, 1.5, DIM)
        return type("Output", (), {"last_hidden_state": tokens})()


def _member() -> StudyModel:
    torch.manual_seed(SEED)
    return StudyModel(_FakeBackbone(), DIM, SLOTS)


def test_load(tmp_path: Path):
    """Weights round-trip, and the pooling rule comes from the checkpoint rather than a default."""
    saved = _member()
    checkpoint = tmp_path / "member.pt"
    torch.save({"model": saved.state_dict(), "config": {"slice_pooling": "channels"}}, checkpoint)

    torch.manual_seed(SEED)
    loaded = HoldoutPredictions(torch.device("cpu")).load(checkpoint, _FakeBackbone(), DIM, SLOTS)

    images = torch.randint(0, 256, (2, SLOTS, GROUP, SIZE, SIZE), dtype=torch.uint8)
    mask = torch.ones(2, SLOTS)
    with torch.no_grad():
        assert torch.allclose(loaded.forward(images, mask, SIZE), saved.eval().forward(images, mask, SIZE))


def test_load_honours_a_checkpoint_trained_with_per_slice_pooling(tmp_path: Path):
    """A member saved as `max` must not be rebuilt as `channels` — they are different functions."""
    torch.manual_seed(SEED)
    saved = StudyModel(_FakeBackbone(), DIM, SLOTS, StudyDesign(slice_pooling="max"))
    checkpoint = tmp_path / "max.pt"
    torch.save({"model": saved.state_dict(), "config": {"slice_pooling": "max"}}, checkpoint)

    torch.manual_seed(SEED)
    loaded = HoldoutPredictions(torch.device("cpu")).load(checkpoint, _FakeBackbone(), DIM, SLOTS)
    assert loaded.slice_pooling == "max"


def test_predict():
    """Probabilities for every study, in [0, 1], one row per study and one column per target."""
    predictions = HoldoutPredictions(torch.device("cpu"))
    generator = torch.Generator().manual_seed(SEED)
    images = torch.randint(0, 256, (5, SLOTS, SLICES, SIZE, SIZE), generator=generator, dtype=torch.uint8).numpy()
    mask = torch.ones(5, SLOTS).numpy()

    scored = predictions.predict(_member().eval(), images, mask, Window(starts=[0, 3], width=GROUP, size=SIZE))

    assert scored.shape == (5, len(TARGETS))
    assert scored.min() >= 0.0
    assert scored.max() <= 1.0


def test_predict_over_one_window_matches_a_single_forward():
    """With a single window there is nothing to average, so it must equal the model's own sigmoid."""
    model = _member().eval()
    generator = torch.Generator().manual_seed(SEED)
    images = torch.randint(0, 256, (3, SLOTS, SLICES, SIZE, SIZE), generator=generator, dtype=torch.uint8)
    mask = torch.ones(3, SLOTS)

    scored = HoldoutPredictions(torch.device("cpu")).predict(
        model, images.numpy(), mask.numpy(), Window(starts=[0], width=GROUP, size=SIZE)
    )

    with torch.no_grad():
        direct = torch.sigmoid(model.forward(images[:, :, 0:GROUP], mask, SIZE))
    assert torch.allclose(torch.from_numpy(scored).float(), direct, atol=1e-5)
