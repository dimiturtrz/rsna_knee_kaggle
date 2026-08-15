import numpy as np
import torch

from kneemri.labels.targets import Targets
from kneemri.models.model import TARGETS
from kneemri.training.head_trial import BATCH, HeadTrial

SEED = 2026
STUDIES = 200
SLOTS = 6
SLICES = 4
WIDTH = 16


def _corpus() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(SEED)
    features = generator.normal(0, 1, (STUDIES, SLOTS, SLICES, WIDTH)).astype(np.float16)
    mask = np.ones((STUDIES, SLOTS), dtype=np.float32)
    mask[:, -1] = 0.0
    return features, mask


def test_pooled():
    """Slices reduce by the elementwise maximum, and an absent slot is ZEROED rather than left as-is.

    The head's softmax reads the mask, so a stale vector sitting behind a zero mask is a value waiting
    to be used by accident — the shape this project has now found ten times.
    """
    features, mask = _corpus()

    pooled = HeadTrial().pooled(features, mask)

    assert pooled.shape == (STUDIES, SLOTS, WIDTH)
    expected = torch.from_numpy(features.astype(np.float32)).amax(dim=2)
    assert torch.allclose(pooled[:, 0], expected[:, 0], atol=1e-3), "a present slot is the max over its slices"
    assert torch.all(pooled[:, -1] == 0.0), "an absent slot carries no features at all"


def test_run():
    """A head fitted on features that CARRY the label beats one fitted on noise.

    The point of the harness is that a head experiment is answerable in seconds. If a signal planted in
    the features cannot be recovered, the harness cannot rank head variants and is not worth trusting.
    """
    generator = np.random.default_rng(SEED)
    truth = (generator.random((STUDIES, len(TARGETS))) > 0.5).astype(np.float32)
    mask = np.ones((STUDIES, SLOTS), dtype=np.float32)
    signal = np.zeros((STUDIES, SLOTS, SLICES, WIDTH), dtype=np.float32)
    signal[:, 0, 0, : len(TARGETS)] = truth * 4.0 - 2.0
    noise = generator.normal(0, 1, signal.shape).astype(np.float32)

    targets = Targets(truth, np.ones_like(truth), [str(i) for i in range(STUDIES)])
    split = (np.arange(0, 160, dtype=np.int64), np.arange(160, STUDIES, dtype=np.int64))
    trial = HeadTrial(epochs=6, rank_weight=0.0)

    carried = trial.run(trial.pooled(signal.astype(np.float16), mask), torch.from_numpy(mask), targets, split)
    unrelated = trial.run(trial.pooled(noise.astype(np.float16), mask), torch.from_numpy(mask), targets, split)

    assert carried > unrelated, "a planted signal must be recoverable, or the harness ranks nothing"
    assert carried > 0.7, "and recoverable well, not merely better than noise"


def test_batches():
    """Batches are shuffled and the partial last one is dropped, as the pixel trainer does.

    Dropped for the same reason: the ranking term scores pairs WITHIN a batch, so a short final batch
    steps on a different number of pairs than every other one.
    """
    studies = np.arange(BATCH * 2 + 7, dtype=np.int64)

    batches = HeadTrial()._batches(studies, np.random.default_rng(SEED))

    assert len(batches) == 2, "the remainder of seven is dropped, not padded"
    assert all(len(batch) == BATCH for batch in batches)
    assert set(np.concatenate(batches).tolist()) <= set(studies.tolist())
