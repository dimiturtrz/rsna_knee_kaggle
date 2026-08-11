"""One real study through the real pipeline, ending in the columns Kaggle is handed.

Everything below this layer runs on stand-ins: a fake encoder, synthetic pixels, a hand-built label
table. That is deliberate and it is fast, but it means no test in this repository has ever asserted that
the actual cached corpus, an actual trained member and the actual target order agree with each other.

The failure this guards is the one the fingerprint exists for and the unit layer cannot see: a member
loads whenever the shapes line up, and shapes line up across a changed normalisation, a changed slice
band, a changed crop and a changed slot order. Such a run completes and writes a plausible submission.

Skipped rather than failed when the data root or a member is absent, so CI stays green on a machine that
has neither — but skipping is reported loudly enough that "it passed" is never mistaken for "it ran".
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from core.paths import DataRoot
from kneemri.backbone import ENCODERS, SMALL
from kneemri.ensemble import Ensemble
from kneemri.holdout_predictions import GROUP, HoldoutPredictions
from kneemri.model import TARGETS
from kneemri.slot_reader import IMAGE_SIZE
from kneemri.slots import SLOTS
from kneemri.study_cache import StudyCache

STUDIES = 4
PATHS = Path("paths.yaml")


def _root() -> DataRoot:
    """The configured data root, or a skip when this machine has no data."""
    if not PATHS.is_file():
        pytest.skip(f"{PATHS} absent — no data root configured on this machine")
    root = DataRoot.from_config(PATHS)
    if not root.processed().is_dir():
        pytest.skip(f"{root.processed()} absent — corpus not built on this machine")
    return root


def _member(root: DataRoot) -> Path:
    """Any trained member on this machine, or a skip."""
    members = sorted(root.processed().glob("*_fold0.pt"))
    if not members:
        pytest.skip("no trained member on this machine")
    return members[0]


def test_a_real_study_becomes_a_submission_row():
    """Cache, member, prediction and target order all agree on real data."""
    root = _root()
    corpus, identifiers = StudyCache(root.processed(), slices=12).load()
    assert len(identifiers) > STUDIES, "the cached corpus is smaller than this smoke test's sample"

    checkpoint = _member(root)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoder = ENCODERS[str(saved.get("config", {}).get("variant", SMALL))]

    from transformers import AutoModel  # noqa: PLC0415 - a 90 MB import that only this test needs

    predictions = HoldoutPredictions(torch.device("cpu"))
    model = predictions.load(checkpoint, AutoModel.from_pretrained(encoder.repository), encoder.dim, len(SLOTS))

    rows = np.arange(STUDIES)
    scored = predictions.predict(model, corpus.images[rows], corpus.mask[rows], [0], IMAGE_SIZE)

    assert scored.shape == (STUDIES, len(TARGETS))
    assert np.isfinite(scored).all(), "a non-finite probability would be written straight into a submission"
    assert scored.min() >= 0.0
    assert scored.max() <= 1.0

    table = Ensemble.submission(list(identifiers[:STUDIES]), scored)
    assert list(table) == ["StudyInstanceUID", *TARGETS]
    assert len(table["StudyInstanceUID"]) == STUDIES


def test_the_cached_corpus_has_the_shape_the_model_expects():
    """Slot count, group size and image size are agreed between the cache and the model, not assumed.

    Each of these is a number written in two places, and a mismatch does not raise — it silently feeds
    the encoder something it never trained on, which is exactly how 32-slice inference cost 0.018.
    """
    root = _root()
    corpus, _ = StudyCache(root.processed(), slices=12).load()

    assert corpus.images.shape[1] == len(SLOTS)
    assert corpus.images.shape[2] >= GROUP
    assert corpus.images.shape[-1] == IMAGE_SIZE
    assert corpus.mask.shape[1] == len(SLOTS)
