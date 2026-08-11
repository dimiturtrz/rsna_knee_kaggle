from typing import override

import numpy as np
import torch
from torch import Tensor, nn

from kneemri.report_embeddings import ReportEmbeddings

SEED = 2026
WIDTH = 8
REPORTS = ["rotura de menisco interno", "medial meniscal tear", "no acute abnormality", "derrame articular"]


class _FakeTokenizer:
    """Turns each report into token ids by word count, with a mask that marks the padding."""

    def __call__(
        self, batch: list[str], *, padding: bool, truncation: bool, max_length: int, return_tensors: str
    ) -> dict[str, Tensor]:
        lengths = [len(text.split()) for text in batch]
        widest = max(lengths)
        ids = torch.zeros(len(batch), widest, dtype=torch.long)
        mask = torch.zeros(len(batch), widest, dtype=torch.long)
        for row, length in enumerate(lengths):
            ids[row, :length] = torch.arange(1, length + 1)
            mask[row, :length] = 1
        return {"input_ids": ids, "attention_mask": mask}


class _FakeTextEncoder(nn.Module):
    """States that depend on the token id, so pooling over padding is detectable."""

    @override
    def forward(self, *, input_ids: Tensor, attention_mask: Tensor) -> object:
        states = input_ids.unsqueeze(-1).float() * torch.linspace(0.1, 1.0, WIDTH)
        return type("Output", (), {"last_hidden_state": states})()


def test_encode():
    """Every report becomes one unit vector, and padding does not contribute to it.

    Pooling over the padded LENGTH rather than the mask would drag short reports toward a common
    vector - exactly the reports whose findings are shortest and most decisive.
    """
    embeddings = ReportEmbeddings(torch.device("cpu"))
    vectors = embeddings.encode(REPORTS, _FakeTokenizer(), _FakeTextEncoder())

    assert vectors.shape == (len(REPORTS), WIDTH)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    alone = embeddings.encode([REPORTS[2]], _FakeTokenizer(), _FakeTextEncoder())
    assert np.allclose(alone[0], vectors[2], atol=1e-5), "a report's vector must not depend on its batch mates"


def test_reduce():
    """The reduction keeps the leading components and lands on an average unit norm."""
    generator = np.random.default_rng(SEED)
    vectors = generator.normal(size=(30, WIDTH)).astype(np.float32)

    reduced = ReportEmbeddings(torch.device("cpu")).reduce(vectors, dims=3)
    assert reduced.shape == (30, 3)
    assert abs(float(np.linalg.norm(reduced, axis=1).mean()) - 1.0) < 1e-5

    spread = np.linalg.norm(reduced, axis=1)
    assert spread.std() > 0.05, "per-row normalisation would make every study the same distance out"


def test_reduce_keeps_the_directions_the_corpus_actually_varies_along():
    """A dimension the reports do not vary in must not survive the reduction.

    The components are fitted on OUR reports rather than taken from the encoder, so a constant
    direction carries no information and has to be dropped in favour of one that does.
    """
    generator = np.random.default_rng(SEED)
    vectors = np.zeros((40, WIDTH), dtype=np.float32)
    vectors[:, 0] = generator.normal(size=40)
    vectors[:, 1] = generator.normal(size=40) * 0.01
    vectors[:, 2] = 7.0

    reduced = ReportEmbeddings(torch.device("cpu")).reduce(vectors, dims=1)
    ranking = np.abs(np.corrcoef(reduced[:, 0], vectors[:, 0])[0, 1])
    assert ranking > 0.99, "the leading component must follow the dimension with the most variance"
