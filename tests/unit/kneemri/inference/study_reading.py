"""The shared reading loop, over stubs — the point is that it holds a member and a feed to a CONTRACT.

No `timm` and no cache here on purpose: what is worth asserting about this loop is the part both families
depend on, which is that a member is called once per row with one study, in the order asked for, and that
what comes back is a probability. A stub proves that; a real member would only prove `torch` works.
"""

import numpy as np
import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from core.findings import TARGETS
from kneemri.inference.study_reading import StudyReadings

STUDIES = 4
LOGIT = 2.0


class _Member:
    """A member that answers with a constant logit and records which studies it was handed."""

    def __init__(self) -> None:
        self.seen: list[float] = []

    def to(self, device: torch.device) -> "_Member":
        del device
        return self

    def __call__(self, batch: Float[Tensor, "..."]) -> Float[Tensor, "b n"]:
        self.seen.append(float(batch.flatten()[0].item()))
        return torch.full((1, len(TARGETS)), LOGIT)


class _Feed:
    """A feed whose study is its own row index, so a loop that reads the wrong row is visible."""

    def batch(self, row: int, device: torch.device) -> Float[Tensor, "..."]:
        return torch.full((1, 1), float(row), device=device)


def test_predict():
    """One forward per requested row, in the requested order, and every value is a probability."""
    member, feed = _Member(), _Feed()

    values = StudyReadings(torch.device("cpu"), torch.float32).predict(member, feed, [0, 2, 3])

    assert values.shape == (3, len(TARGETS))
    assert ((values >= 0.0) & (values <= 1.0)).all()
    assert member.seen == [0.0, 2.0, 3.0], "rows are read in the order given, and none is read twice"
    assert values[0, 0] == pytest.approx(1.0 / (1.0 + np.exp(-LOGIT))), "the loop's job is the sigmoid"


def test_contracts():
    """Both stubs satisfy the protocols the loop declares — the check a mistyped feed would fail."""
    assert isinstance(_Member(), StudyReadings.Member)
    assert isinstance(_Feed(), StudyReadings.Feed)
