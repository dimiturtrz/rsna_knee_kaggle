import pytest
import torch

from kneemri.device import VisibleCards


def test_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """A visible card is used; CUDA installed but with NO visible card is CPU, not a broken cuda device.

    The second case is the whole reason this class exists, and it is not hypothetical: the card here is
    handed to another project with `CUDA_VISIBLE_DEVICES=`, under which `is_available()` stays true and
    `device_count()` is 0. Every call site that asked the first question selected a device that owns no
    card and died on `AssertionError: Invalid device id`.
    """
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert VisibleCards().device().type == "cuda"

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert VisibleCards().device().type == "cpu", "no visible card is CPU, whatever availability claims"
