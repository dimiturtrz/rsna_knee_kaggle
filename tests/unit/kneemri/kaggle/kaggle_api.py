import sys
import types

import pytest

from kneemri.kaggle.kaggle_api import Kaggle


class FakeApi:
    def __init__(self) -> None:
        self.signed_in = False

    def authenticate(self) -> None:
        self.signed_in = True


def test_authenticated(monkeypatch: pytest.MonkeyPatch) -> None:
    extended = types.ModuleType("kaggle.api.kaggle_api_extended")
    monkeypatch.setattr(extended, "KaggleApi", FakeApi, raising=False)
    monkeypatch.setitem(sys.modules, "kaggle.api.kaggle_api_extended", extended)
    api = Kaggle.authenticated()
    assert isinstance(api, FakeApi)
    assert api.signed_in
