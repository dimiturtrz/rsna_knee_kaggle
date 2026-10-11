"""The anchor: it lands exactly as often as declared, or it refuses."""

import pytest

from kneemri.kaggle.script_seat import Anchor


def test_applied() -> None:
    """One site is rewritten; none or two are refused; a declared two-site anchor rewrites both."""
    text = "a = 1\nb = 2\nb = 2\n"
    assert Anchor("a = 1", "a = 9").applied(text, "src") == "a = 9\nb = 2\nb = 2\n"
    assert Anchor("b = 2", "b = 7", sites=2).applied(text, "src") == "a = 1\nb = 7\nb = 7\n"
    with pytest.raises(ValueError, match=r"src spells 'c = 3' 0 times, not 1"):
        Anchor("c = 3", "c = 4").applied(text, "src")
    with pytest.raises(ValueError, match="2 times, not 1"):
        Anchor("b = 2", "b = 7").applied(text, "src")
