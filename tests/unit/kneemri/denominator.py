"""The refusal every check owes when it measured nothing."""

import logging

import pytest

from kneemri.denominator import Denominator


def test_refuse_empty(caplog: pytest.LogCaptureFixture):
    """Zero measured exits non-zero and says what was empty; anything measured returns quietly.

    The failure it removes is a check reporting "0 arms checked, 0 refused" and exiting 0 — the same
    sentence and the same exit code a clean tree produces, from a tool that read nothing.
    """
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as refusal:
        Denominator(0, "no arm printed an invocation").refuse_empty()

    assert refusal.value.code == 1
    assert "no arm printed an invocation" in caplog.text, "the refusal names the empty set, not the pass"

    caplog.clear()
    Denominator(1, "no arm printed an invocation").refuse_empty()
    assert caplog.text == "", "one thing measured is a check that ran"
