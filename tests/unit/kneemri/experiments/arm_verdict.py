import logging

import pytest

from kneemri.experiments.arm_verdict import ArmVerdict


def test_report(caplog: pytest.LogCaptureFixture):
    """The ASYMMETRY, which is the whole design: every finding by name while arming, a count otherwise.

    Seventy arms predate whichever rule an advisory check enforces and nearly all of them already ran, so
    naming each on every sweep buries the REFUSED and UNCACHED lines that cost a queue slot — which is how
    a pre-flight check earns a `--no-verify`. Partitioned on the three states an author can put it in:
    arming, sweeping, and the `always` checks whose every finding costs a slot either way. The RETURN is
    the second axis and a separate question from the voice: a finding is reported either way, and whether
    it refuses this run is the verdict's own policy.
    """
    verdict = ArmVerdict(
        "UNSTATED",
        {"silent": "says nothing about --slices"},
        clause="name none of them",
        bead="k-1",
        arming_refuses=True,
    )

    with caplog.at_level(logging.INFO):
        assert verdict.report(70, arming=True) is True, "an advisory rule refuses the run that ARMS an arm"

    assert caplog.messages == ["UNSTATED silent: says nothing about --slices (k-1)"]
    caplog.clear()

    with caplog.at_level(logging.INFO):
        assert verdict.report(70, arming=False) is False, "and only that run — the sweep still SAYS it"

    assert caplog.messages == ["UNSTATED: 1 of 70 arms name none of them; --only <arm> names them (k-1)"]
    caplog.clear()

    with caplog.at_level(logging.INFO):
        named = ArmVerdict("REFUSED", {"typo": "unrecognised --fld"}, always=True, refuses=True).report(
            70, arming=False
        )
        advisory = ArmVerdict("OVERSTATED", {"fat": "asks 10 GB over"}, clause="ask over").report(70, arming=True)
        empty = ArmVerdict("UNSTATED", {}, clause="name none of them", refuses=True).report(70, arming=False)

    assert (named, advisory, empty) == (True, False, False), (
        "a finding that costs a queue slot refuses on any run; one that costs only wall-clock refuses none; "
        "and a check that found nothing cannot refuse whatever its policy says"
    )
    assert caplog.messages == [
        "REFUSED typo: unrecognised --fld",
        "OVERSTATED fat: asks 10 GB over",
    ], "a finding that costs a queue slot is named on every sweep, and an empty check says nothing at all"
