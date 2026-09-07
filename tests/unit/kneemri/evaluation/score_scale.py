import pytest

from kneemri.evaluation.score_scale import RAW, SCALES, ProbitScale, RawScale


def test_raw_scale_of():
    """The identity, which is what makes raw AUC the reading every transfer constant was fitted on."""
    assert RawScale().of(0.944) == pytest.approx(0.944)


def test_raw_scale_back():
    """The identity again, so a raw requirement is already quoted in AUC."""
    assert RawScale().back(0.944) == pytest.approx(0.944)


def test_raw_scale_delta():
    """The plain difference — the quantity every compression ratio this project owns divides."""
    assert RawScale().delta(0.9530, 0.9440) == pytest.approx(0.0090)


def test_raw_scale_advance():
    """A climb of raw points lands raw points higher."""
    assert RawScale().advance(0.8751, 0.0092) == pytest.approx(0.8843)


def test_probit_scale_of():
    """The normal quantile: the median maps to zero and 0.9772 to two sigma."""
    assert ProbitScale().of(0.5) == pytest.approx(0.0)
    assert ProbitScale().of(0.9772) == pytest.approx(2.0, abs=1e-3)


def test_probit_scale_back():
    """Round-trips, or a requirement priced on this scale cannot be quoted in AUC again."""
    assert ProbitScale().back(ProbitScale().of(0.8751)) == pytest.approx(0.8751)


def test_probit_scale_delta():
    """The gap widens near the ceiling — the whole reason the scale exists.

    The same 0.009 of AUC is a larger separation at 0.944 than at 0.860, which is what a raw ratio
    near a bounded metric's edge cannot express and reports as a falling slope instead.
    """
    high = ProbitScale().delta(0.9530, 0.9440)
    low = ProbitScale().delta(0.8690, 0.8600)
    assert high > low


def test_probit_scale_advance():
    """A climb on the quantile scale comes back as the AUC that reaches it."""
    scale = ProbitScale()
    assert scale.advance(0.8751, scale.delta(0.8843, 0.8751)) == pytest.approx(0.8843)


def test_probit_scale_of_clamps_a_saturated_score():
    """A column dragged to exactly 1.0 becomes a finite number, not an infinity inside a median."""
    assert ProbitScale().of(1.0) < float("inf")


def test_scales_is_the_closed_vocabulary_a_flag_reads():
    """An unknown scale name must refuse rather than silently read raw, so the flag reads from here."""
    assert sorted(SCALES) == ["probit", "raw"]
    assert SCALES["raw"] is RAW
