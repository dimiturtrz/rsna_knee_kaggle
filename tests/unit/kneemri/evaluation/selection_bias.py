"""The measured search-selection bias, read as the window a raw score may honestly be recorded in."""

from kneemri.evaluation.selection_bias import SELECTION_BIAS, SelectionBias

RAW = 0.8731
"""`peak8`'s own in-sample greedy score — the row this constant was written for."""


def test_corrects():
    """A correction inside the measured band, and both ways of falling outside it."""
    bias = SelectionBias()

    assert bias.corrects(RAW, 0.8704), "peak8's recorded 0.8704 is a correction of +0.0027"
    assert bias.corrects(0.8768, 0.8740), "greedyseat_v1's +0.0028, measured on the board at +0.0027"

    assert not bias.corrects(RAW, RAW), "the raw score uncorrected is the defect this exists for"
    assert not bias.corrects(RAW, RAW - SELECTION_BIAS[0] / 2), "half the smallest measurement is not one"
    assert not bias.corrects(RAW, RAW - 2 * SELECTION_BIAS[1]), (
        "an over-correction buys a better transfer residual and is refused as loudly as a raw score"
    )

    assert bias.corrects(RAW, RAW - SELECTION_BIAS[0]), "the band is closed at both ends"
    assert bias.corrects(RAW, RAW - SELECTION_BIAS[1])


def test_stated():
    """The two numbers a writer needs, ordered as a window rather than as band endpoints."""
    low, high = SELECTION_BIAS
    stated = SelectionBias().stated(RAW)

    assert stated == f"{RAW - high:.4f} to {RAW - low:.4f}"
    assert stated.startswith("0.8700"), "the larger correction gives the SMALLER honest score, and comes first"
