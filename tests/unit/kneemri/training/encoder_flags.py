"""The trunk flags moved out of the vocabulary; the test states that the parser still accepts them."""

import argparse

from kneemri.models.backbone import SMALL
from kneemri.training.encoder_flags import EncoderFlags


def test_add() -> None:
    """Every trunk flag reaches argv, and the encoder variant still defaults to the committed one."""
    parser = argparse.ArgumentParser()
    EncoderFlags.add(parser)
    parsed = parser.parse_args(["--unfreeze-last", "4", "--through-plane"])
    assert parsed.variant == SMALL
    assert parsed.unfreeze_last == 4
    assert parsed.through_plane
