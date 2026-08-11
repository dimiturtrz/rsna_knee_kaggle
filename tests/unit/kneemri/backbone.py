"""The encoder table: one variant, one repository, one width."""

from kneemri.backbone import BASE, ENCODERS, SMALL


def test_mounted_name():
    """A kernel has no internet, so each variant must name the Kaggle model it is mounted from."""
    assert ENCODERS[SMALL].mounted_name == "metaresearch/dinov2/PyTorch/small/1"


def test_every_variant_states_its_own_width():
    """The width follows from the variant — it is what three separate constants used to disagree about."""
    assert ENCODERS[SMALL].dim == 384
    assert ENCODERS[BASE].dim == 768
    assert {name: encoder.variant for name, encoder in ENCODERS.items()} == {SMALL: SMALL, BASE: BASE}
