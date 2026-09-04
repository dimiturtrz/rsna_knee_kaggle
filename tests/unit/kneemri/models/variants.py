"""The vocabulary has to survive being typed on a command line, and it has to be what the rows use."""

from kneemri.models import variants
from kneemri.models.backbone import ENCODERS

NAMES = [value for name, value in vars(variants).items() if name.isupper() and isinstance(value, str)]


def test_every_name_is_a_single_lowercase_token():
    """These are `--variant` and `--family` values. A capital or a space makes a name that argparse
    splits or that `locates` fails to match against a lower-cased Kaggle mount path."""
    assert NAMES
    assert all(name and name == name.lower() and " " not in name for name in NAMES)


def test_every_registry_key_is_declared_here():
    """A row keyed by a bare literal instead of the constant is the failure this module exists to stop:
    it reads identically, and then the vocabulary and the table disagree with nothing to notice."""
    assert set(ENCODERS) <= set(NAMES)
