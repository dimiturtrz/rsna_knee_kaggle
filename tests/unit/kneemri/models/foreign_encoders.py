"""The foreign half of the registry: it merges into one namespace, and it keys itself honestly."""

from kneemri.models.backbone import ENCODERS
from kneemri.models.foreign_encoders import FOREIGN_ENCODERS
from kneemri.models.variants import BASE, LARGE, SMALL

DINO_VARIANTS = {SMALL, BASE, LARGE}


def test_keys_match_their_variant():
    """The dict key IS the encoder's variant — a row filed under the wrong key resolves `--variant` to a
    different trunk, which is a shape error deep in a load rather than a failure at the lookup."""
    assert all(key == encoder.variant for key, encoder in FOREIGN_ENCODERS.items())


def test_merges_into_the_registry_without_shadowing():
    """The split is invisible to callers: every foreign row reaches `ENCODERS`, and none of them lands on
    a key the DINOv2 rows already hold — `**FOREIGN_ENCODERS` would silently win if one did."""
    assert set(FOREIGN_ENCODERS) <= set(ENCODERS)
    assert not set(FOREIGN_ENCODERS) & DINO_VARIANTS
