import random
from argparse import Namespace

import pytest

from kneemri.models.backbone import ENCODERS
from kneemri.models.variants import SMALL
from kneemri.training.masked_objectives import Objectives

# `Objectives.choices` reads as a draw to the size gate; seeded so a failure here is reproducible either way.
random.seed(0)


def test_choices():
    """The parser's `--objective` values come from the dispatch table, so the two cannot disagree."""
    assert Objectives.choices() == ("latent", "spectral")


def test_named():
    """An unknown objective is a KeyError at build time, not a silent fall-back to the pixel one.

    Only the refusal is asserted: building either real objective downloads a published checkpoint, which
    is exactly what a unit test must not reach for.
    """
    with pytest.raises(KeyError):
        Objectives(Namespace(), ENCODERS[SMALL]).named("nonesuch")
