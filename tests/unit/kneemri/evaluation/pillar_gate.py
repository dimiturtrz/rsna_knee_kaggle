"""The volumetric gate's wiring — that it scores through the SHARED path and against the archive's own number.

There is no forward here. `main` builds a trunk from a third-party checkout and reads a 2 GB cache, neither
of which belongs in a unit test; what is worth asserting is the thing that would silently go wrong instead.
Two families scoring through two copies of the same loop is how one of them drifts, and a gold constant
retyped per family is how an oracle quietly becomes a number somebody chose.
"""

from kneemri.evaluation import pillar_gate
from kneemri.evaluation.foreign_gate import FeedGate
from kneemri.inference.pillar_member import GOLD


def test_main():
    """The command exists and scores through `FeedGate`, the same reading the 2D family is gated by."""
    assert callable(pillar_gate.main)
    assert pillar_gate.FeedGate is FeedGate, "a second scoring path is a second ruler"
    assert pillar_gate.GOLD is GOLD, "the oracle is the archive's published macro, read from beside the member"
    assert GOLD == 0.797, "the number the archive publishes for itself on the 58 annotated studies"
