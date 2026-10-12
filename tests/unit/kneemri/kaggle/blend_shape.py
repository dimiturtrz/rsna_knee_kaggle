"""Each blend spelling taken apart only when the call is its own, and the zip only inside a probit script."""

import ast

from kneemri.kaggle.blend_shape import CombinedShape, Fusion, SeatShape, Split, ZipShape


def _call(source: str) -> ast.Call:
    node = ast.parse(source, mode="eval").body
    assert isinstance(node, ast.Call)
    return node


def test_combined_shape_split():
    """Three arguments split ours, theirs and the weights; any other arity is not this shape."""
    combined = _call("RaptorSeat.combined(a, b, w)")
    assert CombinedShape().split(combined, probit=False) == Split(
        (combined.args[0], combined.args[1]), combined.args[2], remainder=True, fusion=Fusion.RANK
    )
    assert CombinedShape().split(_call("RaptorSeat.combined(a, b)"), probit=False) is None
    assert CombinedShape().split(_call("combined(a, b, w)"), probit=False) is None, "a bare name is not the method"


def test_seat_shape_split():
    """The seat's own `weight=` keyword is the foreign share; a seat without one is not priced here."""
    split = SeatShape().split(_call("RaptorSeat(weight=0.5).blended(a, b)"), probit=False)
    assert split is not None
    assert (split.remainder, split.fusion, ast.literal_eval(split.weights)) == (True, Fusion.RANK, 0.5)
    assert SeatShape().split(_call("RaptorSeat().blended(a, b)"), probit=False) is None, "no weight keyword"
    assert SeatShape().split(_call("seat.blended(a, b)"), probit=False) is None, "a seat built elsewhere"


def test_zip_shape_split():
    """A two-argument zip is a probit table-and-share pairing only when the script maps ranks through ndtri."""
    zipped = _call("zip(tables, shares)")
    split = ZipShape().split(zipped, probit=True)
    assert split is not None
    assert (len(split.tables), split.remainder, split.fusion) == (1, False, Fusion.PROBIT)
    assert ZipShape().split(zipped, probit=False) is None
    assert ZipShape().split(_call("zip(a, b, c)"), probit=True) is None
