"""The ways a kernel spells its final blend of our table with foreign ones, each taken apart without running it."""

import ast
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

COMBINED_ARGUMENTS = 3
"""`RaptorSeat.combined(ours, theirs, weights)`."""

PAIR_ARGUMENTS = 2
"""`RaptorSeat(...).blended(ours, theirs)` and `zip(tables, shares)`."""


class Fusion(StrEnum):
    """How an entry's tables become one: a weighted rank average, or weighted z-scores of the ranks."""

    RANK = "rank"
    PROBIT = "probit"


@dataclass(frozen=True)
class Split:
    """One blend call taken apart: the table expressions in order, their weights, and whether the first
    table takes the remainder `1 - sum(weights)` instead of a weight of its own."""

    tables: tuple[ast.expr, ...]
    weights: ast.expr
    remainder: bool
    fusion: Fusion


class BlendShape(Protocol):
    """One way a kernel spells its final blend."""

    def split(self, call: ast.Call, *, probit: bool) -> Split | None:
        """The call taken apart, or None when it is not this shape."""
        ...


class CombinedShape:
    """`RaptorSeat.combined(ours, theirs, weights)`: our table takes the remainder, a rank average."""

    def split(self, call: ast.Call, *, probit: bool) -> Split | None:
        """The call taken apart, or None when it is not a three-argument `.combined`."""
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "combined" and len(call.args) == COMBINED_ARGUMENTS):
            return None
        return Split((call.args[0], call.args[1]), call.args[2], remainder=True, fusion=Fusion.RANK)


class SeatShape:
    """`RaptorSeat(weight=w).blended(ours, theirs)`: the foreign side takes `w`, ours the rest."""

    def split(self, call: ast.Call, *, probit: bool) -> Split | None:
        """The call taken apart, or None when it is not a two-argument `.blended` on a weighted seat."""
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "blended" and len(call.args) == PAIR_ARGUMENTS):
            return None
        seat = func.value
        keywords: list[ast.keyword] = seat.keywords if isinstance(seat, ast.Call) else []
        weight = next((keyword.value for keyword in keywords if keyword.arg == "weight"), None)
        if weight is None:
            return None
        return Split((call.args[0], call.args[1]), weight, remainder=True, fusion=Fusion.RANK)


class ZipShape:
    """`zip(tables, (shares...))` around `ndtri`: every table carries its own share, a probit fusion."""

    def split(self, call: ast.Call, *, probit: bool) -> Split | None:
        """The call taken apart, or None when it is not a two-argument `zip` in a probit script."""
        func = call.func
        if not (probit and isinstance(func, ast.Name) and func.id == "zip" and len(call.args) == PAIR_ARGUMENTS):
            return None
        return Split((call.args[0],), call.args[1], remainder=False, fusion=Fusion.PROBIT)


SHAPES: tuple[BlendShape, ...] = (CombinedShape(), SeatShape(), ZipShape())
