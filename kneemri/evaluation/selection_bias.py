"""How much a SEARCH over a member pool flatters the holdout score it was searched on.

THIS NUMBER HAD NO HOME. `MemberCurve.honest` MEASURES it — it refits the greedy on half the studies,
scores the resulting prefixes on the other half, and prints the in-sample minus held-out gap on every
seat line — but it prints it and keeps nothing, so every consumer that wanted the correction as a VALUE
re-typed one out of a log or out of prose. That is the same shape as the stale-slope failure
`board_check` was written against one level down: a measured constant quoted from a document rather
than from the place that measures it. It lives here, once, and `board_check` reads it from here.

It is a BAND rather than a point on purpose. Four logs measure it — `logs/greedy_all.log:191-207`,
`logs/honest_packaged.log:92-108`, `logs/honest_pool_20260829.log:189-202` — and they put it at +0.0025
to +0.0031 for seven to twelve seats over pools of 77 to 119 members. A single midpoint would be a
number nothing measured, and a tolerance around it would be a second free parameter; the interval the
replications actually span is the honest statement, and a correction inside it is one the evidence
supports.

THE BOARD PRICES THE SAME CORRECTION INDEPENDENTLY, which is why this is a constant and not a guess.
`greedyseat_v1` (2026-08-30) carried a raw greedy offline of 0.8768. Through the two prior fold-0
anchors (0.8712 -> 0.933 and 0.8751 -> 0.937, slope 1.026) the raw number predicts 0.939 and the board
read 0.936, a miss of -0.0027; the bias-corrected 0.8740 predicts 0.936 and missed by +0.0001. An
offline measurement of +0.0028 and a board measurement of +0.0027, on two rulers that share nothing.

WHAT IT MUST NOT BE USED FOR is a blanket subtraction over the whole record (knee-wz81). Applied to
every historical search-selected row it moves retro-prediction DOWN — `board_transfer`'s Spearman
0.961 -> 0.947 over all rows, 0.819 -> 0.647 above board 0.930 — because the early four-seat entries
were searched over pools of ten to twenty-two members and a large-pool constant does not describe
them. The band is right for the regime it was measured in, and the regime is part of the constant.
"""

SELECTION_BIAS = (0.0025, 0.0031)
"""The interval the four replications span, in AUC, for 7-12 seats over pools of 77-119 members.

The module docstring carries the derivation; this is the value it derives. Read as the correction a
search-selected score may honestly be reduced by, not as an amount of score that is available.
"""


RECORDED_PLACES = 4
"""The decimals a holdout AUC is written to, and therefore the resolution any comparison of two of them
gets. The band's own endpoints are four-decimal numbers, so subtracting one from a four-decimal score in
binary floating point lands a ten-thousandth of a hair OUTSIDE the interval it was constructed from —
`0.8731 - 0.0025` is not `0.8706`. Rounding to the resolution the record actually writes is what makes
a closed band mean the thing it says."""


class SelectionBias:
    """The gap between a search's in-sample score and the same seating scored where it was not fitted."""

    def __init__(self, band: tuple[float, float] = SELECTION_BIAS) -> None:
        self.band = band

    def corrects(self, raw: float, recorded: float) -> bool:
        """Whether `recorded` is `raw` less a correction these measurements actually support.

        Both ends are closed and both matter. Too small a correction is the defect this exists to
        catch; too large a one is a row quietly buying itself a better transfer residual, which reads
        as a confirmation of the ruler rather than as the fit it would be.
        """
        low, high = self.band
        return low <= round(raw - recorded, RECORDED_PLACES) <= high

    def stated(self, raw: float) -> str:
        """The window a raw search score may be recorded in, said as the two numbers a writer needs."""
        low, high = self.band
        return f"{raw - high:.4f} to {raw - low:.4f}"
