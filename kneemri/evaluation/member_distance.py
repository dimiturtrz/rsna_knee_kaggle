"""How far a candidate's ORDERING sits from the roster's — the axis the holdout cannot see.

Printed by `python -m kneemri.kaggle.seat_value`, beside the seat value it is meant to be read against.

THE DEFECT THIS EXISTS TO COVER. On 2026-08-21 two entries differed in one member and split the board:
`sixfresh` seated `lr4e-5_ws` (solo 0.8525, seat value -0.0003) and scored 0.931; `sixstrong` seated
`band_span` (solo 0.8565, seat value +0.0002, and a near-twin of the incumbent `calib_span`) and scored
0.930. Same vote, same `overlap`, nothing else moved. The better member by both offline rulers cost a
tick, and the worse one took the entry.

WHY BOTH OFFLINE RULERS POINT THE WRONG WAY THERE, BY CONSTRUCTION. The vote is a per-label rank average,
so a member that orders the studies the way an incumbent already does contributes that one ordering
TWICE. The average barely moves, which is exactly why a twin reads as harmless on the holdout — a seat
that buys nothing also costs nothing to the mean. `seat_value` inherits the blindness rather than
correcting it: it prices the same combined array against the same truth, and 0 of 7 candidates cleared
`BOARD_RESOLVABLE` in the sweep that preceded the split. A ruler that cannot separate a twin from a
stranger cannot answer the question a sixth seat actually asks.

SO THE DISTANCE ITSELF READS NO LABELS. Distance is a property of the predictions, and the truth vector
is what drags every other reading back onto the quality axis. Keeping it out of `against` and `spread` is
the point, not an economy: a high-distance reading can never be a restatement of a high-quality one.

`report` DOES open the label table, for one column and one reason. It printed distance alone until
2026-09-04, when `fault_ruler` measured that an ordering by distance alone is inverted here — see below.
The candidate has to be read on both axes at once, so the report carries the solo score beside the
distance rather than leaving the reader to fetch it from somewhere else and pair the two by hand.

WHAT IT MEASURES. Rank correlation, per label, averaged over labels — the same per-label ranking the
shipped vote applies (`Vote.ranks`), so the number is computed in the space the decision is made in
rather than in logit space where a calibration difference would read as disagreement. Two summaries per
candidate, because they answer different questions:

  * NEAREST — the highest correlation to any single incumbent. This is the twin detector, and it is the
    one that fires on `band_span`. A seat is lost to ONE redundant neighbour, not to an average.
  * MEAN — correlation to the roster as a whole, which says whether the candidate is far from everyone
    or merely far from the roster's centre while sitting on top of one member.

THE ROSTER'S OWN SPREAD IS THE SCALE, and without it these are unitless numbers nobody can act on. The
report prints what each SEATED member's nearest neighbour reads, so a candidate has something to be
compared against: above the roster's own worst nearest is a twin of somebody already seated, below its
best nearest is genuinely new information.

IT RANKS OFFLINE PAIR GAIN, AND IT STILL DOES NOT PRICE A SEAT. `kneemri.evaluation.fault_ruler` measured
this statistic against the gain it is used to predict — 166 members, 1200 sampled pairs, 48000 discordance
trials each — and it came first of four candidate rulers at +0.7086 rank correlation, ahead of the
disagreement-error ratio (+0.6999), the raw disagreement rate (+0.6955) and the double-fault rate
(+0.5911) that the literature says is the only one that works. Four decimals do not separate four
collinear statistics over 1200 dependent samples; the honest reading is that this one is not worse.

THE CORRELATION ONLY HOLDS WITH PAIR QUALITY HELD FIXED, and that caveat is the load-bearing half. Read
raw, every one of the four statistics correlates NEGATIVELY with gain (this one at -0.5313), because the
distant members of this pool are the weak ones and distance proxies for "one of these two is bad".
`report` below ranks candidates across the whole quality range, so its ordering is exposed to exactly
that confound — compare candidates of comparable solo score, never across a quality gap. See
`interpretations/kneemri/2026-09-04_the-diversity-ruler-was-checked-and-the-citation-does-not-transfer.md`.

Nothing here says how many board ticks a unit of distance buys — the gains above are offline, and
`record_mix` measures the holdout pricing diversity ~2.4x too cheaply against the board. It ranks pairs;
it does not price a roster, and no slot should be spent on the strength of this number alone.
"""

import numpy as np
from jaxtyping import Float

from kneemri.evaluation.pool import Pool
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.vote import RankAverage

NEIGHBOURED = 2
"""The smallest roster in which a seated member has anyone to be near.

A roster of one has no nearest neighbour, and `max()` over the empty list of its neighbours is where
this module crashed on 2026-08-24 (knee-1gdf) once the pinned champion moved to the all-data pool and
left exactly one scoreable member. The scale this report exists to print does not exist at that size,
so it says so rather than inventing one.
"""


class MemberDistance:
    """Rank-space distance between members: label-free to compute, read beside the solo score."""

    def __init__(self, roster: Pool, candidates: Pool, scores: LabelScores | None = None) -> None:
        self.roster = roster
        self.candidates = candidates
        self.scores = scores or LabelScores()

    def against(self, member: Float[np.ndarray, "studies targets"], skip: int | None = None) -> list[float]:
        """One member's agreement with each seated member, optionally skipping one seat.

        `skip` is what lets an INCUMBENT be measured on the same footing as a candidate: a seated member
        correlates 1.0 with itself, and leaving that in would report every incumbent as its own twin and
        make the roster's spread — the only scale this report has — meaningless.
        """
        return [
            RankAverage.agreement(member, seated)
            for index, seated in enumerate(self.roster.predictions)
            if index != skip
        ]

    def spread(self) -> list[str]:
        """What each seated member's NEAREST neighbour reads — the scale a candidate is judged on.

        A roster too small to have neighbours reports the absence of the scale, because a candidate's
        distance means nothing without something to be far from. Printing the candidate columns anyway
        would let a number computed against ONE member read as a number computed against a roster.
        """
        if len(self.roster.predictions) < NEIGHBOURED:
            return [f"  no spread: a roster of {len(self.roster.predictions)} has no neighbour to be near"]
        return [
            f"  seated {name}: nearest {max(self.against(seated, skip=index)):.4f}"
            for index, (name, seated) in enumerate(zip(self.roster.names, self.roster.predictions, strict=True))
        ]

    def report(self) -> list[str]:
        """Every candidate by distance from the roster, most distant first, WITH the solo score beside it.

        The solo column is the only part of this class that opens the label table, and it is here because
        `fault_ruler` measured what happens without it: read across a quality gap, distance correlates
        -0.5313 with the gain it is supposed to predict, and +0.7086 once the pair's quality is held
        fixed. This pool's distant members are its weak ones, so an ordering by distance alone is an
        ordering by weakness wearing a diversity label. Two candidates are comparable on the distance
        column when their solo scores are comparable; a distant candidate half a point below the seat bar
        is not a find.
        """
        rows = [
            (max(agreements), float(np.mean(agreements)), self.scores.mean(candidate, self.candidates.truth), name)
            for name, candidate, agreements in (
                (name, candidate, self.against(candidate))
                for name, candidate in zip(self.candidates.names, self.candidates.predictions, strict=True)
            )
        ]
        qualifier = f"@roster{len(self.roster.predictions)}/{self.scores.name()}"
        lines = [f"roster of {len(self.roster.predictions)}, its own nearest-neighbour spread:", *self.spread()]
        lines.append(
            f"candidates, most distant first (nearest = twin detector; compare within a solo band). "
            f"EVERY number below is {qualifier}: `nearest` is a max over THIS roster, not over the "
            f"member pool that D35/D157 are fitted on, and `solo` is a macro at THIS cut:"
        )
        lines.extend(
            f"  {name}: nearest {nearest:.4f}{qualifier}  mean {mean:.4f}  solo {solo:.4f}@{self.scores.name()}"
            for nearest, mean, solo, name in sorted(rows)
        )
        return lines
