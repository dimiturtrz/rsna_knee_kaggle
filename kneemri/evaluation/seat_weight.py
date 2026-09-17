"""What a seat is worth at each WEIGHT, not just at one seat of n. `python -m kneemri.evaluation.seat_weight`.

WHY THE WEIGHT NEEDED ITS OWN RULER. Every offline seat number this project owns is taken at ONE weight:
`seat_value` adds a candidate to a roster and each member gets one vote of n+1. The champion does not
serve that — it serves the foreign seat at 1/3 beside six members, which is a weight of about 2.4 seats,
chosen on the board at a cost of one slot per value. So the quantity the entry actually turns on has only
ever been priced by spending slots on it, and `raptorweight_v1` (1/7 to 1/3, +0.001) is the whole record.

HOW A WEIGHT IS EXPRESSED WITHOUT A NEW VOTE RULE. `Vote.combine` takes a list of member tables and
averages them, so a member appearing `w` times in that list has weight `w` — which is exactly what a
SEAT is. Integer weights are therefore the shipped arithmetic rather than a model of it, and no vote
class learns a parameter it would then have to be trusted with at serving time. It also keeps the two
rules comparable: the same repetition means the same thing under `LogitAverage` and `RankAverage`.

THE CUT IS LEDGER75, matching `seat_add`, and every printed line names it. The two seat instruments have
to answer on the same ruler or their numbers cannot be compared, and the midpoint is the cut that orders
pairs the other way from the board: on the 39 ensemble entries carrying both, midpoint inverts 12 of 414
resolved pairs and ledger75 inverts 0 of 397. `--midpoint` asks the other question deliberately.

THE INTERVAL IS PAIRED, AND THAT IS THE POINT. The candidate at weight `w` and the base without it are
read on the SAME studies, so the honest question is how much the DELTA moves under resampling, not how
wide each score is. A per-arm interval would be dominated by which studies are in the holdout, which is
identical for both arms and cancels.

WHAT IT CANNOT DO, STATED BEFORE THE NUMBERS. A foreign FAMILY of folds cannot be unrolled into five
seats and priced here. Their five checkpoints partition the corpus by their own split, so on our 909
holdout four of the five have trained on any given study, and a five-seat rank-mean over these studies
reads four in-sample opinions per row. The fold-matched column (`foreign_price --emit`) is the honest
stand-in and it is a FLOOR on the served form, because the entry serves five folds per study and this is
one. A five-seat number is available on the board and nowhere else.
"""

import logging
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from kneemri.evaluation.ledger75 import LedgerSeventyFive
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.inference.vote import LogitAverage, RankAverage, Vote

logger = logging.getLogger(__name__)

DRAWS = 2000
"""Bootstrap resamples behind each interval. Enough that the 2.5th percentile is stable to a ten-thousandth."""

INTERVAL_SEED = 2026
"""The resampling seed, fixed so two readings of the same pool are the same number rather than nearly."""

PERCENTILES = (2.5, 97.5)
"""The interval reported: a two-sided 95%, the same band every seat number in the ledger carries."""


class SeatWeight:
    """One candidate's contribution to a base pool at each integer weight, under one vote rule."""

    def __init__(self, pool: Pool, seat: int, scores: LabelScores | None = None) -> None:
        """`scores` defaults to LEDGER75, the cut `seat_add` reads every quoted add under."""
        self.pool = pool
        self.seat = seat
        self.scores = scores if scores is not None else LedgerSeventyFive().scorer()

    @staticmethod
    def rules() -> dict[str, Vote]:
        """The vote rules a weight may be priced under, by the name the CLI takes.

        Both, because the champion serves `LogitAverage` while every ledger reading was taken under
        `RankAverage`, and a weight chosen under one rule and served under the other is a knob fitted on
        a ruler the entry does not use.
        """
        return {LogitAverage().named(): LogitAverage(), RankAverage().named(): RankAverage()}

    def base(self) -> list[int]:
        """The pool's members except the candidate — the arm every weighted arm is read against."""
        return [index for index in range(len(self.pool)) if index != self.seat]

    def blended(self, weight: int) -> Float[NDArray[np.float64], "studies targets"]:
        """The base plus `weight` copies of the candidate, combined under the pool's own vote rule.

        Repetition IS the weight: `Vote.combine` averages the list it is given, so a member listed twice
        carries two seats of the vote. Weight 0 is the base itself, which is why the control needs no
        separate code path and cannot drift from the candidate's arm.
        """
        return self.pool.combined([*self.base(), *([self.seat] * weight)])

    def scored(self, weight: int) -> float:
        """The mean AUC of the base with the candidate at this weight."""
        return self.scores.mean(self.blended(weight), self.pool.truth)

    def interval(self, weight: int, draws: int = DRAWS, seed: int = INTERVAL_SEED) -> tuple[float, float]:
        """A paired bootstrap of the DELTA against the base, over studies.

        Both arms are resampled on the same rows because they are read on the same studies; resampling
        them independently would report the holdout's own variability, which is common to both and is
        not what a seat decision turns on.
        """
        weighted = self.blended(weight)
        control = self.blended(0)
        rows = np.arange(len(self.pool.truth))
        generator = np.random.default_rng(seed)
        deltas = []
        for _ in range(draws):
            drawn = generator.choice(rows, size=len(rows), replace=True)
            truth = self.pool.truth[drawn]
            deltas.append(self.scores.mean(weighted[drawn], truth) - self.scores.mean(control[drawn], truth))
        low, high = np.nanpercentile(deltas, PERCENTILES)
        return float(low), float(high)

    def on(self, rows: Float[NDArray[np.int64], " studies"]) -> "SeatWeight":
        """The same seat read on a subset of the studies — the half a weight was NOT chosen on."""
        pool = Pool(
            names=self.pool.names,
            predictions=[member[rows] for member in self.pool.predictions],
            truth=self.pool.truth[rows],
            vote=self.pool.vote,
        )
        return SeatWeight(pool, self.seat, self.scores)

    def split_half(self, weights: list[int], seed: int = INTERVAL_SEED) -> list[str]:
        """Choose the weight on one half of the studies, READ it on the other, then swap.

        One parameter chosen on 909 studies is a small optimism, not a free one, and the objection
        raised against a twelve-weight form has to be answered for the one-weight form too. The number
        that matters is the delta the chosen weight earns on studies it was not chosen on — if that
        collapses toward the 1x delta, the weight is a holdout artefact and the entry ships 1x.
        """
        order = np.random.default_rng(seed).permutation(len(self.pool.truth))
        halves = (order[: len(order) // 2], order[len(order) // 2 :])
        rows = []
        for fit, read in (halves, halves[::-1]):
            chooser, reader = self.on(fit), self.on(read)
            chosen = max(weights, key=chooser.scored)
            gain = reader.scored(chosen) - reader.scored(0)
            flat = reader.scored(1) - reader.scored(0)
            rows.append(
                f"  chosen {chosen}x on {len(fit)} studies: {gain:+.4f} out-of-sample on the other "
                f"{len(read)} (1x earns {flat:+.4f} there)"
            )
        return ["split-half, the weight chosen on one half and read on the other:", *rows]

    def lines(self, weights: list[int], draws: int = DRAWS) -> list[str]:
        """One row per weight: the mix, the delta against the base, and the delta's own interval."""
        control = self.scored(0)
        rows = [f"{self.pool.rule()} base of {len(self.base())} members: {control:.4f} without the candidate"]
        for weight in weights:
            low, high = self.interval(weight, draws=draws)
            gain = self.scored(weight) - control
            rows.append(
                f"  {self.pool.names[self.seat]} at {weight}x: {self.scored(weight):.4f} "
                f"({gain:+.4f} [{low:+.4f}, {high:+.4f}])"
            )
        return rows


def main() -> None:
    """Price a candidate seat at several weights against a base of members, under each vote rule."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--base", type=Path, nargs="+", required=True, help="the base members' records")
    command.parser.add_argument("--candidate", type=Path, required=True, help="the seat being weighted")
    command.parser.add_argument("--weights", type=int, nargs="+", default=[1, 2, 3], help="integer seat counts")
    command.parser.add_argument("--foreign", action="store_true", help="the candidate's weights are someone else's")
    command.parser.add_argument("--draws", type=int, default=DRAWS, help="bootstrap resamples behind each interval")
    command.parser.add_argument("--split-half", action="store_true", help="choose the weight on half, read on half")
    command.parser.add_argument("--midpoint", action="store_true", help="read at the midpoint instead of LEDGER75")
    arguments = command.parsed()

    records = [*arguments.base, arguments.candidate]
    mixed = RecordMix(records, foreign=arguments.foreign)
    scores = LabelScores(MIDPOINT) if arguments.midpoint else None
    lines: list[str] = []
    for name, vote in SeatWeight.rules().items():
        pool = mixed.per_member(vote)
        weighted = SeatWeight(pool, len(pool) - 1, scores)
        lines += [f"--- {name} @{weighted.scores.name()}", *weighted.lines(arguments.weights, draws=arguments.draws)]
        if arguments.split_half:
            lines += weighted.split_half(arguments.weights)
    command.report(lines)


if __name__ == "__main__":
    main()
