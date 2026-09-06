"""How wide a board delta is at the PUBLIC leaderboard's own size, which is about four hundred studies.

`python -m kneemri.evaluation.board_floor`.

WHY IT EXISTS (knee-6j8b4). Nine axes in this project are recorded as CLOSED on a board flat: the
seventh seat, count six-to-seven, encoder family at seat seven, member type, the vote rule, serving
views, a kit refresh, ordinal separation, and count at high redundancy. Every one of those closures
reads a board difference of 0.000-0.001 between two entries and concludes the axis is inert. But the
public leaderboard scores roughly 30% of 1,322 hidden studies — about four hundred — and a difference
measured on four hundred studies carries a sampling interval of its own. If that interval is wider than
the offline deltas those swaps were priced at (0.001-0.003), then the flats never said "inert". They
said "below the board's resolution", which licenses different work: the axis stays open, and it simply
cannot be settled by spending slots on it.

WHAT IT MEASURES, AND ON WHOSE STUDIES. A PAIRED study bootstrap of the macro-AUC difference between two
member subsets of one pool, resampled at a stated size. Paired because both arms are read on the same
studies and the holdout's own variability is common to them; resampling the arms independently would
report how variable our 909 studies are, which is not what a seat decision turns on. The rows are drawn
from OUR holdout, so this is the width of a delta at that sample size — it inherits our case mix, not
the hidden test's, and it says nothing about the LEVEL either board or holdout reads (knee-l4xx6). A
board flat that this shows to be inside the interval is unresolved; one outside it is a real flat.

THE BOARD ALSO QUANTISES. Scores are published to three decimals, so a true 0.0005 effect is unreadable
whatever the sample size. That floor is reported beside the sampling one, because the larger of the two
is the resolution, and a plan aimed under either is a plan aimed at noise.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.bootstrap import INTERVAL, RESAMPLE_SEED, RESAMPLES
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.runs.holdout_record import SUFFIX

logger = logging.getLogger(__name__)

PUBLIC_STUDIES = 400
"""The public leaderboard's own size: about 30% of the 1,322 hidden studies, rounded to a round number.

Round because the share is quoted rather than published, and a false precision here would read like the
count is known. The width goes as 1/sqrt(n), so 350 or 450 move it by under a tenth of itself."""

BOARD_QUANTUM = 0.001
"""The board publishes three decimals, so half a quantum is the smallest difference it can even show."""


@dataclass(frozen=True)
class BoardFloor:
    """Two member subsets of one pool, and the width of the difference between them at a given size."""

    pool: Pool
    arms: tuple[list[int], list[int]]
    scores: LabelScores

    @staticmethod
    def seats(pool: Pool, names: list[str]) -> list[int]:
        """The pool positions these member names sit at, refusing a name the pool does not carry.

        A name is taken with or without the record suffix, because a pool names its members by FILE.

        A missing member is the failure this guards: `record_mix` DROPS a record whose ruler or study set
        does not align, so an arm named in the CLI can silently become an arm of one fewer member, and
        the difference measured would be that drop rather than the swap under test.
        """
        carried = [held.removesuffix(SUFFIX) for held in pool.names]
        missing = [name for name in names if name not in carried]
        if missing:
            message = f"the pool does not carry {missing}; it carries {carried}"
            raise ValueError(message)
        return [carried.index(name) for name in names]

    def delta(self, rows: Float[NDArray[np.int64], " studies"]) -> float:
        """Arm A's macro AUC minus arm B's, both voted and both read on exactly these rows."""
        first, second = (self.pool.combined(arm)[rows] for arm in self.arms)
        truth = self.pool.truth[rows]
        return self.scores.mean(first, truth) - self.scores.mean(second, truth)

    def interval(self, studies: int, draws: int = RESAMPLES, seed: int = RESAMPLE_SEED) -> tuple[float, float]:
        """The 95% interval of that difference when only `studies` studies are scored, drawn with replacement."""
        generator = np.random.default_rng(seed)
        rows = np.arange(len(self.pool.truth))
        deltas = [self.delta(generator.choice(rows, size=studies, replace=True)) for _ in range(draws)]
        low, high = np.nanpercentile(deltas, INTERVAL)
        return float(low), float(high)

    def lines(self, sizes: list[int], draws: int = RESAMPLES) -> list[str]:
        """The width at each size, beside the quantum, because the resolution is the larger of the two."""
        whole = np.arange(len(self.pool.truth))
        reported = [
            f"observed delta on our {len(whole)} studies: {self.delta(whole):+.4f}",
            f"board quantum: {BOARD_QUANTUM:.4f} published, so {BOARD_QUANTUM / 2:.4f} is unreadable by rounding",
        ]
        for size in sizes:
            low, high = self.interval(size, draws)
            reported.append(f"n={size}: 95% interval [{low:+.4f}, {high:+.4f}], half-width {(high - low) / 2:.4f}")
        return reported


def main() -> None:
    """Say how wide a one-swap board difference is at the public leaderboard's size, from our own holdout."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--records", type=Path, nargs="+", required=True, help="every member of either arm")
    command.parser.add_argument("--arm-a", nargs="+", required=True, help="the first entry's members, by name")
    command.parser.add_argument("--arm-b", nargs="+", required=True, help="the second entry's members, by name")
    command.parser.add_argument(
        "--sizes", type=int, nargs="+", default=[PUBLIC_STUDIES], help="study counts to report the width at"
    )
    command.parser.add_argument("--draws", type=int, default=RESAMPLES, help="bootstrap resamples behind each interval")
    arguments = command.parsed()

    pool = RecordMix(arguments.records).per_member()
    floor = BoardFloor(
        pool, (BoardFloor.seats(pool, arguments.arm_a), BoardFloor.seats(pool, arguments.arm_b)), LabelScores()
    )
    command.report(floor.lines(arguments.sizes, arguments.draws))


if __name__ == "__main__":
    main()
