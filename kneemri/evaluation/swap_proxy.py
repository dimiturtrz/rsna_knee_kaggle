"""What an ensemble loses when its seats are replaced by distilled twins of themselves.

`python -m kneemri.evaluation.swap_proxy --pair band_dense_s2026=oof_teacher_s2026 --pair ...`

THE QUESTION IS DIVERSITY, NOT MEMBER QUALITY. Every distilled twin this project owns beats its own
control solo — that is measured and not in doubt. What is in doubt is what happens when SEVERAL seats
are distilled against the same posterior at once: Allen-Zhu & Li 2021 report that a re-ensemble of
members distilled from one shared teacher captures almost none of the original ensemble's margin, and OKDDip
2020 measures peer-ensemble error UP 5-14%. Solo gains that each clear their control can therefore sum
to an ensemble that is FLAT or worse, because what the seats had was disagreement and the teacher is
what they now agree with.

WHY A PROXY AND NOT THE SIX-SWAP ITSELF. The champion's six seats are all fitted with `--fold -1` —
they hold nothing out, which is what makes them the strongest members the project can package and what
makes them unreadable offline. Their distilled twins (`*_df`) are dark for the same reason. So the
swap cannot be priced on the seats it will actually be performed on; it can only be priced on seats
that DID hold fold 0 out, and where both the original and its one-variable distilled twin wrote a
holdout record. Those pairs are named on the command line, and the number this module returns is a
statement about THEM.

WHAT IT REPORTS. Three things, because a single number cannot separate the two ways the swap can fail:
the originals voting together, the full swap, and EVERY intermediate configuration (each seat
independently original or distilled). A full swap that is flat while a half swap is up says the loss
is collective — shared agreement — and not a property of any one twin. Beside each configuration goes
the mean pairwise disagreement of its members, which is the mechanism the score is supposed to be
explained by: if the swap costs ensemble points, disagreement must have fallen too, or the story is
wrong.

WHAT IT CANNOT SAY. The fold-0 holdout prices diversity too cheaply — `record_mix` measured one
collapse at 0.0059 offline against 0.014 on the board, 2.4x — so a flat reading here is weaker evidence
than a negative one. And a pair is only a pair because the caller says so: this module cannot check
that the two runs differ in the teacher alone, and a mis-declared pair is a comparison of two unrelated
members wearing the word "swap".
"""

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.denominator import Denominator
from core.paths import DataRoot
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.vote import LogitAverage

logger = logging.getLogger(__name__)

WATCHED = ("MCL", "Lateral OA")
"""The two labels the swap is expected to move first, called out of the per-label table by name.

Not a selection rule — every label is reported. They are named because the distillation term's whole
claim is that it carries the teacher's ordering on the labels our own supervision is thinnest on, and
a reader scanning twelve rows for the two that matter is a reader who will quote the wrong one.
"""


@dataclass(frozen=True)
class Seat:
    """One seat and the twin that would replace it — the unit the swap is performed in."""

    original: str
    distilled: str

    @classmethod
    def parsed(cls, declared: str) -> "Seat":
        """`original=distilled`, refused rather than guessed at when it is not that shape."""
        original, separator, distilled = declared.partition("=")
        if not separator or not original or not distilled:
            message = f"a pair is 'original=distilled', not {declared!r}"
            raise ValueError(message)
        return cls(original, distilled)

    def named(self, *, swapped: bool) -> str:
        """Which member this seat contributes under one configuration."""
        return self.distilled if swapped else self.original


@dataclass(frozen=True)
class Configuration:
    """One assignment of every seat to original or distilled, scored."""

    swapped: tuple[bool, ...]
    score: float
    per_label: dict[str, float]
    disagreement: float

    def count(self) -> int:
        """How many seats are distilled — the axis the table is read along."""
        return sum(self.swapped)

    def line(self, seats: list[Seat], baseline: float) -> str:
        """The row a decision is read off, as a delta against the all-originals configuration."""
        members = ", ".join(seat.named(swapped=swap) for seat, swap in zip(seats, self.swapped, strict=True))
        return (
            f"  {self.count()}/{len(seats)} distilled  {self.score:.4f}  "
            f"{self.score - baseline:+.4f}  disagreement {self.disagreement:.4f}   {members}"
        )


class SwapProxy:
    """Every original/distilled configuration of the named seats, voted over their shared holdout."""

    def __init__(self, seats: list[Seat], pool: Pool) -> None:
        self.seats = seats
        self.pool = pool
        self._index = {Path(name).stem: position for position, name in enumerate(pool.names)}

    @classmethod
    def over(cls, seats: list[Seat], root: DataRoot) -> "SwapProxy":  # devtools-ignore: test-mirror
        """The named members as one pool, joined onto the studies every one of them held out.

        Not unit-testable and deliberately unmirrored: it reads checkpoints off the data root, which is
        the one thing a unit test here may not touch. Both halves of every pair must survive the join.
        A seat whose twin was dropped is not a seat that can be swapped, and scoring the rest under the
        caller's seat count is the wrong-denominator
        failure this project rules out — so it is a refusal, not a warning.
        """
        named = [name for seat in seats for name in (seat.original, seat.distilled)]
        mix = RecordMix(root.checkpoints(named))
        for dropped in mix.narrow():
            logger.warning("%s cannot be mixed with the rest, dropped", dropped)
        pool = mix.per_member(LogitAverage())
        seated = {Path(name).stem for name in pool.names}
        missing = [name for name in named if name not in seated]
        if missing:
            message = f"these members carry no mixable holdout record, so their seats cannot be swapped: {missing}"
            raise FileNotFoundError(message)
        Denominator(len(pool), "no named member carries a mixable record").refuse_empty()
        return cls(seats, pool)

    def members(self, swapped: tuple[bool, ...]) -> list[int]:
        """The pool positions one configuration seats."""
        return [self._index[seat.named(swapped=swap)] for seat, swap in zip(self.seats, swapped, strict=True)]

    def disagreement(self, subset: list[int]) -> float:
        """Mean pairwise absolute difference between the seated members' probabilities.

        The mechanism the score is supposed to be explained by. Absolute difference rather than a
        divergence because the members are read here as the vote reads them — as probabilities on one
        scale — and a divergence would need a binning choice that nothing in the vote makes.
        """
        stacked: Float[np.ndarray, "members studies targets"] = np.asarray(
            [self.pool.predictions[position] for position in subset], dtype=np.float64
        )
        pairs = [
            float(np.abs(stacked[first] - stacked[second]).mean())
            for first, second in itertools.combinations(range(len(subset)), 2)
        ]
        return float(np.mean(pairs)) if pairs else 0.0

    def scored(self, scores: LabelScores, swapped: tuple[bool, ...]) -> Configuration:
        """One configuration's mean AUC, its per-label table and its internal disagreement."""
        subset = self.members(swapped)
        combined: NDArray[np.float64] = self.pool.combined(subset)
        per_label = scores.per_label(combined, self.pool.truth)
        return Configuration(
            swapped=swapped,
            score=float(np.nanmean(list(per_label.values()))),
            per_label=per_label,
            disagreement=self.disagreement(subset),
        )

    def every(self, scores: LabelScores) -> list[Configuration]:
        """All 2^n configurations, originals first so every later row is a delta against it."""
        return [self.scored(scores, swapped) for swapped in itertools.product([False, True], repeat=len(self.seats))]

    def lines(self, scores: LabelScores) -> list[str]:
        """The reading as a report: the two endpoints, the halves ranked, then the per-label swap table."""
        configurations = self.every(scores)
        originals = configurations[0]
        full = configurations[-1]
        halves = sorted(
            (one for one in configurations if 0 < one.count() < len(self.seats)),
            key=lambda one: one.score,
            reverse=True,
        )
        rows = [
            f"{len(self.seats)} seats over {len(self.pool.truth)} studies under {self.pool.rule()}",
            f"  ALL ORIGINALS   {originals.score:.4f}  disagreement {originals.disagreement:.4f}",
            f"  FULL SWAP       {full.score:.4f}  {full.score - originals.score:+.4f}  "
            f"disagreement {full.disagreement:.4f}",
            "  every intermediate configuration, best first:",
        ]
        rows.extend(one.line(self.seats, originals.score) for one in halves)
        best = max(configurations, key=lambda one: one.score)
        rows.append(
            f"BEST configuration: {best.count()}/{len(self.seats)} distilled at {best.score:.4f} "
            f"({best.score - originals.score:+.4f} vs originals, {best.score - full.score:+.4f} vs the full swap)"
        )
        rows.append("per-label, full swap against the originals:")
        rows.extend(
            f"  {label:<18} {originals.per_label[label]:.4f} -> {full.per_label[label]:.4f}  "
            f"{full.per_label[label] - originals.per_label[label]:+.4f}" + ("   <- watched" if label in WATCHED else "")
            for label in originals.per_label
        )
        return rows


def main() -> None:  # devtools-ignore: test-mirror
    """Report what a distillation swap costs an ensemble, over seats whose twins both held fold 0 out."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="a seat as original=distilled, repeated once per seat",
    )
    arguments = command.parsed()

    seats = [Seat.parsed(declared) for declared in arguments.pair]
    Denominator(len(seats), "no seat named by --pair").refuse_empty()
    root = DataRoot.from_config(arguments.paths)
    command.report(SwapProxy.over(seats, root).lines(LabelScores()))


if __name__ == "__main__":
    main()
