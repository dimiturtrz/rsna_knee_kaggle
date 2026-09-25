"""What lifting MCL and Lateral OA would actually BUY, measured instead of assumed: `ColumnLift`.

`python -m kneemri.kaggle.column_lift [--target 0.0117]`

WHY THIS EXISTS. `interpretations/kneemri/2026-09-06_two-columns-carry-three-quarters-of-the-gap.md`
decomposed the gap to first place by finding and named two columns — MCL 0.8283 and Lateral OA 0.8369
against a 0.8765 pool mean — worth +0.0073 of the +0.0097 gap IF they were lifted to that mean. That
arithmetic is a DECOMPOSITION of the stack's own per-label scores. It is not a measurement of what a
member shaped that way would add, because a member does not replace the stack's column; it joins a
rank-average and moves it part of the way. `lean_weak_columns` is on the card behind that arithmetic
with a hand-priced P(seat) of 0.15, and nothing has measured the number that guess stands on.

THE INSTRUMENT. Seat a SYNTHETIC eighth member next to the greedy-optimal seven. Its ten other columns
are the seated blend's own combined prediction, so it agrees with the base perfectly everywhere outside
the two findings and can add nothing there — every point it earns comes from the columns under test.
Its two named columns are that same prediction blended toward truth at a strength `s`, in rank space, so
sweeping `s` from 0 to 1 walks the member from "identical to the blend" to "reads those two findings
perfectly". Reading the seat gain against the column AUC that produced it inverts into the number the
queued arm needs: **how high must MCL and Lateral OA go before the member is worth a seat?**

WHAT MAKES IT AN INSTRUMENT AND NOT A SIMULATION. `s = 1` is a member with two ORACLE columns. Its gain
is therefore a hard CEILING on the whole two-column route — not an estimate of one. If a member that
reads MCL and Lateral OA perfectly and is otherwise the blend itself still does not add `TARGET_GAIN`,
then no member shaped along these two columns closes the gap, and that is settled without the card.

THE NULL IS PRINTED, NOT ASSUMED. The `s = 0` row is the same member with no lift at all, and its gain
is not exactly zero: seating a copy of the blend gives the blend two of n+1 votes, which moves the
rank-average slightly on its own. That row is therefore the ARTEFACT, and every gain above it has to be
read against it rather than against zero.

WHAT IT DOES NOT ANSWER. It prices the columns, not the arm: a real member differs from the blend on all
twelve columns, and that difference is worth something the synthetic one is built to have none of. So a
gain here is a floor on what a genuinely distinct member at the same column quality would add, and the
required column AUC it prints is correspondingly an upper bound on what such a member must reach.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.inference.vote import RankAverage, Vote
from kneemri.kaggle.pool_ceiling import WEAK_COLUMNS
from kneemri.kaggle.seat_requirement import DEFAULT_BASE, TARGET_GAIN

logger = logging.getLogger(__name__)

PERFECT = 1.0
"""Where a lifted column can rise no further, so a later row differs only in HOW it is tied."""

STRENGTHS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
"""How far toward truth the two columns are dragged, coarse at the top where the answer is already known.

Legislated rather than a linspace so the ORACLE end (1.0) is always measured: that point is the ceiling
the whole reading turns on, and a grid chosen for evenness could miss it.
"""


@dataclass(frozen=True)
class LiftReading:
    """One strength: what the synthetic member's two columns read, and what seating it added."""

    strength: float
    columns: tuple[float, ...]
    gain: float

    def weakest(self) -> float:
        """The lower of the two column AUCs — the one a real arm would have to reach FIRST."""
        return min(self.columns)


class ColumnLift:
    """Seats a synthetic member whose only advantage is the two weak columns, and prices it."""

    def __init__(
        self,
        models: Path,
        base: tuple[str, ...] = DEFAULT_BASE,
        findings: tuple[str, ...] = WEAK_COLUMNS,
    ) -> None:
        self.models = models
        self.base = base
        self.findings = findings
        self.scores = LabelScores()

    def seated(self) -> Pool:  # devtools-ignore: test-mirror
        """The greedy-optimal seven on their shared studies, on the corpus ruler."""
        mix = RecordMix([self.models / f"{name}.pt" for name in self.base], foreign=True)
        mix.narrow()
        return mix.per_member(RankAverage())

    @staticmethod
    def ranked(column: Float[np.ndarray, " studies"]) -> Float[np.ndarray, " studies"]:
        """A column mapped onto [0, 1] by rank, so blending it with truth is a monotone move and not a scale clash.

        Blending raw sigmoids with a 0/1 target would let the target's scale dominate wherever the model
        is confident, which changes the column by MAGNITUDE rather than by ORDER — and the metric only
        reads order.
        """
        return Vote.ranks(column[:, None])[:, 0]

    def lifted(self, pool: Pool, strength: float) -> Float[np.ndarray, "studies targets"]:
        """The base blend with the named findings dragged `strength` of the way toward truth.

        Everything outside those findings is the blend itself, by identity — the member is built to be
        able to add nothing there, which is what makes the gain attributable to the columns alone.
        """
        blend = pool.combined(list(range(len(pool)))).copy()
        for finding in self.findings:
            column = TARGETS.index(finding)
            truth = (pool.truth[:, column] > MIDPOINT).astype(np.float64)
            blend[:, column] = (1.0 - strength) * self.ranked(blend[:, column]) + strength * truth
        return blend

    def readings(self, pool: Pool, strengths: tuple[float, ...] = STRENGTHS) -> list[LiftReading]:
        """Seat gain and the two column AUCs at every strength, all taken on one pool and one truth."""
        base_only = pool.combined(list(range(len(pool))))
        floor = self.scores.mean(base_only, pool.truth)
        readings: list[LiftReading] = []
        for strength in strengths:
            candidate = self.lifted(pool, strength)
            seated = Pool(
                names=[*pool.names, "lifted"],
                predictions=[*pool.predictions, candidate],
                truth=pool.truth,
            )
            per_label = self.scores.per_label(candidate, pool.truth)
            readings.append(
                LiftReading(
                    strength=strength,
                    columns=tuple(per_label[finding] for finding in self.findings),
                    gain=self.scores.mean(seated.combined(list(range(len(seated)))), pool.truth) - floor,
                )
            )
        return readings

    @staticmethod
    def required(readings: list[LiftReading], target: float) -> LiftReading | None:
        """The cheapest strength that earns `target`, or nothing when even the oracle does not.

        Nothing is the DECISIVE answer, not a missing one: the last reading is a member reading both
        findings perfectly, so a target it misses is a target this axis cannot reach.
        """
        return next((reading for reading in readings if reading.gain >= target), None)

    def report(self, target: float = TARGET_GAIN, readings: list[LiftReading] | None = None) -> list[str]:
        """The sweep, and what it says the queued arm's two columns would have to reach."""
        readings = self.readings(self.seated()) if readings is None else readings
        lines = [
            f"a synthetic eighth member, identical to the blend of {len(self.base)} except on "
            f"{' + '.join(self.findings)}:",
            "  strength   " + "  ".join(f"{finding:>16}" for finding in self.findings) + "   seat gain",
        ]
        lines += [
            f"  {reading.strength:8.2f}   "
            + "  ".join(f"{column:16.4f}" for column in reading.columns)
            + f"   {reading.gain:+.4f}"
            for reading in readings
        ]
        saturated = next((reading for reading in readings if reading.weakest() >= PERFECT), None)
        if saturated is not None and saturated.strength < readings[-1].strength:
            lines.append(
                f"column AUC saturates at strength {saturated.strength:.2f} for {saturated.gain:+.4f}, while the "
                f"oracle row reads the same 1.0000 for {readings[-1].gain:+.4f} — AUC does not determine the seat. "
                "A fully dragged column is BINARY and ties every study within a class, which the rank-average "
                "treats as a confident vote; a member with the same AUC and a real within-class ordering is worth "
                "the smaller number. Read the saturating row as the achievable oracle and the last as the bound."
            )
        ceiling = readings[-1]
        lines.append(f"ORACLE CEILING on this axis: {ceiling.gain:+.4f} against a required {target:+.4f}")
        needed = self.required(readings, target)
        if needed is None:
            lines.append(
                "REFUSED: a member reading both findings PERFECTLY, otherwise the blend itself, does not "
                f"reach {target:+.4f} — the two-column route cannot close the gap on its own"
            )
        else:
            lines.append(
                f"a seat needs strength {needed.strength:.2f}: "
                + ", ".join(
                    f"{finding} at {column:.4f}" for finding, column in zip(self.findings, needed.columns, strict=True)
                )
            )
        return lines


def main() -> None:
    """Price the two-column route before the arm behind it finishes, rather than after."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--target", type=float, default=TARGET_GAIN, help="the seat gain first place needs")
    arguments = command.parsed()
    root = command.rooted(arguments)
    command.report(ColumnLift(root.models()).report(arguments.target))


if __name__ == "__main__":
    main()
