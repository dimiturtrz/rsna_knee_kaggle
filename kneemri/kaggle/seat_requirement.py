"""What a member would have to BE to close the gap to first place — the inversion nothing here did.

`python -m kneemri.kaggle.seat_requirement [--target 0.0117]`

WHY THIS EXISTS. Every seat tool in this repo answers "what is this member worth?" — `seat_value` for a
candidate against the roster, `priceable_blend` for the best subset, `member_distance` and `member_axes`
for how far apart two members sit. All of them price members that EXIST. None of them answers the
question that decides what to put on the card next: given the relationship this pool exhibits between a
member's solo quality, its distance from the seated set, and what it adds, **what would a member need to
look like to add the amount first place actually requires?**

That amount is known. `interpretations/kneemri/2026-09-06_what-first-place-costs-in-member-points.md`
converts the board gap into offline points: the ensemble must reach 0.8862 and it reads 0.8751, so
`TARGET_GAIN` is +0.0117 — and
`interpretations/kneemri/2026-09-06_the-ensemble-ceiling-is-not-a-selection-artefact.md` shows that no
selection over the 206 members owned produces it. So the next member has to, and this says what it is.

THE FIT, AND ITS HONEST WEAKNESS. Marginal gain is regressed on solo score and on rank agreement with
the seated blend, over every priceable member not already seated. Two coefficients and an intercept
from ~200 points is not a fragile fit. What IS fragile is the EXTRAPOLATION: observed single-seat gains
run to about +0.002, and the target is six times the largest one ever measured here. A linear fit
evaluated six times outside its data range is an argument about DIRECTION and about ORDER OF MAGNITUDE,
never a prediction. `report` prints the extrapolation factor beside every requirement it states, so the
number can never be read without the reason to distrust it.

WHAT A REFUSAL WOULD MEAN. If the fit says no achievable member closes the gap — if the solo score
required exceeds 1.0, or the agreement required is below what any two models of the same task can have
— then the single-member route is arithmetically closed, and the remaining routes are full-fit dose and
a structurally different POOL rather than one more member in this one. That is a decisive answer and it
costs no card, which is the whole point of asking it before spending 75 minutes.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core import cli
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.vote import RankAverage
from kneemri.kaggle.priceable_blend import PriceableBlend

logger = logging.getLogger(__name__)

TARGET_GAIN = 0.0117
"""What the ensemble's offline score must rise by for the board to read first place.

0.8862 required minus `honest12`'s 0.8751. Derived in
`interpretations/kneemri/2026-09-06_what-first-place-costs-in-member-points.md`, where all three of its
assumptions are stated as pushing the same way — so this is a FLOOR on what is needed, not a target that
might turn out generous.
"""

DEFAULT_BASE = (
    "chainfour_s2026",
    "calib_dense_hold_s2026",
    "stack_increments_s2026",
    "blendhold_s2026",
    "labelwise_dense_s2026",
    "margin_seat_s2026",
    "finding_calib_dense_s2026",
)
"""The greedy optimum over the 206-member pool, measured 2026-09-06 at offline 0.8764.

The seated set has to be the BEST one available rather than the champion's, because the question is what
a member adds to the best ensemble this pool can form. Measured against a worse base, every candidate
looks more valuable than it is, and the requirement this module prints would come out too small.
"""

AGREEMENT_BANDS = ((0.90, 1.01), (0.85, 0.90), (0.80, 0.85), (0.70, 0.80), (0.00, 0.70))
"""Where the pool's members sit relative to the seated blend, coarse enough that a band holds members.

The bands are legislated rather than quantiles on purpose: the question is whether a member exists at a
given DISTANCE, and quantile bands would always report five populated bands however tight the cloud is.
"""

MIN_READINGS = 3
"""Below this a two-coefficient fit is under-determined and would report a line through noise."""


@dataclass(frozen=True)
class SeatReading:
    """One candidate, on the three axes the fit relates: what it scores, how far it sits, what it adds."""

    name: str
    solo: float
    agreement: float
    gain: float


@dataclass(frozen=True)
class SeatFit:
    """Marginal gain as a plane over (solo, agreement), with the scale of what it was fitted on."""

    intercept: float
    per_solo: float
    per_agreement: float
    residual: float
    best_observed: float

    def gain(self, solo: float, agreement: float) -> float:
        """What the fit expects a member at this quality and this distance to add."""
        return self.intercept + self.per_solo * solo + self.per_agreement * agreement

    def solo_needed(self, agreement: float, target: float) -> float:
        """The solo score a member at this agreement would need in order to add `target`.

        Returns infinity when quality does not move gain at all, because a flat coefficient means no solo
        score answers the question and a division would invent one.
        """
        if self.per_solo == 0.0:
            return float("inf")
        return (target - self.intercept - self.per_agreement * agreement) / self.per_solo

    def agreement_needed(self, solo: float, target: float) -> float:
        """The agreement a member at this quality would need in order to add `target`.

        The other inversion, and the more interesting one: this pool's members agree with each other at
        0.8-0.95, so an answer below what two models of one task can plausibly reach is a refusal.
        """
        if self.per_agreement == 0.0:
            return float("inf")
        return (target - self.intercept - self.per_solo * solo) / self.per_agreement

    def extrapolation(self, target: float) -> float:
        """How many times the largest gain ever observed the target is — the reason to distrust a number.

        Printed beside every requirement. A fit evaluated well outside the range it was taken in is an
        argument about direction, and this is the factor that says how far outside.
        """
        return float("inf") if self.best_observed <= 0.0 else target / self.best_observed


class SeatRequirement:
    """Fits what a seat is worth on this pool, then inverts it into what the next seat must be."""

    def __init__(self, models: Path, base: tuple[str, ...] = DEFAULT_BASE) -> None:
        self.models = models
        self.base = base
        self.blend = PriceableBlend(models, "rank", len(base))

    def readings(self) -> list[SeatReading]:  # devtools-ignore: test-mirror
        """Every priceable member not already seated, measured against the seated set.

        Solo, agreement and gain are all taken inside ONE `RecordMix` per candidate, on the split that
        candidate shares with the base. Taking the base's score once outside the loop would price each
        candidate's gain against a different set of studies than its own score was taken on, which is the
        row-index-versus-study-name defect `priceable_blend.scored_on` already had to fix once.
        """
        seated = [self.models / f"{name}.pt" for name in self.base]
        readings: list[SeatReading] = []
        for candidate in self.blend.on_the_corpus_ruler(self.blend.candidates()):
            if candidate.stem in self.base:
                continue
            mix = RecordMix([*seated, candidate], foreign=True)
            mix.narrow()
            pool = mix.per_member(RankAverage())
            base_only = pool.combined(list(range(len(seated))))
            with_it = pool.combined(list(range(len(seated) + 1)))
            scores = LabelScores()
            readings.append(
                SeatReading(
                    name=candidate.stem,
                    solo=scores.mean(pool.predictions[-1], pool.truth),
                    agreement=RankAverage.agreement(pool.predictions[-1], base_only),
                    gain=scores.mean(with_it, pool.truth) - scores.mean(base_only, pool.truth),
                )
            )
        return readings

    @staticmethod
    def fitted(readings: list[SeatReading]) -> SeatFit | None:
        """Least squares of gain on (1, solo, agreement), or nothing when there is too little to fit."""
        if len(readings) < MIN_READINGS:
            return None
        design = np.array([[1.0, reading.solo, reading.agreement] for reading in readings])
        observed = np.array([reading.gain for reading in readings])
        coefficients, *_ = np.linalg.lstsq(design, observed, rcond=None)
        predicted = design @ coefficients
        return SeatFit(
            intercept=float(coefficients[0]),
            per_solo=float(coefficients[1]),
            per_agreement=float(coefficients[2]),
            residual=float(np.sqrt(np.mean((observed - predicted) ** 2))),
            best_observed=float(observed.max()),
        )

    @staticmethod
    def frontier(readings: list[SeatReading]) -> list[str]:
        """The best member owned inside each agreement band — the shape of the distance/quality tradeoff.

        This is the line that says whether "buy distance" is a plan or a wish. If every member below a
        given agreement scores near chance, then the pool has no distance it can USE, and the seat that
        closes the gap has to be built rather than selected.
        """
        lines: list[str] = []
        for low, high in AGREEMENT_BANDS:
            inside = [reading for reading in readings if low <= reading.agreement < high]
            if not inside:
                lines.append(f"  [{low:.2f}, {high:.2f})  none owned")
                continue
            best = max(inside, key=lambda reading: reading.solo)
            lines.append(
                f"  [{low:.2f}, {high:.2f})  {len(inside):3d} members, best solo {best.solo:.4f} "
                f"({best.name}, gain {best.gain:+.4f})"
            )
        return lines

    def report(self, target: float = TARGET_GAIN, readings: list[SeatReading] | None = None) -> list[str]:
        """The fit, the best members it was taken on, and what closing `target` would require.

        `readings` is injectable so the formatting can be exercised without a data root, and so a caller
        that already paid for the ~200 mixes can re-ask at a second target without paying again.
        """
        readings = self.readings() if readings is None else readings
        fit = self.fitted(readings)
        if fit is None:
            return [f"REFUSED: {len(readings)} priceable candidates on disk, and a plane needs {MIN_READINGS}"]
        ranked = sorted(readings, key=lambda reading: reading.gain, reverse=True)
        lines = [f"{len(readings)} candidates against a {len(self.base)}-seat base", "the five that add most:"]
        lines += [
            f"  {reading.name:34s} solo {reading.solo:.4f}  agreement {reading.agreement:.4f}  gain {reading.gain:+.4f}"
            for reading in ranked[:5]
        ]
        lines.append(
            f"fit: gain = {fit.intercept:+.4f} {fit.per_solo:+.4f}*solo {fit.per_agreement:+.4f}*agreement"
            f"  (rms residual {fit.residual:.4f}, best observed gain {fit.best_observed:+.4f})"
        )
        typical = float(np.median([reading.agreement for reading in readings]))
        strongest = max(reading.solo for reading in readings)
        furthest = min(readings, key=lambda reading: reading.agreement)
        lines.append(
            f"the furthest member owned: {furthest.name} at agreement {furthest.agreement:.4f}, "
            f"solo {furthest.solo:.4f}, gain {furthest.gain:+.4f}"
        )
        lines.append("the distance/quality frontier — best solo owned inside each agreement band:")
        lines += self.frontier(readings)
        lines.append(f"TO ADD {target:+.4f} — {fit.extrapolation(target):.1f}x the largest gain ever observed here")
        lines.append(
            f"  at the pool's median agreement {typical:.4f}: solo must reach {fit.solo_needed(typical, target):.4f}"
        )
        lines.append(
            f"  at the pool's best solo {strongest:.4f}: "
            f"agreement must fall to {fit.agreement_needed(strongest, target):.4f}"
        )
        return lines


def main() -> None:
    """Invert this pool's seat economics: what would a member have to be to close the gap to first place?"""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--target", type=float, default=TARGET_GAIN, help="the offline gain to close")
    arguments = command.parsed()
    root = command.rooted(arguments)
    command.report(SeatRequirement(root.models()).report(arguments.target))


if __name__ == "__main__":
    main()
