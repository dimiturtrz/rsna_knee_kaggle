"""How far apart two members are in CONFIDENCE — the axis a rank average throws away.

Printed by `python -m kneemri.kaggle.seat_value`, directly beneath `MemberDistance`, because the two
answer different halves of one question and either alone is misleading.

WHY A SECOND DISTANCE COLUMN AT ALL. `MemberDistance` measures rank correlation, and rank correlation is
invariant to any monotone rescaling of a member's scores by construction. So two members that order every
study identically read as twins there whatever they predict — one saturated at 0.02/0.98, the other
hedged at 0.45/0.55, same ordering, correlation 1.0. That difference is real, it is the difference a
LOGIT average is sensitive to and a rank average is not, and the champion has shipped `LogitAverage`
since `kneemri-vote`. A ruler that cannot see the axis the shipped vote reads on is a ruler that is
silent about the shipped vote.

WHAT IT MEASURES. Jensen-Shannon divergence in bits between the two members' predictions read as
Bernoulli beliefs, per study, averaged over studies and then over labels — the same per-label-then-mean
shape `RankAverage.agreement` uses, for the same reason: one number over the flattened array is dominated
by whichever label has the widest prevalence spread, and the metric that decides the board scores each
label alone. Jensen-Shannon rather than KL, which the honest counter in knee-c2ew names: KL is asymmetric
and unbounded and explodes wherever the reference has near-zero mass, which is exactly the rare-finding
regime where our per-label scores are already weakest. JS is symmetric, bounded in [0, 1] bits, and a
proper metric under its square root.

IT READS NO LABELS FOR ITS DISTANCES, and `Pool.truth` is touched in exactly one place — `labelled`,
which prints each label's POSITIVE COUNT beside its divergence and nothing else. That is the support
knee-c2ew requires beside any thin-label divergence, and it is a count rather than a score: a divergence
that turned out to be a restatement of a quality reading would be worthless for the same reason
`MemberDistance` refuses the truth vector entirely.

THE HONEST COUNTER, and it is the whole reason this column is read BESIDE the rank one rather than
instead of it. A member that merely disagrees about CALIBRATION — same ordering, wider spread — scores
high here, and calibration is the cheapest thing in this project to change: `--calibrated-targets` moves
it, temperature moves it, and `level_calibration` measures it directly. So a high divergence with a high
rank correlation is a member that is not new information, only differently scaled, and seating it buys
what a temperature knob would have. The pair to look for is high divergence AND low rank correlation.
The pair to distrust is high divergence alone.

UNVALIDATED AS A PREDICTOR, exactly like the rank column and stated in the same words: nothing here says
how many board ticks a bit of divergence buys, and no submission slot should be spent on this number
alone. Unlike the rank column it does not even have the one paired board observation to reproduce — it
was written from a mechanism, not from a split, so it is the weaker of the two until a seat decision
turns on it and lands.
"""

import numpy as np
from jaxtyping import Float

from kneemri.evaluation.pool import Pool

NEIGHBOURED = 2
"""The smallest roster in which a seated member has anyone to be near.

Same bar and same reason as `MemberDistance.NEIGHBOURED`: the roster's own spread is the only scale
these numbers have, a roster of one has no spread, and inventing one is worse than printing its absence.
"""

PRESENT = 0.5
"""Where a soft target counts as the finding being present, for the support column only.

MEASURED ON ITS OWN FIRST RUN, 2026-08-30, and it is the reason this constant exists rather than a bare
`> 0`. The truth vector `RecordMix` builds is NOT a binary indicator — over the 909-study fold-0 holdout
it takes seven values (0, 0.08, 0.28, 0.68, 0.82, 0.94, 1.0), a reader-agreement scale rather than a
label. Counting anything above zero reports 99% support on every one of the twelve findings, which is a
statement about how rarely all readers agree on absence and says nothing about prevalence. At 0.5 the
same columns read 0.23 to 0.51, which is the prevalence the ledger quotes.
"""

BOUND = 1e-6
"""How far a probability is held off 0 and 1 before its logarithm is taken.

Not a smoothing parameter and deliberately far below any difference worth reading: bf16 training and
fp16 serving both make a saturated sigmoid reachable (`vote.py`), so 0.0 and 1.0 genuinely occur, and
`0 * log 0` is the one place this arithmetic is undefined rather than merely extreme. At 1e-6 the
largest divergence two members can read is 1.0 bit, which is the true bound, so the clamp changes the
scale nowhere it is not preventing a nan.
"""


class MemberDivergence:
    """Confidence-space distance between members, on the same pool the rank distance is read on."""

    def __init__(self, roster: Pool, candidates: Pool) -> None:
        self.roster = roster
        self.candidates = candidates

    @staticmethod
    def js(one: Float[np.ndarray, "studies targets"], other: Float[np.ndarray, "studies targets"]) -> float:
        """Mean per-study Jensen-Shannon divergence in bits, per label and then averaged over labels.

        The two members are read as Bernoulli beliefs about each study, which is what a sigmoid head
        emits and what the serving logit average combines. Identical predictions read 0.0; a study one
        member calls certain-yes and the other certain-no reads 1.0, the bound.
        """
        left = np.clip(one, BOUND, 1.0 - BOUND)
        right = np.clip(other, BOUND, 1.0 - BOUND)
        middle = (left + right) / 2.0
        per_study = (MemberDivergence.kl(left, middle) + MemberDivergence.kl(right, middle)) / 2.0
        return float(np.mean(np.mean(per_study, axis=0)))

    @staticmethod
    def kl(
        belief: Float[np.ndarray, "studies targets"], reference: Float[np.ndarray, "studies targets"]
    ) -> Float[np.ndarray, "studies targets"]:
        """Per-study Bernoulli KL in bits, both arms of it — the half a JS is built from.

        PUBLIC since `UncertaintySplit`, which reads a Bernoulli ENTROPY through it rather than adding
        a second spelling of the same logarithms: `H(p) = 1 bit - KL(p || 0.5)` exactly, so the one
        clamped log lives here and the decomposition beside it inherits `BOUND` instead of choosing
        its own. Two entropy helpers that disagree by a clamp is how a divergence and a total stop
        summing to each other.
        """
        return belief * np.log2(belief / reference) + (1.0 - belief) * np.log2((1.0 - belief) / (1.0 - reference))

    def against(self, member: Float[np.ndarray, "studies targets"], skip: int | None = None) -> list[float]:
        """One member's divergence from each seated member, optionally skipping one seat.

        `skip` carries the same load it carries on `MemberDistance`: a seated member diverges 0.0 from
        itself, and leaving that in would report every incumbent as its own nearest neighbour and flatten
        the only scale this report has.
        """
        return [self.js(member, seated) for index, seated in enumerate(self.roster.predictions) if index != skip]

    def spread(self) -> list[str]:
        """What each seated member's NEAREST neighbour reads — the scale a candidate is judged on.

        Nearest is the MINIMUM here, where the rank column takes a maximum: agreement rises towards a
        twin and divergence falls towards one.
        """
        if len(self.roster.predictions) < NEIGHBOURED:
            return [f"  no spread: a roster of {len(self.roster.predictions)} has no neighbour to be near"]
        return [
            f"  seated {name}: nearest {min(self.against(seated, skip=index)):.4f} bits"
            for index, (name, seated) in enumerate(zip(self.roster.names, self.roster.predictions, strict=True))
        ]

    def labelled(self) -> list[str]:
        """Each label's median pairwise divergence WITHIN the roster, with its positive count beside it.

        The support column is not decoration. A label with a handful of positives has a divergence
        computed almost entirely on studies nobody has the finding, so it reports how far apart two
        members' NEGATIVE confidence is — a real quantity, and not the one a reader assumes they are
        looking at. knee-c2ew's counter asks for the count to be printed next to the number rather than
        for the thin labels to be dropped, because which labels are thin is itself a reading.

        The count is taken at `PRESENT`, and the denominator is printed beside it, because the truth
        vector here is a reader-agreement scale rather than a binary label — see that constant for what
        the naive count reported instead.
        """
        if len(self.roster.predictions) < NEIGHBOURED:
            return ["  no per-label spread: a roster of one has no pair to diverge"]
        pairs = [
            (self.roster.predictions[left], self.roster.predictions[right])
            for left in range(len(self.roster.predictions))
            for right in range(left + 1, len(self.roster.predictions))
        ]
        lines = [
            "per label, median divergence within the roster (support = studies whose soft target "
            f"reaches {PRESENT}, not a count of non-zero ones):"
        ]
        for label in range(self.roster.truth.shape[1]):
            divergences = [self.js(one[:, label : label + 1], other[:, label : label + 1]) for one, other in pairs]
            column = self.roster.truth[:, label]
            support = int(np.sum(column >= PRESENT))
            lines.append(
                f"  label {label}: median {float(np.median(divergences)):.4f} bits  "
                f"support {support}/{len(column)} at >= {PRESENT}"
            )
        return lines

    def report(self) -> list[str]:
        """Every candidate by confidence divergence from the roster, most divergent first."""
        rows = [
            (min(divergences), float(np.mean(divergences)), name)
            for name, divergences in (
                (name, self.against(candidate))
                for name, candidate in zip(self.candidates.names, self.candidates.predictions, strict=True)
            )
        ]
        lines = [
            f"confidence divergence (Jensen-Shannon, bits) over the same roster of "
            f"{len(self.roster.predictions)}, its own nearest-neighbour spread:",
            *self.spread(),
            "candidates, most divergent first (nearest = the seat they would be scaled like):",
        ]
        lines.extend(
            f"  {name}: nearest {nearest:.4f}  mean {mean:.4f}" for nearest, mean, name in sorted(rows, reverse=True)
        )
        lines.extend(self.labelled())
        return lines
