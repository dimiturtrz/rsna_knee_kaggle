"""How much of what an ensemble does not know is REDUCIBLE: `python -m kneemri.evaluation.uncertainty_split`.

For K members holding Bernoulli beliefs `p_k` about each study and each finding:

    total     = H[mean_k p_k]      what the ensemble does not know
    aleatoric = mean_k H[p_k]      the label's own ambiguity — no member can remove it
    epistemic = total - aleatoric  what a different or better member could still remove

The epistemic term IS the K-way Jensen-Shannon divergence, which is why this module owns no arithmetic
of its own beyond an average: `MemberDivergence.kl` is the clamped logarithm, the entropy is read off it
as `1 bit - KL(p || 0.5)`, and the pairwise ruler `seat_value` already prints is the K=2 case of the
number below. A second entropy helper would differ from that one by a clamp and the two would stop
summing.

WHY THE FRACTION AND NOT THE DIVERGENCE. `MemberDivergence` reports a DISTANCE, and a distance cannot
say what share of the remaining uncertainty it accounts for — the same 0.02 bits of disagreement means
one thing among members who are otherwise certain and another among members who are all hedging at 0.5.
`epistemic / total` is scale-free and is the only number here that adjudicates anything: low says the
members genuinely agree and no further member of any kind can pay, high says they disagree and the
COMBINER is throwing it away.

PER LABEL AND THEN AVERAGED, the shape `MemberDivergence` and `RankAverage.agreement` already use and
for the same reason: one number over the flattened array is dominated by whichever finding carries the
widest prevalence, and the metric that decides the board scores each finding alone. The pooled row here
is the mean of the twelve per-label totals over the mean of the twelve per-label aleatorics, never a
mean of twelve ratios — a ratio of means is what the twelve rows above it add up to.

WHAT IT REFUSES, and both refusals are the point rather than an error path:

 * A MEMBER THAT HELD NOTHING OUT. A member fitted at `--fold -1` has seen every study on this holdout,
   so its beliefs there are memories and its agreement with anyone is not evidence about anything. It
   writes no `.holdout.npz` at all, so the failure is silent everywhere: it would simply be absent from
   the pool and the fraction would be reported over a smaller K under the entry's name. Named members
   with no record are therefore a hard refusal, and a pool that comes back shorter than the members
   handed to it is a second one — that is how a foreign record, dropped by `RecordMix` with a warning,
   would otherwise change the denominator without changing the report. This is `foreign_overlap`'s
   default-is-guilt rule applied to our own checkpoints: a member is in-sample unless its own record
   proves otherwise.

 * FEWER THAN TWO MEMBERS. One member diverges from nobody, so its epistemic term is 0.0 by
   construction and its fraction is 0.0 — a number that reads exactly like a measured finding of "no
   reducible uncertainty at all" and is a statement about the arithmetic instead.

WHAT IT CANNOT SEE, and it bounds the answer in one direction only. Separately-trained members are an
honest sample of the model class; TTA windows and view variants of one checkpoint share weights, so
folding them in would count one belief K times and drive the epistemic term DOWN. Every member here is
one trained checkpoint, so the fraction is a LOWER bound on what a genuinely different model class
could disagree about, never an upper one.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.member_divergence import BOUND, MemberDivergence
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.runs.holdout_record import HoldoutRecord

logger = logging.getLogger(__name__)

DECOMPOSABLE = 2
"""The smallest pool whose disagreement is a quantity rather than a definition.

Same bar and same reason as `MemberDivergence.NEIGHBOURED`: with one member the epistemic term is
identically zero, and zero printed as a measurement is the one wrong answer this report could give.
"""

UNINFORMED = 0.5
"""The maximum-entropy Bernoulli belief, which is what makes the entropy here a KL rather than a helper.

`KL(p || 0.5) = 1 + p log2 p + (1-p) log2(1-p) = 1 - H(p)` exactly, in bits, so a Bernoulli entropy is
one subtraction away from the divergence `MemberDivergence` already clamps and tests. Not an
approximation and not a smoothing choice — an identity, and the reason this module adds no logarithm.
"""

CERTAIN = 1.0
"""The entropy of `UNINFORMED` in bits, and the ceiling every quantity in this report is measured under.

Named rather than written as a bare 1.0 because it appears as the subtrahend of the identity above,
where a reader has to be able to tell "one bit" from "one member" and from "one label".
"""


@dataclass(frozen=True)
class UncertaintyShare:
    """One finding's three entropies, and the share of them a better member could still remove.

    A type rather than three floats returned in a tuple, because `total` and `aleatoric` are only
    meaningful together — `epistemic` is their difference and the deliverable is their ratio, so a
    caller that received them separately could pair one label's total with another's aleatoric and the
    arithmetic would succeed.
    """

    total: float
    aleatoric: float

    @property
    def epistemic(self) -> float:
        """What the members disagree about — the K-way Jensen-Shannon divergence, in bits."""
        return self.total - self.aleatoric

    @property
    def reducible(self) -> float:
        """`epistemic / total`: the share of the remaining uncertainty another member could address.

        Zero when there is no uncertainty to divide, which is a degenerate pool rather than a reading —
        a total of exactly 0.0 needs every member certain and unanimous on every study, and the ratio
        there is 0/0. Reported as 0.0 rather than raised on, because the row it belongs to still has to
        print its own totals beside the eleven others.
        """
        return self.epistemic / self.total if self.total > 0.0 else 0.0

    def line(self, name: str) -> str:
        """This finding's row: the three bits, then the only number that adjudicates anything."""
        return (
            f"  {name}: total {self.total:.4f}  aleatoric {self.aleatoric:.4f}  "
            f"epistemic {self.epistemic:.4f} bits  reducible {self.reducible:.1%}"
        )


class UncertaintySplit:
    """An ensemble's remaining uncertainty, divided into the half a member could fix and the half it could not."""

    def __init__(self, pool: Pool) -> None:
        self.pool = pool

    @classmethod
    def over(cls, members: list[Path]) -> "UncertaintySplit":
        """These checkpoints' recorded holdout beliefs, joined on the studies ALL of them held out.

        The leak guard lives here rather than in the arithmetic because this is the only place that
        still knows what was ASKED for. A member with no record is refused by name; a pool that comes
        back shorter than the ask is refused by count. Downstream neither is visible: `RecordMix` says
        `no holdout record, skipped` at info level and returns a perfectly well-formed smaller pool.
        """
        recordless = [member.stem for member in members if not HoldoutRecord(member).exists()]
        if recordless:
            message = (
                f"no holdout record for {', '.join(sorted(recordless))} — a member fitted at --fold -1 "
                "held nothing out and has seen every study this would be measured on, so its agreement "
                "with the others is a memory rather than evidence"
            )
            raise ValueError(message)
        pool = RecordMix(members).per_member()
        if len(pool) != len(members):
            message = (
                f"{len(pool)} of the {len(members)} named members reached the pool — a record was "
                "dropped (foreign weights, or a run with no checkpoint), and the denominator moved"
            )
            raise ValueError(message)
        return cls(pool)

    @staticmethod
    def entropy(beliefs: Float[np.ndarray, "studies targets"]) -> Float[np.ndarray, "studies targets"]:
        """Bernoulli entropy in bits, read as `1 bit - KL(p || 0.5)` so the clamped log has one home.

        Certainty in either direction reads 0.0 and a belief of 0.5 reads the full bit. The clamp is
        `MemberDivergence.BOUND`, inherited rather than chosen, since a saturated sigmoid genuinely
        occurs under bf16 training and `0 * log 0` is where this arithmetic is undefined.
        """
        clamped = np.clip(beliefs, BOUND, 1.0 - BOUND)
        return CERTAIN - MemberDivergence.kl(clamped, np.full_like(clamped, UNINFORMED))

    def per_label(self) -> list[UncertaintyShare]:
        """Each finding's own decomposition, in `TARGETS` order — the twelve rows the pooled row averages."""
        if len(self.pool) < DECOMPOSABLE:
            message = (
                f"a pool of {len(self.pool)} has no disagreement to measure: the epistemic term of one "
                f"member is 0.0 by construction, and at least {DECOMPOSABLE} are needed for it to be a reading"
            )
            raise ValueError(message)
        beliefs = np.stack(self.pool.predictions)
        total = np.mean(self.entropy(np.mean(beliefs, axis=0)), axis=0)
        aleatoric = np.mean([np.mean(self.entropy(member), axis=0) for member in beliefs], axis=0)
        return [UncertaintyShare(float(one), float(other)) for one, other in zip(total, aleatoric, strict=True)]

    def pooled(self) -> UncertaintyShare:
        """The whole ensemble's split: the mean of the per-label totals over the mean of the aleatorics.

        A ratio of means rather than a mean of ratios, so the pooled row is the twelve rows above it
        added up — a mean of twelve fractions weights a finding nobody is uncertain about as heavily as
        one everybody is, which is the flattening the per-label shape exists to avoid.
        """
        shares = self.per_label()
        return UncertaintyShare(
            float(np.mean([share.total for share in shares])),
            float(np.mean([share.aleatoric for share in shares])),
        )

    def denominator(self) -> str:
        """What the numbers below are over: how many members, how many studies, and that they are held out."""
        return (
            f"{len(self.pool)} separately trained members over {len(self.pool.truth)} studies every one "
            f"of them HELD OUT (no member here was fitted at --fold -1); members: {', '.join(self.pool.names)}"
        )

    def report(self) -> list[str]:
        """The denominator, the twelve findings, and the pooled share that is reducible."""
        shares = self.per_label()
        return [
            self.denominator(),
            "uncertainty split (Bernoulli bits per study, per finding, averaged over studies):",
            *[share.line(name) for name, share in zip(TARGETS, shares, strict=True)],
            self.pooled().line("POOLED"),
        ]


def main() -> None:
    """Split the remaining uncertainty of these members into its reducible and irreducible halves."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--records",
        type=Path,
        nargs="+",
        required=True,
        help="the checkpoints to decompose, sharing ONE split. Every one must carry a holdout record: a "
        "member fitted at --fold -1 has seen these studies and is refused by name rather than skipped.",
    )
    arguments = command.parsed()
    command.report(UncertaintySplit.over(list(arguments.records)).report())


if __name__ == "__main__":
    main()
