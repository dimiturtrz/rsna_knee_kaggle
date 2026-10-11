"""The best roster a DECODE BUDGET can buy: `python -m kneemri.kaggle.decode_entry`.

THE EFFICIENCY TRACK PAYS ITS OWN PODIUM ($7k/$6k/$5k, `comp/COMPETITION_RULES.txt`) and we sit at rank
16 there against rank 28 on accuracy. Its ruler weights runtime hard, and runtime on a four-CPU Kaggle
host is dominated by DECODE — `Ensemble.reads` calls it "the I/O-bound half", and a fleet pass measured
the card at 0% while pydicom worked. So an efficiency entry is not a small roster, it is a roster on FEW
READS, and those are different objects: `roster_reads` measured 108 members riding one decode.

WHY IT COULD NOT BE ASKED BEFORE. Both halves of the question already had tools and nothing joined them.
`roster_reads` prices a set's COST in decodes and says nothing about its score; `seat_value` prices the
score of ADDING one member to a fixed board entry and says nothing about cost, and it cannot answer
"which roster" at all because it never removes anyone. The question an efficiency entry actually asks —
*of everything on disk, what is the best-scoring roster that costs at most N decodes* — is a search over
subsets under a cost ceiling, and it is the join of those two tools.

WHAT IT DOES NOT DO. It does not predict a rank: the efficiency metric's formula is not published (the
evaluation page is JS-only), so this reports score and decodes and never a placing. It does not replace
`dry_run`, which is the kernel actually decoding. And its score is the fold-0 holdout, so the usual
caveat stands unchanged — a `_cf` or `_ff` member has seen every holdout study and is excluded by
`RecordMix` rather than by judgement.

THE SEARCH IS GREEDY, AND THAT IS A CHOICE WITH A KNOWN COST. Subsets of ninety members are not
enumerable, and greedy forward selection is what `member_curve` already uses to choose what to submit,
so the two agree by construction. It can miss a pair that only works together; it cannot miss it
SILENTLY, because the walk prints every step and a flat step is visible.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from core import cli
from core.denominator import Denominator
from core.roster_flags import RosterFlags
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.kaggle.roster_reads import RosterReads

logger = logging.getLogger(__name__)

NO_GAIN = 0.0
"""What a step must beat to be taken. A greedy walk that accepts a tie grows forever at no score, and
on this pool a tie is common: rank-averaging a near-twin onto a roster changes the mean by nothing at
all. Stopping at the first non-improving step is what makes the reported roster the SMALLEST one at its
score, which is the object the efficiency track pays for."""


@dataclass(frozen=True)
class BudgetedRoster:
    """One step of the walk: who is seated, what they score, and what they cost."""

    names: list[str]
    score: float
    decodes: int

    @property
    def per_decode(self) -> float:
        """Seats bought per decode spent — the trade, at this step."""
        return len(self.names) / self.decodes

    def line(self) -> str:
        """This step as one row, newest member last so the walk reads as a sequence of decisions."""
        return (
            f"  {len(self.names):2d} seats / {self.decodes} decodes = {self.per_decode:.1f} per decode"
            f"  score {self.score:.4f}  +{self.names[-1]}"
        )


@dataclass(frozen=True)
class DecodeEntry:
    """A pool of scoreable members, each with the decode it rides, searched under a ceiling on decodes."""

    pool: Pool
    keys: dict[str, str]
    """Member file name to a printable form of its decode key. A STRING rather than the `Reading` tuple
    because the search only ever counts DISTINCT keys and never inspects one; carrying the tuple would
    invite a caller to reach into it and re-derive what `Member.decode_key` already decided."""

    @classmethod
    def over(cls, paths: list[Path]) -> "DecodeEntry":
        """Every named checkpoint that both records a holdout AND names a decode.

        The two filters are separate and both are reported, because they drop different things for
        different reasons: `RecordMix` drops a member with no holdout record or one scored under another
        label table, and `RosterReads` drops a file that describes no reading at all. A member that
        survives one and not the other is not a member for this search, and a pool silently shorter than
        the caller asked for is the wrong-denominator failure this project rules out everywhere else.
        """
        members = RosterReads.read_members(paths)
        keys = {
            member.file.name: str(RosterReads.keyed(member))
            for member in members
            if RosterReads.keyed(member) is not None
        }
        Denominator(len(keys), f"none of the {len(paths)} named checkpoints describes a decode").refuse_empty()
        # Foreign records are seated here, against the mixer's default: this search is ABOUT the decodes,
        # and a foreign member is where a distinct one comes from. What it may never do is price a seat.
        mix = RecordMix([path for path in paths if path.name in keys], foreign=True)
        for dropped in mix.narrow():
            logger.warning("%s cannot be mixed with the rest, dropped from the search", dropped)
        pool = mix.per_member()
        logger.info("searching %d members over %d distinct decodes", len(pool), len(set(keys.values())))
        return cls(pool, keys)

    def cost(self, chosen: list[int]) -> int:
        """How many decodes this subset would spend — distinct keys, which is what the kernel pays for."""
        return len({self.keys[self.pool.names[index]] for index in chosen})

    def walk(self, scores: LabelScores, max_decodes: int) -> list[BudgetedRoster]:
        """Greedy forward selection, refusing any step that would exceed the decode ceiling.

        THE CEILING IS CHECKED PER CANDIDATE, not once at the end. A member on a read already seated is
        free and one on a new read costs the budget's last unit, so the same member is affordable at
        step three and not at step nine — which is exactly the trade this tool exists to expose.
        """
        chosen: list[int] = []
        steps: list[BudgetedRoster] = []
        best_score = NO_GAIN
        while True:
            candidates = [
                index
                for index in range(len(self.pool))
                if index not in chosen and self.cost([*chosen, index]) <= max_decodes
            ]
            if not candidates:
                return steps
            scored = [
                (scores.mean(self.pool.combined([*chosen, index]), self.pool.truth), index) for index in candidates
            ]
            score, index = max(scored)
            if score <= best_score:
                return steps
            chosen.append(index)
            best_score = score
            steps.append(BudgetedRoster([self.pool.names[i] for i in chosen], score, self.cost(chosen)))

    def lines(self, scores: LabelScores, max_decodes: int) -> list[str]:
        """The walk, then the roster it ended on — the entry a slot would be spent on."""
        steps = self.walk(scores, max_decodes)
        Denominator(len(steps), f"no member is affordable within {max_decodes} decodes").refuse_empty()
        rows = [f"greedy walk under a ceiling of {max_decodes} decodes:"]
        rows.extend(step.line() for step in steps)
        best = steps[-1]
        rows.append(f"BEST at {max_decodes} decodes: {best.score:.4f} on {len(best.names)} seats")
        rows.append(f"  {' + '.join(best.names)}")
        return rows


def main() -> None:
    """Report the best-scoring roster available within a ceiling on decodes."""
    command = cli.Command(__doc__, logger)
    RosterFlags.add(command.parser)
    command.parser.add_argument(
        "--max-decodes", type=int, default=1, help="the decode budget the entry must fit inside"
    )
    arguments = command.parsed()

    paths = RosterFlags.required(arguments, command.rooted(arguments))
    scores = LabelScores()
    for line in DecodeEntry.over(paths).lines(scores, arguments.max_decodes):
        logger.info(line)


if __name__ == "__main__":
    main()
