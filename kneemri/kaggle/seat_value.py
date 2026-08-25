"""What a candidate member would add to the champion roster, card-free.

`python -m kneemri.kaggle.seat_value --candidates a.pt b.pt`

THE SECOND RULER, and the one that was missing. Every arm is read against a MATCHED control — the same
recipe with one field changed — which is the right question for `did this knob do anything` and the
wrong one for `does this earn a seat`. The two came apart once the roster stopped being the control's
family. `depth_regions_narrow` beat the fixed control `wsfold0_e6` by +0.0097 and beat its own
predecessor `depth_regions` by +0.0005; only the first number was going to be reported.

`res392_ws` is the case that forces the distinction. It scores 0.8523 alone, it was a member of the four
that scored 0.925 on the board, and over 22 same-split members the greedy takes it NINTH — the first
pick that does not raise the mix. Solo score ranked it into a submission; marginal contribution would
not have.

THE ROSTER IS READ FROM THE PIN, not typed. `comp/champion.json` holds the members of the entry that
actually scored, and `champion_check` is what keeps it true. A hand-listed roster in an arm header is a
copy that goes stale silently, and every delta under a stale roster is against the wrong base.

READ IT AS A SIGN. What transfers is the ordering, not the value. `dinov3_base_lora` reads -0.0025
against `greedy4_v1` and read -0.0013 against an earlier champion without the member changing, which is
the shrinking-edge effect a rising champion produces everywhere: a seat value is a statement about a
PAIR, and quoting one without its roster is quoting half of it.

AND THE SIGN IS READ CONSERVATIVELY, WHICH IS THE OPPOSITE OF WHAT THIS DOC USED TO SAY (knee-c0u9).
The old wording said the holdout under-prices diversity by 2.4x, so a small negative sat "inside the
region where this ruler is known to be wrong" — a licence to spend a slot on a negative reading. Two
entries were fired on exactly that theory and both were refuted. The 2.4x came off the DECODE axis, by
collapsing three members into one pixel group, and it does not generalise to a member swap. What three
paired entries measure instead is that the holdout RANKS correctly and gets only the magnitude wrong, in
the direction of UNDERSTATING what a weak member costs: crossenc / maxdiv / longseat read 0.8691 /
0.8678 / 0.8668 offline and 0.930 / 0.926 / 0.925 on the board, so -0.0013 offline became -0.004 and
-0.0023 became -0.005. An offline delta inside the noise floor is therefore not free; it is a board cost
of roughly three times its size, and "inside the floor" argues for NOT spending the slot.

TAKEN TOGETHER those two bounds leave one usable rule, and the report states it rather than leaving it
to be remembered: a candidate earns a slot only by clearing `BOARD_RESOLVABLE`, the delta the board
actually resolves. Above that the reading is real but capped; between zero and it the ruler cannot tell
the candidate from the incumbent; below zero the board charges about `BOARD_COST_MULTIPLE` times what is
shown. `MemberCurve.swap` prints whichever of the three applies.

AND IT SATURATES, which bounds what a positive reading is worth (knee-825t). Six entries now pair an
offline joint-holdout score with a board score and the chain rises over the first 0.003 of offline
spread, then stops: `longseat` carried +0.0012 of seat value — the largest single move in a sweep of all
56 same-split checkpoints, six times the second-best — and tied the board rather than moving it. So this
tool answers `which of these is the best move` reliably and `is the best move worth a slot` not at all.
Below roughly 0.0015 its winner and its runner-up are the same submission.

AND SINCE 2026-08-22 IT DOES NOT PRINT ALONE. Everything above is a statement about the QUALITY axis, and
the board split that closed that day was decided on the other one: `lr4e-5_ws` priced -0.0003 here and
took 0.931 while `band_span` priced +0.0002 and cost a tick, because the second was a near-twin of a
seated member and a rank average cannot charge for contributing one ordering twice. `MemberDistance`
appends the label-free distance reading to this report for that reason. The two columns disagree on that
pair, deliberately visibly — quoting this one without the other is quoting the ruler that was wrong about
the last decision it was asked to make.
"""

import logging
from pathlib import Path

from core import cli
from kneemri.evaluation.member_curve import COMBINE_SEED, MemberCurve
from kneemri.evaluation.member_distance import MemberDistance
from kneemri.evaluation.record_mix import MixReading, RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.kaggle.champion_check import BOARD, Champion
from kneemri.kaggle.priceable_entries import PriceableEntries
from kneemri.runs.holdout_record import HoldoutRecord

logger = logging.getLogger(__name__)

PRICEABLE = 2
"""The smallest roster a seat value can be priced against, and it is a claim about meaning not a guard.

A marginal contribution is what a candidate adds to a MIX. Against one member there is no mix — the
number would be a pairwise comparison wearing a roster's name, and every arm's log would carry it as
though it had been priced against the entry that scored.

This is not hypothetical arithmetic. On 2026-08-24 the pinned champion became `allfit`, six of whose
seven members were fitted at `--fold -1` and are unscoreable forever, leaving exactly one. Every arm
that finished after that printed a traceback where its seat should be, and the ruler that decides
whether a member ships went dark as a side effect of the board improving (knee-1gdf).
"""


class SeatValue:
    """The pinned champion's members, and what each candidate would add to them."""

    def __init__(self, champion: Champion, processed: Path, board: Path = BOARD) -> None:
        self.champion = champion
        self.processed = processed
        self.board = board

    def roster(self) -> list[Path]:
        """The champion's members as checkpoints on disk.

        A member the pin names but the directory does not hold is left for `RecordMix.split` to refuse.
        Reporting it here as well would put the same failure in two places with two wordings, and the
        one downstream is the one that also catches a member present but recordless.
        """
        return [self.processed / f"{name}.pt" for name in self.champion.members]

    def scoreable(self, roster: list[Path]) -> tuple[list[Path], list[Path]]:
        """The roster split into the members that carry a holdout record and the ones that never can.

        The shipped champion contains a member trained with `--fold -1`, which held nothing out and is
        therefore unscoreable FOREVER rather than pending. `RecordMix.split` refuses such a roster, and
        that refusal is right: a roster silently scored at 6 of 7 makes every delta below it read against
        the wrong base. But the refusal leaves the entry that ACTUALLY SCORED unpriceable against any
        candidate, which is the wrong end of the trade.

        So the roster is priced against its scoreable subset and the denominator is stated in the output.
        A labelled proxy is honest arithmetic; the failure this guards is the SILENT one.
        """
        priced = [member for member in roster if HoldoutRecord(member).exists()]
        return priced, [member for member in roster if member not in priced]

    def named(self, candidate: Path) -> Path:
        """A candidate given by NAME becomes the checkpoint of that name; one given by path is left alone.

        `roster` has always applied this rule to the champion's own members and nothing applied it to the
        candidates beside them, so a candidate named the way every arm in this repository is named — and
        the way this module's own usage line spells it, `--candidates a.pt b.pt` — resolved against the
        CURRENT DIRECTORY. It is not there, so it carried no holdout record, so it was dropped with one
        `no holdout record, skipped` line and the report still printed a complete-looking roster figure.
        That is the failure mode worth the method: a silently unpriced candidate reads exactly like a
        candidate priced at nothing, and `band_span_s2026` was read that way once before this existed.
        """
        return self.processed / candidate.name if candidate.parent == Path() else candidate

    def unpriceable(self, roster: list[Path], unscoreable: list[Path]) -> list[str]:
        """Why no seat value exists for this entry, with the denominator that makes it a fact.

        A refusal is the deliverable here, not an error path: the alternative is a number priced against
        fewer members than the entry has, printed under that entry's name. The way out is named in the
        last line, because a refusal a reader cannot act on is only half of one.
        """
        return [
            f"NO SEAT VALUE against {self.champion.entry}: {len(roster)} of "
            f"{len(roster) + len(unscoreable)} members carry a holdout record, and pricing a marginal "
            f"contribution needs at least {PRICEABLE}",
            f"unscoreable (fold -1, they have seen every holdout study): "
            f"{', '.join(sorted(member.stem for member in unscoreable))}",
            PriceableEntries(self.processed, self.board).suggestion(PRICEABLE),
        ]

    def report(self, candidates: list[Path], combine_seed: int = COMBINE_SEED) -> list[str]:
        """Each candidate's marginal contribution to the champion, best first — and its DISTANCE beside it.

        The two are printed together because the 2026-08-21 split is a statement about the pair, not
        about either alone: `lr4e-5_ws` priced -0.0003 here and took 0.931, `band_span` priced +0.0002
        and cost a tick, and the distance column is the one that orders them the way the board did.
        Quoting a seat value without it is quoting the ruler that was wrong about the last decision it
        was asked to make.
        """
        roster, unscoreable = self.scoreable(self.roster())
        if len(roster) < PRICEABLE:
            return self.unpriceable(roster, unscoreable)
        named = [self.named(candidate) for candidate in candidates]
        mixed = RecordMix(list(dict.fromkeys([*roster, *named]))).per_member()
        curve = MemberCurve(LabelScores(), combine_seed)
        held, rest = mixed.split(roster)
        denominator = (
            f"priced against {len(roster)} of {len(roster) + len(unscoreable)} champion members; "
            f"{', '.join(sorted(member.stem for member in unscoreable))} unscoreable (fold -1)"
            if unscoreable
            else f"priced against all {len(roster)} champion members"
        )
        return [
            denominator,
            *MixReading(mixed, roster).lines(curve, None, paired=False, labels=False),
            *MemberDistance(held, rest).report(),
        ]


def main() -> None:
    """Price candidate members against the entry that actually scored."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--candidates", type=Path, nargs="+", required=True, help="checkpoints to price")
    command.parser.add_argument("--combine-seed", type=int, default=COMBINE_SEED)
    command.parser.add_argument(
        "--against",
        default=None,
        help="board entry id to price against instead of the pinned champion (see comp/board.jsonl)",
    )
    arguments = command.parsed()

    champion = Champion.from_board(arguments.against) if arguments.against else Champion.recorded()
    logger.info("%s scored %.3f on the board with %d members", champion.entry, champion.board, len(champion.members))
    seats = SeatValue(champion, command.rooted(arguments).processed())
    command.report(seats.report(arguments.candidates, arguments.combine_seed))


if __name__ == "__main__":
    main()
