"""Is the champion's plateau reducible at all? `python -m kneemri.kaggle.champion_uncertainty`.

`UncertaintySplit` owns the arithmetic and knows nothing about entries; this names the members. The
default is the pinned champion, read from `comp/champion.json` the way `seat_value` reads it and for
the same reason — a hand-listed roster in a header is a copy that goes stale, and a decomposition of
last week's entry is a statement about nothing anyone ships.

WHAT IT ANSWERS. `epistemic / total` over the entry's own members: the share of its remaining
uncertainty that a different or better member could still remove. Low says the members genuinely agree
and the reseating search is dead ground whatever the pool grows to; high says they disagree and the
combination rule is discarding it. Either reading redirects card time, which is why it is worth a
command and not a note.

WHAT IT REFUSES, AND HOW OFTEN THAT WILL BE THE ANSWER. Members fitted at `--fold -1` have seen every
holdout study, so they can carry no decomposition — and the board's entire top is built from them. The
pinned champion at the time of writing has ONE of six members scoreable, so the honest output for it is
a refusal naming the other five plus the entries that could answer instead, exactly as `seat_value`'s
is. That is not this tool failing: the all-data axis bought board position by spending the ability to
measure, and a fraction printed over whichever members happened to survive would be the wrong
denominator wearing a result's clothes.

A PARTIAL ENTRY IS DECOMPOSED AND SAYS SO. Where enough members remain, the split is reported over
them with the dropped ones named in the denominator — a labelled subset is honest arithmetic about a
smaller pool, and it is the silent version this refuses.
"""

import logging
from pathlib import Path

from core import cli
from kneemri.evaluation.uncertainty_split import DECOMPOSABLE, UncertaintySplit
from kneemri.kaggle.champion_check import BOARD, Champion
from kneemri.kaggle.package_members import MemberRoster
from kneemri.kaggle.priceable_entries import PriceableEntries
from kneemri.runs.holdout_record import HoldoutRecord

logger = logging.getLogger(__name__)


class ChampionUncertainty:
    """A named set of shipped members, and the reducible share of what they still do not know."""

    def __init__(self, processed: Path, board: Path = BOARD) -> None:
        self.processed = processed
        self.board = board

    def members(self, champion: Champion) -> list[Path]:
        """The entry's members as checkpoints under the processed root — the pin stores bare names."""
        return [self.processed / f"{name}.pt" for name in champion.members]

    def held(self, members: list[Path]) -> tuple[list[Path], list[Path]]:
        """The members that held something out, and the ones that can never be scored on this holdout.

        Split HERE rather than left to `UncertaintySplit.over`, which refuses a recordless member by
        name and is right to: its caller asked for exactly those members. This caller asked for an
        ENTRY, and an entry's unscoreable members are a property of the entry rather than a mistake, so
        they are named in the denominator instead of ending the report.
        """
        scoreable = [member for member in members if HoldoutRecord(member).exists()]
        return scoreable, [member for member in members if member not in scoreable]

    def undecomposable(self, scoreable: list[Path], unscoreable: list[Path]) -> list[str]:
        """Why this entry carries no decomposition, with the denominator that makes it a fact.

        The way out is named rather than left to the reader, because a refusal nobody can act on is
        half of one — and the entries that CAN answer are already surveyed by `PriceableEntries`.
        """
        return [
            f"NO UNCERTAINTY SPLIT: {len(scoreable)} of {len(scoreable) + len(unscoreable)} members "
            f"carry a holdout record, and a split needs at least {DECOMPOSABLE}",
            f"unscoreable (fold -1, they have seen every holdout study): "
            f"{', '.join(sorted(member.stem for member in unscoreable))}",
            PriceableEntries(self.processed, self.board).suggestion(DECOMPOSABLE),
        ]

    def report(self, members: list[Path]) -> list[str]:
        """The split over whichever of these members held something out, or the refusal instead."""
        scoreable, unscoreable = self.held(members)
        if len(scoreable) < DECOMPOSABLE:
            return self.undecomposable(scoreable, unscoreable)
        dropped = (
            [f"dropped, fold -1 and unscoreable: {', '.join(sorted(member.stem for member in unscoreable))}"]
            if unscoreable
            else []
        )
        return [*dropped, *UncertaintySplit.over(scoreable).report()]


def main() -> None:
    """Decompose a shipped entry's remaining uncertainty into its reducible and irreducible halves."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--against",
        default=None,
        help="board entry id to decompose instead of the pinned champion (see comp/board.jsonl). The "
        "champion is usually the wrong subject through no fault of its own: the board's top entries are "
        "built from --fold -1 members and can carry no split at all.",
    )
    command.parser.add_argument(
        "--roster",
        type=Path,
        default=None,
        help="decompose every member this roster file packages instead of an entry's — the widest "
        "honest pool the project owns, rather than the six seats one submission happened to hold",
    )
    arguments = command.parsed()

    processed = command.rooted(arguments).processed()
    split = ChampionUncertainty(processed)
    if arguments.roster:
        logger.info("decomposing the members %s packages", arguments.roster)
        command.report(split.report(MemberRoster.read(arguments.roster).checkpoints(processed)))
        return
    champion = Champion.from_board(arguments.against) if arguments.against else Champion.recorded()
    logger.info("%s scored %.3f on the board with %d members", champion.entry, champion.board, len(champion.members))
    command.report(split.report(split.members(champion)))


if __name__ == "__main__":
    main()
