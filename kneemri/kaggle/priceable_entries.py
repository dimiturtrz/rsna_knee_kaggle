"""Which scored entries can still price a seat, since the ones that scored best cannot.

`seat_value`'s refusal used to end at "price against a scoreable entry instead: --against <board entry
id>", which reads as actionable and is not: the caller then has to open `comp/board.jsonl`, resolve every
row's members to checkpoints, and test each for a holdout record before it can fill in that placeholder.

Done by hand on 2026-08-25 that survey said something the refusal could have said itself — MOST of the
board's top is unpriceable, and structurally so. An entry built from the `_ff` members, fitted at
`--fold -1`, has seen every holdout study, so 0.938 `votefit_v1`, 0.937 `dark6_v1` and 0.937 `dark7_v1`
price nothing at all.

THE TOP IS NOT ENTIRELY UNPRICEABLE, and this docstring said it was until 2026-09-05. The claim rested
on `EntryOffline`, which resolved holdout records against `root.processed()` while every record lives
under `root.models()` — so the check could never succeed and every entry that survived the in-sample
test was falsely declined `no-record`. With that one word fixed, `distinct14` prices at offline 0.8749
against a board 0.937 and `fitfour_v1` at 0.8711, and `board_transfer`'s top regime went from two
resolved pairs to four. `distinct14` is now the anchor every board prediction in this package uses; the
highest fully scoreable entry is NOT `gradeswap_v1` at 0.933 (knee-q40e), which is what a survey run
against a broken record reported.

THE GAP BETWEEN THOSE TWO IS NOT A QUALITY GAP and a caller must not read it as one. Converting five of
seven members to full-fit moved `finalfit_v1` 0.934 to `allfit_v1` 0.937; converting the SEVENTH bought
nothing at all (`lastseat_v1` 0.937 with `small_batch6_s2026`, `dark7_v1` 0.937 with `small_batch6_ff`,
one flag apart). So the difference this class steps down by is a dose effect that has already saturated,
plus the fusion tick `votefit` added on top — see knee-97ib. Pricing stands in fold-0 space; this only
says where.
"""

import json
import logging
from pathlib import Path

from kneemri.kaggle.champion_check import BOARD, PINNABLE, Champion
from kneemri.runs.holdout_record import HoldoutRecord

logger = logging.getLogger(__name__)

SUGGESTED = 3
"""How many alternatives a refusal names. Enough to show the range it is choosing from, few enough that
the line stays a suggestion rather than the survey it was read out of."""


class PriceableEntries:
    """The board entries whose every member can still be scored on the holdout."""

    def __init__(self, processed: Path, board: Path = BOARD) -> None:
        self.processed = processed
        self.board = board

    def entries(self) -> list[Champion]:
        """Every board row that is an entry at all, skipping the ones nothing can be rebuilt from.

        A row can be present and unusable — 31 of 48 were written without the fields that make an entry
        a comparison target — and `from_board` says so by raising. Here that is not an error: a survey
        of what CAN price a seat has no business failing on a row that cannot.
        """
        rows = [json.loads(line) for line in self.board.read_text(encoding="utf-8").splitlines() if line.strip()]
        held = []
        for row in rows:
            if row.get("id") is None or row.get("scope") != PINNABLE:
                continue
            try:
                held.append(Champion.from_board(str(row["id"]), self.board))
            except ValueError:
                continue
        return held

    def fully(self, minimum: int) -> list[Champion]:
        """The entries whose EVERY member carries a holdout record, best board score first.

        Fully, not mostly: a partially scoreable roster is exactly what `SeatValue.scoreable` already
        falls back to, and suggesting one here would hand the caller a second labelled proxy when a
        clean base exists. `minimum` is passed rather than assumed so the suggestion can never name an
        entry the caller's own threshold would refuse in turn.
        """
        whole = [
            entry
            for entry in self.entries()
            if len(entry.members) >= minimum
            and all(HoldoutRecord(self.processed / f"{name}.pt").exists() for name in entry.members)
        ]
        return sorted(whole, key=lambda entry: -entry.board)

    def suggestion(self, minimum: int, shown: int = SUGGESTED) -> str:
        """The refusal's last line, with the entries that would answer it named."""
        whole = self.fully(minimum)[:shown]
        if not whole:
            return f"and NO board entry is fully scoreable at {minimum} members or more, so no seat can be priced"
        named = ", ".join(f"{entry.entry} ({entry.board:.3f}, {len(entry.members)} members)" for entry in whole)
        return f"price against a fully scoreable entry instead: --against {whole[0].entry}   [{named}]"
