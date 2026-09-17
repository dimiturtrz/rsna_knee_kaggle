"""Why is a board entry's `offline` null? `python -m kneemri.kaggle.entry_offline [--write]`.

`comp/board.jsonl` carries 74 submissions and 34 of them record members but no offline number. A null
there reads as an oversight, and it was treated as one: knee-awxb was filed to "back-fill the missing
holdout reads" as if the numbers were simply unwritten. Most of them cannot be written at all, and the
reason differs per row. This says which reason, per row, and fills the ones that CAN be filled.

WHY IT MATTERS THAT THE NULLS ARE EXPLAINED RATHER THAN FILLED. `board_transfer` fits the offline-to-board
compression from the pairs this file resolves, and its top regime rests on three of them. A wrongly filled
row is worse than a null: `raptor_solo` carried offline 0.8938 read on a holdout that `foreign_overlap`
then showed was 98.6% inside that member's own training set, and eleven of fifteen apparent ruler
inversions were that one contaminated cell. So the rule here is that a number is written only when every
member of the entry held out the studies it is scored on, and every other row says out loud why not.

THE THREE REASONS, and none of them is an oversight:

- IN-SAMPLE: a member fitted at `--fold -1` (`_ff`, or `final_ws_e12`) has seen every holdout study.
  `BoardCensus.unpriceable` already refuses a row that records offline over such a pool; this names the
  same rows from the other side, before a number exists. Every board entry at 0.936 and above is one.
- FOREIGN-IN-SAMPLE: a foreign checkpoint whose published training rows cover our holdout — measured, not
  assumed, by `kneemri.evaluation.foreign_overlap`.
- NO-RECORD: a member that left no `HoldoutRecord`. The `_cf` cross-fit merges are the live case; older
  members predate the file.

WHAT IS LEFT AFTER THAT, and it is not nothing: ten entries between board 0.921 and 0.933, six of them
inside the 0.930-0.937 regime whose slope was fitted on three pairs. The compression ABOVE 0.937 stays
unmeasured and structurally so — that is the finding, not a gap to fill later.

THE VOTE RULE IS AN ARGUMENT AND ITS DEFAULT IS RANK, because every entry this can fill shipped before
`kneemri-vote` served `LogitAverage`. Scoring an old entry under the rule it did not serve would put a
number in the record that no submission ever computed.
"""

import json
import logging
from collections.abc import Callable
from pathlib import Path

from core import cli
from kneemri.evaluation.ledger75 import LedgerSeventyFive
from kneemri.evaluation.record_mix import VOTE_RULES, RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.kaggle.board_check import RECORD, BoardCensus, BoardRow
from kneemri.runs.holdout_record import HoldoutRecord

logger = logging.getLogger(__name__)

RULERS: dict[str, Callable[[], LabelScores]] = {
    "midpoint": LabelScores,
    "ledger75": lambda: LedgerSeventyFive().scorer(),
}
"""The cut an entry's offline number is read under, and the column that number goes in.

`board_transfer:88` has mapped `ledger75 -> offline75` since it was written, and NOT ONE ROW HAS EVER
CARRIED THAT FIELD — `git log -S offline75` finds the reader and no data commit. So the `--ruler ledger75`
arm of the transfer fit has always dropped every row and reported on an empty set, while `ledger75.py:26-28`
cites board evidence out of that arm. This is the producer that column never had.
"""

COLUMNS: dict[str, str] = {"midpoint": "offline", "ledger75": "offline75"}

FRAME_GAP = 0.0062
"""How far a recomputed midpoint may sit from a recorded `offline` before the two are different questions.

The arm-vs-arm floor this project quotes everywhere, borrowed deliberately: below it the gap between a
row's recorded number and a fresh read of the same pool is not something a reader could act on, and above
it the row is quoting a DIFFERENT statistic under the same column name. `nine_v1` is the case it was
written for — its `offline` is 0.9026 @ledger75, its midpoint is 0.8759, and the 0.0267 between them is
four times the board gap sitting in the column every midpoint consumer reads.
"""

SCOPE_KEY = "offline_scope"
"""The field a row carries INSTEAD of an offline number, naming which of the reasons applies.

Written beside the null rather than into `why`, because `why` is prose a person wrote about a result and
a check cannot parse it. A machine-readable scope is what lets the next pass be an exact set comparison
— the same argument `board_check` makes for `scope: excluded`.
"""

IN_SAMPLE = "in-sample: fitted at --fold -1, has seen every holdout study"
FOREIGN_IN_SAMPLE = "foreign-in-sample: the foreign member's published training rows cover our holdout"
NO_RECORD = "no-record: these members left no holdout predictions file"

FOREIGN_HOLDOUT_SEEN = frozenset({"raptor_v4_swa"})
"""Foreign checkpoints measured to have trained on our holdout — 896 of 909 studies for this one.

A frozenset of names rather than a rule about names, because the measurement is per checkpoint and lives
in `foreign_overlap`; a heuristic on the string would either miss a future foreign member or refuse a
clean one.
"""

DEFAULT_VOTE = "rank"

EXCLUDED_SCOPE = "excluded"
"""The `scope` value that keeps a row out of `board_transfer`'s fit, and out of this file's writes."""


class EntryOffline:
    """Each board entry's offline number, or the reason it does not get one."""

    def __init__(
        self, processed: Path, record: Path = RECORD, vote: str = DEFAULT_VOTE, ruler: str = "midpoint"
    ) -> None:
        self.processed = processed
        self.record = record
        self.vote = vote
        self.ruler = ruler
        self.scores = RULERS[ruler]()
        self.column = COLUMNS[ruler]

    def stored(self) -> dict[str, dict[str, object]]:
        """Every recorded line by entry id, because a column this file writes is one `BoardRow` cannot see."""
        found: dict[str, dict[str, object]] = {}
        for line in self.record.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row: dict[str, object] = json.loads(line)
                found[str(row.get("id", row.get("ref", "")))] = row
        return found

    def rows(self) -> list[BoardRow]:
        """The entries this file has anything to say about: members named, this ruler's column still empty.

        An `excluded` row is none of its business. The record has already ruled that row out of the fit
        in the field the fitter reads, and a number written into a cell that has been ruled out reads as
        a price on a re-run — which is how this ran once and silently restored two voided offlines.
        """
        stored = self.stored()
        return [
            row
            for row in BoardCensus(record=self.record).rows()
            if row.members_named
            and stored.get(row.identifier, {}).get(self.column) is None
            and stored.get(row.identifier, {}).get("scope") != EXCLUDED_SCOPE
        ]

    def declined(self, row: BoardRow) -> str | None:
        """Why this entry cannot be scored on our holdout, or None when it can.

        Order matters: an entry can be in-sample AND missing a record, and the in-sample reason is the
        one that would still stand if every record were written.
        """
        if row.all_data():
            return f"{IN_SAMPLE} ({', '.join(row.all_data())})"
        seen = [name for name in row.members_named if name in FOREIGN_HOLDOUT_SEEN]
        if seen:
            return f"{FOREIGN_IN_SAMPLE} ({', '.join(seen)})"
        missing = [name for name in row.members_named if not HoldoutRecord(self.processed / f"{name}.pt").exists()]
        if missing:
            return f"{NO_RECORD} ({', '.join(missing)})"
        return None

    def scored(self, row: BoardRow) -> float:
        """This entry's members, combined under the vote rule and scored on the studies they all held out.

        Foreign records are seated here, against the mixer's default, because this reproduces an entry
        that WENT TO THE BOARD — dropping a member it shipped would price a different ensemble under its
        name. The contamination such a member carries is the concern of `declined`, which voids the row.
        """
        mix = RecordMix([self.processed / f"{name}.pt" for name in row.members_named], foreign=True)
        pool = mix.per_member(VOTE_RULES[self.vote]())
        return self.scores.mean(pool.combined(list(range(len(pool)))), pool.truth)

    def filled(self) -> dict[str, float]:
        """The offline number for every entry that can carry one, by entry id."""
        return {row.identifier: self.scored(row) for row in self.rows() if self.declined(row) is None}

    def scopes(self) -> dict[str, str]:
        """The reason for every entry that cannot, by entry id."""
        return {row.identifier: reason for row in self.rows() if (reason := self.declined(row)) is not None}

    def mismatched(self) -> dict[str, tuple[float, float]]:
        """Rows whose RECORDED `offline` a fresh midpoint read cannot reproduce, by entry id.

        The control on everything else here. A recomputed column is only worth writing if the same code
        path reproduces the column that already exists, so this runs the midpoint arm over the rows that
        already carry a number and names the ones that come back further than `FRAME_GAP` away. A row that
        fails is not a tolerance to widen — it is a cell holding a statistic other than the one its column
        name promises, which is `knee-f9aao` on the board file.
        """
        midpoint = LabelScores()
        found: dict[str, tuple[float, float]] = {}
        for row in BoardCensus(record=self.record).rows():
            if row.offline is None or not row.members_named or self.declined(row) is not None:
                continue
            mix = RecordMix([self.processed / f"{name}.pt" for name in row.members_named], foreign=True)
            pool = mix.per_member(VOTE_RULES[self.vote]())
            fresh = midpoint.mean(pool.combined(list(range(len(pool)))), pool.truth)
            if abs(fresh - row.offline) > FRAME_GAP:
                found[row.identifier] = (row.offline, fresh)
        return found

    def report(self) -> list[str]:
        """One line per entry, the fillable ones first — the survey knee-awxb was filed to produce."""
        filled = self.filled()
        scopes = self.scopes()
        lines = [f"{name} {self.column} {score:.4f} under {self.vote}" for name, score in filled.items()]
        lines += [f"{name} DECLINED {reason}" for name, reason in scopes.items()]
        return [
            *lines,
            f"{len(filled)} fillable, {len(scopes)} declined, of {len(self.rows())} entries missing {self.column}",
        ]

    def written(self) -> int:
        """Write the numbers and the scopes back into the record, and say how many rows changed.

        Rewritten line by line from the original text so a row this file has no opinion about keeps its
        exact bytes — the record is reviewed as a diff, and a wholesale re-serialisation would bury one
        filled cell under seventy reformatted ones.
        """
        filled = self.filled()
        scopes = self.scopes()
        changed = 0
        lines = []
        for line in self.record.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                lines.append(line)
                continue
            row: dict[str, object] = json.loads(line)
            identifier = str(row.get("id", ""))
            if identifier in filled:
                row[self.column] = round(filled[identifier], 4)
                changed += 1
            elif identifier in scopes and row.get(SCOPE_KEY) != scopes[identifier]:
                row[SCOPE_KEY] = scopes[identifier]
                changed += 1
            lines.append(json.dumps(row))
        self.record.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return changed


def main() -> None:
    """Survey the board's null offline cells, and optionally fill the ones that can be filled."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--record", type=Path, default=RECORD, help="the submission record")
    command.parser.add_argument("--vote", choices=tuple(VOTE_RULES), default=DEFAULT_VOTE)
    command.parser.add_argument("--ruler", choices=tuple(RULERS), default="midpoint", help="the cut, and the column")
    command.parser.add_argument("--check", action="store_true", help="recompute the recorded midpoint column")
    command.parser.add_argument("--write", action="store_true", help="write the numbers and scopes back")
    arguments = command.parsed()
    root = command.rooted(arguments)

    entries = EntryOffline(root.models(), arguments.record, arguments.vote, arguments.ruler)
    if arguments.check:
        gaps = entries.mismatched()
        command.report([f"{name} recorded {was:.4f}, recomputed {now:.4f}" for name, (was, now) in gaps.items()])
        logger.info("%d row(s) further than %.4f from a fresh midpoint read", len(gaps), FRAME_GAP)
        return
    command.report(entries.report())
    if arguments.write:
        logger.info("%d row(s) changed in %s", entries.written(), arguments.record)


if __name__ == "__main__":
    main()
