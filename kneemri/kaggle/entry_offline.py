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
from pathlib import Path

from core import cli
from kneemri.evaluation.record_mix import VOTE_RULES, RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.kaggle.board_check import RECORD, BoardCensus, BoardRow
from kneemri.runs.holdout_record import HoldoutRecord

logger = logging.getLogger(__name__)

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


class EntryOffline:
    """Each board entry's offline number, or the reason it does not get one."""

    def __init__(self, processed: Path, record: Path = RECORD, vote: str = DEFAULT_VOTE) -> None:
        self.processed = processed
        self.record = record
        self.vote = vote

    def rows(self) -> list[BoardRow]:
        """The entries this file has anything to say about: members named, offline still null."""
        return [row for row in BoardCensus(record=self.record).rows() if row.offline is None and row.members_named]

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
        return LabelScores().mean(pool.combined(list(range(len(pool)))), pool.truth)

    def filled(self) -> dict[str, float]:
        """The offline number for every entry that can carry one, by entry id."""
        return {row.identifier: self.scored(row) for row in self.rows() if self.declined(row) is None}

    def scopes(self) -> dict[str, str]:
        """The reason for every entry that cannot, by entry id."""
        return {row.identifier: reason for row in self.rows() if (reason := self.declined(row)) is not None}

    def report(self) -> list[str]:
        """One line per entry, the fillable ones first — the survey knee-awxb was filed to produce."""
        filled = self.filled()
        scopes = self.scopes()
        lines = [f"{name} offline {score:.4f} under {self.vote}" for name, score in filled.items()]
        lines += [f"{name} DECLINED {reason}" for name, reason in scopes.items()]
        return [*lines, f"{len(filled)} fillable, {len(scopes)} declined, of {len(self.rows())} entries with a null"]

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
                row["offline"] = round(filled[identifier], 4)
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
    command.parser.add_argument("--write", action="store_true", help="write the numbers and scopes back")
    arguments = command.parsed()
    root = command.rooted(arguments)

    entries = EntryOffline(root.models(), arguments.record, arguments.vote)
    command.report(entries.report())
    if arguments.write:
        logger.info("%d row(s) changed in %s", entries.written(), arguments.record)


if __name__ == "__main__":
    main()
