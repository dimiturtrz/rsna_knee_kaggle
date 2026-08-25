"""Is the submission record COMPLETE? `python -m kneemri.kaggle.board_check --live`.

`comp/board.jsonl` is the only machine-readable account of what this project has actually scored, and
`board_transfer` fits the offline-to-board compression from it. That makes its COMPLETENESS a property of
every number derived from it: a missing row is not a gap in a document, it is a point silently dropped
from a fit that decides which entry gets a slot. On 2026-08-25 the file held 34 of the competition's 58
submissions. Fifteen scored all-ours entries were absent, including `basedense_v1` — the entry pinned in
`comp/champion.json` — and `votefit_v1`, the best score this project owns. The compression constant read
+1.82 against +1.72 once they were entered, and that constant is quoted inside shipped kernels.

THE JOIN IS THE SUBMISSION REF, not a name. Kernel names repeat across entries (`sixfresh` shipped
twice), prose spells them freely ("sixfresh v2", "rate member displaces sliceattn"), and a heading score
is not unique — seven entries read 0.937. The ref is Kaggle's own identifier, it is unique, and asking
Kaggle for the list costs no submission slot.

EVERY submission gets a row, the ones this file will not price included. An entry excluded for a good
reason (a mixed fusion whose members were fitted elsewhere, a 0.500 kit failure, a run that returned no
score) carries `scope: excluded` and the reason, so the exclusion is a recorded decision rather than an
absence. Only then can the check be an exact set comparison instead of a heuristic about which absences
were meant.
"""

import json
import logging
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from core import cli

logger = logging.getLogger(__name__)

RECORD = Path("comp/board.jsonl")
LEDGER = Path("comp/SUBMISSIONS.md")
COMPETITION = "rsna-knee-abnormality-detection"
EXCLUDED = "excluded"
"""The scope of a submission this record deliberately does not price. `board_transfer` already reports
per-scope drops, so an excluded row is counted out loud rather than missing."""

SCORE_TOLERANCE = 5e-4
"""A board score is quoted to three decimals; anything wider than half a tick is a transcription error."""

ALL_DATA_SUFFIX = "_ff"
ALL_DATA_NAMED = frozenset({"final_ws_e12"})
"""How an all-data (`--fold -1`) member is spelled in the roster: the `_ff` suffix, plus the one member
that predates the convention. A pool holding one has seen every holdout study, so an offline number
beside it is not a held-out number — that is the entity error that produced a bad anchor on 2026-08-24.
This is a NAME rule on purpose: the disk rule (does a holdout record exist) needs the data root, and a
record check must run in CI where there is none."""


@dataclass(frozen=True)
class BoardRow:
    """One recorded submission — what the board said, and what this file is willing to say about it."""

    ref: str
    identifier: str
    scope: str
    board: float | None
    offline: float | None
    members_named: tuple[str, ...]

    @classmethod
    def parsed(cls, row: Mapping[str, object]) -> "BoardRow":
        """One JSONL line as a row. An excluded line carries no id, so its ref stands in as one."""
        ref = str(row.get("ref", ""))
        named = row.get("members_named")
        seats: list[object] = named if isinstance(named, list) else []
        return cls(
            ref=ref,
            identifier=str(row.get("id", ref)),
            scope=str(row.get("scope", "")),
            board=None if row.get("board") is None else float(str(row["board"])),
            offline=None if row.get("offline") is None else float(str(row["offline"])),
            members_named=tuple(str(name) for name in seats),
        )

    def disagrees(self, scored: float | None) -> bool:
        """Whether this row's board number says something other than what the board holds.

        The null case is checked BEFORE the float compare and not folded into it: a row recording a
        score for a submission that returned none, and a row recording none for a submission that
        scored, are both defects, and `abs(None - x)` is the crash that would hide them.
        """
        if (self.board is None) != (scored is None):
            return True
        return self.board is not None and scored is not None and abs(self.board - scored) > SCORE_TOLERANCE

    def all_data(self) -> list[str]:
        """The seats in this entry that held nothing out, empty when the entry named no members.

        An entry that names no members says nothing either way, which is why the offline rule below can
        only ever fire on entries that DID name them. Silence is not a pass here, it is a silence — and
        naming the members is what buys the check.
        """
        return [name for name in self.members_named if name.endswith(ALL_DATA_SUFFIX) or name in ALL_DATA_NAMED]


class BoardCensus:
    """The recorded submissions, asked whether they are all of them and whether each says a true thing."""

    def __init__(self, record: Path = RECORD, ledger: Path = LEDGER) -> None:
        self.record = record
        self.ledger = ledger

    def rows(self) -> list[BoardRow]:
        """Every recorded entry, the notes skipped — they carry commentary and no submission."""
        lines = [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines() if line]
        return [BoardRow.parsed(row) for row in lines if "note" not in row]

    def refless(self) -> list[str]:
        """Rows carrying no submission ref — the ones no live comparison can reach at all."""
        return [row.identifier for row in self.rows() if not row.ref]

    def duplicated(self) -> list[str]:
        """Refs recorded more than once. Two rows for one submission make it two points in every fit."""
        refs = [row.ref for row in self.rows() if row.ref]
        return sorted({ref for ref in refs if refs.count(ref) > 1})

    def unpriceable(self) -> list[str]:
        """Entries carrying an offline number over a pool that has seen the holdout it was scored on."""
        return [
            f"{row.identifier} records offline {row.offline} over an all-data pool ({', '.join(row.all_data())})"
            for row in self.rows()
            if row.offline is not None and row.all_data()
        ]

    def unnamed(self) -> list[str]:
        """Priced entries whose id stem the prose ledger never mentions — the record's other half.

        Excluded rows are exempt: the prose ledger is where RECIPES are written up, and a mixed fusion or
        a kit failure earns a row here precisely because it is not one.
        """
        prose = self.ledger.read_text(encoding="utf-8")
        return [
            row.identifier
            for row in self.rows()
            if row.scope != EXCLUDED and row.identifier.rsplit("_v", 1)[0] not in prose
        ]

    def unrecorded(self, live: Mapping[str, float | None]) -> list[str]:
        """Submissions Kaggle reports that this file has never heard of — the failure that was live."""
        recorded = {row.ref for row in self.rows()}
        return sorted(ref for ref in live if ref not in recorded)

    def phantom(self, live: Mapping[str, float | None]) -> list[str]:
        """Recorded refs Kaggle does not report — a typo in a ref, or a row for a submission never made."""
        return sorted(row.ref for row in self.rows() if row.ref and row.ref not in live)

    def mismatched(self, live: Mapping[str, float | None]) -> list[str]:
        """Rows whose recorded board score disagrees with the score Kaggle holds for that submission."""
        return [
            f"{row.identifier} records {row.board} and the board holds {live[row.ref]}"
            for row in self.rows()
            if row.ref in live and row.disagrees(live[row.ref])
        ]

    def refusals(self, live: Mapping[str, float | None] | None) -> list[str]:
        """Every defect found, in the order a reader should act on them; empty when the record is whole."""
        found = [f"{name} carries no ref, so no live listing can reach it" for name in self.refless()]
        found += [f"ref {ref} is recorded twice" for ref in self.duplicated()]
        found += self.unpriceable()
        found += [f"{name} is priced here and named nowhere in {self.ledger}" for name in self.unnamed()]
        if live is None:
            return found
        found += [f"submission {ref} scored and is recorded NOWHERE" for ref in self.unrecorded(live)]
        found += [f"ref {ref} is recorded and the board does not hold it" for ref in self.phantom(live)]
        return found + self.mismatched(live)


@dataclass(frozen=True)
class LiveBoard:
    """What Kaggle says this competition holds for us: every submission's ref against its public score.

    Injected lister rather than a call inline in `main`, for the reason `mount_check.LiveReading` gives:
    a live path reachable only through real credentials is a live path with no test.
    """

    scores: dict[str, float | None]

    @classmethod
    def taken(cls, lister: Callable[[], list[tuple[str, str]]] | None = None) -> "LiveBoard":
        """Ask Kaggle, or ask whatever was injected in its place. Page size covers the whole history."""
        listed = lister() if lister is not None else cls._kaggle()
        return cls({ref: float(score) if score else None for ref, score in listed})

    @staticmethod
    def _kaggle() -> list[tuple[str, str]]:
        """The real round trip, kept in one place so every other path is credential-free."""
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

        api = KaggleApi()
        api.authenticate()
        listed = api.competition_submissions(COMPETITION, page_size=100) or []
        return [(str(sub.ref), str(sub.public_score or "")) for sub in listed if sub is not None]


def main() -> None:
    """Refuse a submission record that is missing an entry, double-counting one, or misquoting one."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--record", type=Path, default=RECORD)
    command.parser.add_argument("--ledger", type=Path, default=LEDGER)
    command.parser.add_argument(
        "--live",
        action="store_true",
        help="also ask Kaggle for the submission list (needs credentials, spends NO slot) - the only "
        "way to see an entry that was fired and never written down",
    )
    arguments = command.parsed()

    census = BoardCensus(arguments.record, arguments.ledger)
    live = LiveBoard.taken().scores if arguments.live else None
    refusals = census.refusals(live)
    logger.info(
        "%d recorded submissions, %s",
        len(census.rows()),
        f"{len(live)} live" if live is not None else "live listing not asked for (--live)",
    )
    for refusal in refusals:
        logger.error("%s", refusal)
    if refusals:
        sys.exit(1)


if __name__ == "__main__":
    main()
