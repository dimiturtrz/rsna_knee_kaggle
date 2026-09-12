"""Is the submission record COMPLETE? `python -m kneemri.kaggle.board_check --live`.

`comp/board.jsonl` is the only machine-readable account of what this project has actually scored, and
`board_transfer` fits the offline-to-board compression from it. That makes its COMPLETENESS a property of
every number derived from it: a missing row is not a gap in a document, it is a point silently dropped
from a fit that decides which entry gets a slot. On 2026-08-25 the file held 34 of the competition's 58
submissions. Fifteen scored all-ours entries were absent, including `basedense_v1` — the entry pinned in
`comp/champion.json` — and `votefit_v1`, the best score this project owns. The compression constant read
+1.82 against +1.72 once they were entered, and that number is quoted inside shipped kernels.

IT IS NOT A CONSTANT, AND THE WHOLE-RECORD MEDIAN IS THE WRONG NUMBER TO PRICE AN ARM WITH.
`board_transfer` reports the compression as a CURVE that saturates: over the whole record the median is
+1.72 board points per holdout point, but restricted to the regime we now ship in (offline at or above
0.8661) it is +0.74, and above 0.8679 no pair resolves at all. Every kernel docstring quoting a single
figure as "the transfer" is quoting the slope of a region it has already left. Read the in-regime line
of `board_transfer`'s own output, not a number copied out of prose — a stale slope has now been quoted
wrong three times, which is the same failure this check was written to stop one level down.

THE JOIN IS THE SUBMISSION REF, not a name. Kernel names repeat across entries (`sixfresh` shipped
twice), prose spells them freely ("sixfresh v2", "rate member displaces sliceattn"), and a heading score
is not unique — seven entries read 0.937. The ref is Kaggle's own identifier, it is unique, and asking
Kaggle for the list costs no submission slot.

EVERY submission gets a row, the ones this file will not price included. An entry excluded for a good
reason (a mixed fusion whose members were fitted elsewhere, a 0.500 kit failure, a run that returned no
score) carries `scope: excluded` and the reason, so the exclusion is a recorded decision rather than an
absence. Only then can the check be an exact set comparison instead of a heuristic about which absences
were meant.

AN OFFLINE NUMBER MUST ALSO SAY HOW ITS MEMBERS WERE CHOSEN. A set picked by SEARCH over a pool and
then scored on the studies the search ran on reports its own overfitting as a result, which is the
mistake `MemberCurve.honest` exists to measure — +0.0025 to +0.0031 at the pool sizes this project
searches. `knee-jsgb` closed on the rule that such a row enters this file BIAS-CORRECTED, and the
`peak8` row was written RAW the following day. The 0.004 of phantom shortfall that produced drove a
write-up titled "the offline ruler fails", which was retracted (knee-wz81) after `board_transfer`
reported 0 of 269 resolved pairs inverted. No gate stood between the rule and the row; `undeclared`
and `uncorrected` below are that gate.
"""

import datetime as dt
import json
import logging
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from core import cli
from kneemri.evaluation.board_transfer import EXCLUDED, SCOPES
from kneemri.evaluation.selection_bias import SELECTION_BIAS, SelectionBias
from kneemri.kaggle.kernel_script import KernelScript

logger = logging.getLogger(__name__)

RECORD = Path("comp/board.jsonl")
LEDGER = Path("comp/SUBMISSIONS.md")
KERNELS = Path("comp/kernels")
COMPETITION = "rsna-knee-abnormality-detection"
PAGE = 100
"""How many submissions the live listing reaches, and it is a CEILING on the API rather than a page size.

Measured on 2026-09-05, after a paging loop written the day before produced no extra rows: pages 1, 2 and
3 each return 100 submissions and the three sets are IDENTICAL. `page_number` is accepted and ignored, so
there is no second page to walk - this competition's listing is the 100 most recent submissions and
nothing older. Every reach beyond it costs an API call, returns the same rows, and moves the account
toward a 429."""
KERNEL_PREFIX = "kneemri-"
VERSION_MARK = "_v"
"""How an entry id carries its revision (`chosen_v2`), and the prefix every kernel directory takes. An
entry that records no `kernel` still names its own script through these two: that join is what found a
surviving script for thirteen of the fourteen rows this file once called unrecoverable."""

SCORE_TOLERANCE = 5e-4
"""A board score is quoted to three decimals; anything wider than half a tick is a transcription error."""

ALL_DATA_SUFFIX = "_ff"
ALL_DATA_NAMED = frozenset({"final_ws_e12"})
"""How an all-data (`--fold -1`) member is spelled in the roster: the `_ff` suffix, plus the one member
that predates the convention. A pool holding one has seen every holdout study, so an offline number
beside it is not a held-out number — that is the entity error that produced a bad anchor on 2026-08-24.
This is a NAME rule on purpose: the disk rule (does a holdout record exist) needs the data root, and a
record check must run in CI where there is none."""

SEARCH = "search"
DESIGNED = "designed"
SELECTIONS = frozenset({SEARCH, DESIGNED})
"""How a row says its seats were chosen. `search` is any argmax over a pool — `member_curve`'s greedy,
`record_mix --greedy`, `seat_value`'s best-seat scan; `designed` is a set named by an argument, which
includes a one-seat swap off an existing entry because the candidate was not selected against a field.

A DECLARED FIELD rather than an inference from the score or a phrase in the prose. Nothing about a
number says which procedure produced it, and `offline_scope` is written to be read by people — the two
rows that already discuss selection bias spell it two different ways in the same paragraph. A gate that
grepped prose would pass whichever wording it happened to be taught."""

SELECTION_RULE = (
    "knee-jsgb: an entry whose members were chosen by SEARCH enters this file with a BIAS-CORRECTED "
    "offline and a raw one beside it, never a raw score alone"
)

SELECTION_RULE_DATE = "2026-08-30"
"""When that rule was written down, and therefore the first date this file enforces it on.

DATED, not retroactive, and the date is the whole argument. The correction is measured over pools of
77-119 members; the record's early search-selected rows were searched over ten to twenty-two, and
subtracting a large-pool constant from them moves `board_transfer`'s retro-prediction DOWN rather than
up (knee-wz81, Spearman 0.961 -> 0.947). So the rows before this date are not silently defaulted to
`designed` and are not corrected either — they are outside a rule that did not exist and whose constant
does not describe them. Re-opening one means measuring the bias at ITS pool size, not applying this
band. ISO dates compare as strings, which is why the record writes them that way."""


@dataclass(frozen=True)
class BoardRow:
    """One recorded submission — what the board said, and what this file is willing to say about it."""

    ref: str
    identifier: str
    scope: str
    board: float | None
    offline: float | None
    members_named: tuple[str, ...]
    names_key: bool
    """Whether the line carried `members_named` AT ALL, which an empty tuple cannot say on its own.

    ABSENT AND EMPTY ARE DIFFERENT CLAIMS. Absent says this entry's seats are unknown; empty says they
    were looked up and there are none, which is false of every ensemble ever submitted. `weakseven_v1`
    carried `[]` through two recovery passes with a kernel and a literal tuple sitting in the tree,
    because a consumer testing membership read it as recovered and one testing truthiness read it as
    missing, and neither was wrong about the JSON.
    """
    kernel: str | None
    members: int | None
    date: str
    selection: str | None
    """How this row says its seats were chosen, or None when it says nothing — see `SELECTIONS`."""
    offline_raw: float | None
    """The search's own in-sample score, when `offline` is that score corrected.

    A SECOND NUMBER rather than a sentence, because the correction has to be checkable arithmetic. Both
    rows that carried a correction before this field existed stated the raw value in their prose, and a
    reader could confirm the subtraction by hand; nothing else could, which is the entire defect.
    """

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
            names_key=named is not None,
            kernel=None if row.get("kernel") is None else str(row["kernel"]),
            members=None if row.get("members") is None else int(str(row["members"])),
            date=str(row.get("date", "")),
            selection=None if row.get("selection") is None else str(row["selection"]),
            offline_raw=None if row.get("offline_raw") is None else float(str(row["offline_raw"])),
        )

    def governed(self, since: str) -> bool:
        """Whether the selection rule reaches this row: it prices something, and it postdates the rule.

        Both halves are the rule's own scope. A row carrying no offline number makes no claim the
        correction could apply to, and a row written before `SELECTION_RULE_DATE` was searched over a
        pool the measured band does not describe — see that constant for why neither is a pass.
        """
        return self.offline is not None and self.date >= since

    def script(self, kernels: Path) -> Path:
        """The kernel script this entry shipped from, its own field first and its id stem as fallback.

        The fallback is the whole point: the earlier recovery offered a row a script ONLY through
        `kernel`, so every row carrying a null there was never handed one and was written off as having
        no surviving source. `chosen_v2` -> `kneemri-chosen` is a two-line join, and it reaches thirteen
        of the fourteen.
        """
        name = self.kernel or KERNEL_PREFIX + self.identifier.rsplit(VERSION_MARK, 1)[0]
        return kernels / name / f"{name}.py"

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

    def __init__(
        self,
        record: Path = RECORD,
        ledger: Path = LEDGER,
        kernels: Path = KERNELS,
        bias: SelectionBias | None = None,
        since: str = SELECTION_RULE_DATE,
    ) -> None:
        self.record = record
        self.ledger = ledger
        self.kernels = kernels
        self.bias = bias or SelectionBias()
        self.since = since

    def rows(self) -> list[BoardRow]:
        """Every recorded entry, the commentary lines skipped — they carry a note and no submission.

        THE TEST IS THE MISSING REF, NEVER THE PRESENT NOTE. This read `"note" not in row` until
        2026-09-12, which made ANNOTATING a scored row delete it from every consumer of this census:
        `submit_check` stopped seeing it as a repeat, `duplicated` could not catch a second copy of it,
        and the transfer fit dropped its point. Four priced rows were invisible that way, two of them at
        the board's own maximum — and the symptom was a GREEN check, which is the shape that costs a slot.
        A note is how a row explains itself; it cannot also be how a row disappears.
        """
        lines = [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines() if line]
        return [BoardRow.parsed(row) for row in lines if row.get("ref") or "note" not in row]

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

    def undeclared(self) -> list[str]:
        """Priced rows the rule reaches that never say how their seats were chosen.

        Silence is the finding, not a default. `unpriceable` above can only fire on rows that DID name
        their members for exactly this reason: what a check cannot be told, it must refuse rather than
        assume. Defaulting a silent row to `designed` would pass every raw search-selected row ever
        written, which is precisely the row this gate exists for.
        """
        return [
            f"{row.identifier} records offline {row.offline} and declares no `selection`, so nothing "
            f"says whether its seats were searched; write one of {sorted(SELECTIONS)} — {SELECTION_RULE}"
            for row in self.rows()
            if row.governed(self.since) and row.selection not in SELECTIONS
        ]

    def misscoped(self) -> list[str]:
        """Rows whose `scope` is in neither reader's vocabulary, and are therefore green in both.

        THE ROW THIS FILE WAS MISSING, found 2026-09-04. `scope` names the DENOMINATOR an `offline`
        number was computed on, and only `board_transfer` acts on that meaning: it fits
        `row["scope"] == "ensemble"` and reports everything else as a per-scope drop. THIS file reads
        the same field for one narrower question — is the row priced at all — and tests only
        `!= EXCLUDED`. So a value in NEITHER vocabulary passes here and silently vanishes there.

        Four rows carried `scope: "included"`, which is the negation of `EXCLUDED` written into a field
        that does not mean that. Three of them carried honest 909-study ensemble offline numbers, and
        sat outside the fit every arm in this project is priced against — while both consumers reported
        clean. `board_transfer` did count the drop; a drop that is COUNTED still reads as intended
        unless something asserts the vocabulary is closed.
        """
        return [
            f"{row.identifier} carries scope={row.scope!r}, which no reader of this field knows; "
            f"board_transfer will drop it from the fit while this check stays green — "
            f"write one of {sorted(SCOPES)}"
            for row in self.rows()
            if row.scope not in SCOPES
        ]

    def uncorrected(self) -> list[str]:
        """Search-selected rows whose offline number is the search's own in-sample score.

        THE ROW THIS FILE WAS MISSING. `peak8` shipped the day after the rule was written and entered
        the raw 0.8731; through the fold-0 anchors that predicts 0.936 against an actual 0.932, and the
        0.004 of phantom shortfall was written up as the offline ruler failing. Corrected to 0.8704 it
        predicts 0.934 and the ruler is intact — `board_transfer` orders 0 of 269 resolved pairs the
        wrong way with the row present either way.

        The correction is checked as ARITHMETIC against the two numbers the row carries, so a row that
        merely SAYS it was corrected cannot pass; and it is checked against the measured band rather
        than a point, so an over-correction that buys a better transfer residual fails as well.
        """
        found = []
        for row in self.rows():
            if row.selection != SEARCH or row.offline is None:
                continue
            if row.offline_raw is None:
                found.append(
                    f"{row.identifier} declares selection={SEARCH} and records offline {row.offline} with no "
                    f"`offline_raw`, so no arithmetic can say the search bias was ever subtracted; carry the "
                    f"greedy's own in-sample score beside it — {SELECTION_RULE}"
                )
            elif not self.bias.corrects(row.offline_raw, row.offline):
                found.append(
                    f"{row.identifier} was chosen by SEARCH and records offline {row.offline} against a raw "
                    f"{row.offline_raw}, a correction of {row.offline_raw - row.offline:+.4f}; the measured "
                    f"selection bias is {SELECTION_BIAS[0]:+.4f} to {SELECTION_BIAS[1]:+.4f}, so this row "
                    f"should carry {self.bias.stated(row.offline_raw)} — {SELECTION_RULE}"
                )
        return found

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

    def hollow(self) -> list[str]:
        """Rows carrying an empty `members_named` — the claim no entry can truthfully make."""
        return [
            f"{row.identifier} carries an EMPTY members_named; omit the key when the seats are unknown, "
            f"because empty reads as 'this entry had none' and no ensemble ever had none"
            for row in self.rows()
            if row.names_key and not row.members_named
        ]

    def miscounted(self) -> list[str]:
        """Rows naming a different number of seats than they record — the rule the hand recovery used.

        `members` is written from the submission log and `members_named` from the script, so the two are
        independently sourced and their disagreement is exactly the signal: a partial recovery that
        stopped at four of six looks complete until these are compared.
        """
        return [
            f"{row.identifier} names {len(row.members_named)} members and records members={row.members}"
            for row in self.rows()
            if row.members_named and row.members is not None and len(row.members_named) != row.members
        ]

    def recoverable(self) -> list[str]:
        """Unnamed rows whose surviving script DOES name its members literally — recovery left on the table.

        Silence about a row's seats is only honest when nothing can break it. A row selecting by rank
        (`KernelRun(members=N)` is `strongest(N)` over recorded holdouts) cannot be replayed against a
        package that has since changed, and `member_names` returns None for it, so those stay silent and
        this stays quiet about them. A row whose script holds a literal tuple has no such excuse.

        Read the tuple from the commit that SHIPPED the entry, not from HEAD: `kneemri-chosen` was edited
        after `chosen_v2` submitted. The tuple was unchanged, which is a thing you check and not a thing
        you assume.
        """
        found = []
        for row in self.rows():
            if row.members_named or row.scope == EXCLUDED:
                continue
            script = row.script(self.kernels)
            names = KernelScript(script).member_names() if script.is_file() else None
            if names:
                found.append(f"{row.identifier} names no members and {script} names {len(names)} of them")
        return found

    def unrecorded(self, live: Mapping[str, float | None]) -> list[str]:
        """Scored submissions this file has never heard of — the failure that was live.

        A submission Kaggle is still scoring carries no board number, so it CANNOT be recorded yet and is
        not a defect: it belongs to `pending`. Refusing on it would leave this gate red for the forty
        minutes after every submission, which is how a gate teaches its reader to ignore it.
        """
        recorded = {row.ref for row in self.rows()}
        return sorted(ref for ref, score in live.items() if ref not in recorded and score is not None)

    def pending(self, live: Mapping[str, float | None]) -> list[str]:
        """Submissions still scoring and not yet written down — a reminder to come back, never a refusal."""
        recorded = {row.ref for row in self.rows()}
        return sorted(ref for ref, score in live.items() if ref not in recorded and score is None)

    def phantom(self, live: Mapping[str, float | None]) -> list[str]:
        """Recorded refs Kaggle does not report — a typo in a ref, or a row for a submission never made.

        ONLY WITHIN THE LISTING'S REACH. Kaggle serves the 100 most recent submissions and ignores
        `page_number`, so once this competition passed 100 entries every older row became invisible to
        the live call — permanently, and with no signal of its own. Reported as phantoms they were seven
        standing alarms against seven submissions that really scored, and an alarm that can never be
        cleared is one a reader learns to scroll past.

        Refs are assigned in increasing order, which is what makes the horizon computable: a recorded ref
        BELOW the oldest live one is out of reach, not missing. A ref above it and absent is still the
        defect this method exists for.
        """
        reach = min((int(ref) for ref in live), default=0)
        return sorted(row.ref for row in self.rows() if row.ref and row.ref not in live and int(row.ref) >= reach)

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
        found += self.hollow()
        found += self.miscounted()
        found += self.recoverable()
        found += self.unpriceable()
        found += self.undeclared()
        found += self.misscoped()
        found += self.uncorrected()
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
    dates: tuple[dt.datetime, ...] = ()
    """When each submission was FIRED, which is the only honest count of a day's spent slots — a
    pending entry has no score and no record row, and `slot_pace` reads this rather than the record."""

    @classmethod
    def taken(cls, lister: Callable[[], list[tuple[str, str, str]]] | None = None) -> "LiveBoard":
        """Ask Kaggle, or ask whatever was injected in its place."""
        listed = lister() if lister is not None else cls._kaggle()
        return cls(
            {ref: float(score) if score else None for ref, score, _ in listed},
            tuple(cls._stamped(when) for _, _, when in listed if when),
        )

    @staticmethod
    def _stamped(when: str) -> dt.datetime:
        """Kaggle's submission date as an aware UTC instant. It reports naive UTC, so it is labelled."""
        parsed = dt.datetime.fromisoformat(when)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)

    @staticmethod
    def _kaggle() -> list[tuple[str, str, str]]:
        """The real round trip, kept in one place so every other path is credential-free.

        ONE CALL, because there is no second page to ask for. This was a paging loop for a day, written
        when the 107th submission pushed seven real rows off the end of a 100-row listing and the check
        reported them as seven records the board does not hold. The loop did not recover them: pages 1-3
        return the same 100 refs, so it made twenty identical calls and produced one page of answers. The
        listing's horizon is a property of the API, and the fix belongs in `phantom`, which now declines
        to call a record missing when it sits BEHIND that horizon.
        """
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

        api = KaggleApi()
        api.authenticate()
        batch = api.competition_submissions(COMPETITION, page_size=PAGE) or []
        return [(str(sub.ref), str(sub.public_score or ""), str(sub.date or "")) for sub in batch if sub is not None]


def main() -> None:
    """Refuse a submission record that is missing an entry, double-counting one, or misquoting one."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--record", type=Path, default=RECORD)
    command.parser.add_argument("--ledger", type=Path, default=LEDGER)
    command.parser.add_argument("--kernels", type=Path, default=KERNELS)
    command.parser.add_argument(
        "--live",
        action="store_true",
        help="also ask Kaggle for the submission list (needs credentials, spends NO slot) - the only "
        "way to see an entry that was fired and never written down",
    )
    arguments = command.parsed()

    census = BoardCensus(arguments.record, arguments.ledger, arguments.kernels)
    live = LiveBoard.taken().scores if arguments.live else None
    refusals = census.refusals(live)
    logger.info(
        "%d recorded submissions, %s",
        len(census.rows()),
        f"{len(live)} live" if live is not None else "live listing not asked for (--live)",
    )
    for ref in census.pending(live) if live is not None else []:
        logger.warning("submission %s is still scoring - record it when it lands", ref)
    for refusal in refusals:
        logger.error("%s", refusal)
    if refusals:
        sys.exit(1)


if __name__ == "__main__":
    main()
