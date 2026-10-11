"""Every board score and rank the prose quotes, against the board record and the live standing:
`python -m kneemri.experiments.board_claims`.

THE AXIS `claims_check` CANNOT SEE. That check asks whether a number quoted beside a RUN is one the run
recorded — a member's holdout, read off disk. It says nothing about the other half of this project's
numbers, which are claims about ENTRIES and about where we stand: `dark6 reads 0.937`, `0.919 is rank 84
of 1758`, `the 0.925 champion`. Those are checkable too, against `comp/board.jsonl` and the dated reads
in `comp/leaderboard.jsonl`, and they rot faster than member scores because the board moves without us.

THE FAILURE IT IS NAMED FOR, all four already paid for:

  - `comp/board.jsonl` held 34 of the competition's 58 submissions until 2026-08-25. Every aggregate
    fitted before that date — the member-count curve, the transfer constant — was fitted on a subset and
    read as a fact about the record (knee-macc, knee-8pp8).
  - `board_transfer`'s slope was quoted from prose at +1.82 against a complete record's +1.55.
  - knee-17am still says "0.919 is rank 84 of 1758", written 2026-08-17. The live read is 0.944 at rank
    30 of 2814 — the quoted gap to the leader is off by a factor of four.
  - knee-u521 calls a 0.925 entry "the champion" through two later champions, which is the same failure
    `CLAUDE.md` carries a standing rule about.

WHY THREE DECIMALS IS THE WHOLE DISCRIMINATOR. Kaggle publishes a board score to three decimals and this
project records member holdouts to four. So `0.944` is a claim about an entry and `0.8578` is a claim
about a member, and the two axes separate on the number's own spelling with no name resolution at all.
A two-decimal spelling (`0.93`) is admitted as the same claim written short.

WHAT IT DOES NOT ACCUSE, each class measured on the corpus rather than guessed:

  - A PRE-REGISTERED BAND. Every submission message states its branches — `>=0.946 favourable`,
    `0.9425-0.9455 FLAT`. Those are bars nothing scored, so a sign, a comparator or a range hyphen
    disqualifies the number, exactly as `claims_check.BOUNDED` does for member scores.
  - A LINE NAMING SEVERAL ENTRIES. An interpretation's job is putting an entry beside its control and
    both numbers are correct.
  - A COLUMNAR LINE. Reused wholesale from `claims_check`: a table row's numbers are named by a header
    this line-scanner never sees.

RANK IS CHECKED AGAINST THE LATEST READ AND NEVER AGAINST A REMEMBERED ONE, which is the rule
`comp/leaderboard.json` states in its own `why`. A rank that disagrees with the newest read is reported
with the instant that read was taken, because the finding is usually "this sentence is old", not "this
sentence was wrong" — and a dated sentence that says so itself is not a defect.

ENFORCED as of 2026-09-01, with `--assert` in `nox -s lint` and in CI (its inputs are all in the repo,
so a runner with no data root can run it). It shipped advisory and sat at 14 standing findings until
somebody swept them — canonical entry ids where prose wrote short forms, an inline read-date on every
historical rank, and one board score genuinely misquoted. At zero it blocks, because a list nobody
sweeps is the thing this check exists to prevent.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from core import cli
from core.denominator import Denominator
from kneemri.experiments.claims_check import CORPUS, ClaimsCheck

logger = logging.getLogger(__name__)

BOARD = Path("comp/board.jsonl")
LEADERBOARD = Path("comp/leaderboard.jsonl")
BEADS = Path(".beads/issues.jsonl")
"""The issue tracker is prose too, and it is where the stale verdicts live: 85 of the open issues carry a
verdict in their title and a fifth of those were last touched before the board record was completed."""

BEAD_FIELDS = ("title", "description", "notes", "design")

BOARD_FLOOR = 0.80
BOARD_CEILING = 0.96
"""The band a three-decimal number has to sit in to be read as a board claim. Our worst scored entry is
0.868 and the leader is 0.952, so the floor clears the deltas and shares that share the corpus and the
ceiling admits the frontier with room over it."""

SCORE = re.compile(r"(?<![><~±+\-=])(?<![><~±+\-=] )(?<![\d.])0\.\d{2,3}(?!\d)")
"""A board score as Kaggle publishes it, minus the ones a comparator or a sign has declared to be a bar
rather than a measurement. `=` is in the class for `>=0.939`, which is a gate on an entry that had not
scored yet — three of the first sweep's fifteen findings. Two decimals admitted because the record
writes 0.93 for 0.930."""

WINDOW = 40
"""How near the entry's name a number has to sit to be a claim ABOUT that entry, in characters.

THE FALSE POSITIVE THIS EXISTS FOR, and it was the majority class. Entry names are dense in this corpus
and so are board numbers that belong to something else: the leader (`calibsolo ... puts 0.950 out of
reach`), a target (`Board 0.946 therefore asks ... (honest12, 0.8751)`), the pinned champion (`below the
pinned 0.943` in a note about honest12), a threshold (`Every entry at 0.936 and above ... use --against
gradeswap_v1`). Every one of those is a correct sentence and none of them quotes the named entry at all.
A claim about an entry's score is written next to the entry — `dark6 reads 0.937`, `basedense_v1 0.925`
— so proximity is what separates the two, and forty characters is the widest gap any true positive in
the corpus spans."""

SENTENCE_END = re.compile(r"[.!?;]\s")
RANGE = re.compile(r"0\.\d{3,4}\s*[-–]\s*0\.\d{3,4}")
"""A pre-registered FLAT band, written as a range. Both ends are bars; neither is anyone's score."""

RANK = re.compile(r"rank (\d+) of ([\d,]+)", re.IGNORECASE)
"""The standing as this project writes it, both halves, so a rank is never checked without its field."""

DATED = re.compile(r"\d{4}-\d{2}-\d{2}")

VERSION = re.compile(r"_v\d+$")
"""An entry's version suffix. Prose names the comparator by its bare stem (`s6third` beside `s6half_v1`), and
a stem nobody recognises turns a two-entry comparison into a one-entry misquote."""
"""A line that carries its own read date is a DATED READ, and a dated read is never stale.

This is the discriminator, not a refinement of one. The first sweep raised 18 rank findings and 13 of
them were sentences like `rank 50 of 2404, read live at 2026-08-25T23:15:59Z` — correct when written,
correct now, and a document that states when it looked is doing exactly what this project asks of it.
Accusing those is how a gate ends up with nobody reading it. What is left is the class that has actually
cost something: a bead TITLE asserting `0.919 is rank 84 of 1758` as a standing fact with no date on it.

A DOCUMENT whose NAME carries a date is exempt whole. `interpretations/` and `research/` are named
`<date>_<topic>.md` by this project's own doc-layer convention, so every sentence in one is a dated read
whether or not the sentence repeats the date. Beads carry no date in their id, which is exactly why the
stale verdicts live there.
"""

DECIMALS = 3

OTHER_RULER = ("member", "holdout", "offline", "gold", "auc", "predicts", "per-label")
"""Words that say the number beside them is measured on a different ruler than the board's.

The board scores studies Kaggle hides; a member score is a holdout AUC; a predicted board score is the
transfer constant's output, not a submission's result. This project has already paid 0.028 for reading
one ruler's number against another's, and a line that names its ruler is doing the right thing — so
`putting solo_v4's member near 0.877` and `predicts 0.934 against an actual 0.932` are both correct
sentences that this check must not accuse."""


@dataclass(frozen=True)
class BoardDocument:
    """One document of the record, reduced to the lines a claim can be read off.

    Beside its reader rather than in a types module: the only thing that produces one is
    `BoardClaims.documents`, and the only thing that consumes one is the sweep below it.
    """

    label: str
    lines: tuple[tuple[int, str], ...]

    @classmethod
    def of(cls, path: Path) -> "BoardDocument":
        """A markdown, python or kernel file, read the way `claims_check` reads one."""
        return cls(label=str(path), lines=tuple(ClaimsCheck.readable(path)))

    @classmethod
    def beads(cls, path: Path) -> list["BoardDocument"]:
        """One document per issue, so a finding cites `knee-abcd` and the field it is in.

        An issue's prose is not a file, so the number is the position of the sentence within the issue's
        own fields read in `BEAD_FIELDS` order — a reader opens the issue, not a line. Columnar rows are
        dropped for the reason `claims_check.readable` states: an issue's notes carry recomputed tables
        whose numbers are named by a header this scanner never sees.
        """
        if not path.exists():
            return []
        found: list[BoardDocument] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("id"):
                continue
            prose = "\n".join(str(record.get(field) or "") for field in BEAD_FIELDS).splitlines()
            text = [
                (number, text)
                for number, text in enumerate(prose, start=1)
                if text.strip() and not ClaimsCheck.columnar(text)
            ]
            found.append(cls(label=str(record["id"]), lines=tuple(text)))
        return found


class BoardClaims:
    """Board scores and ranks quoted in the record, against the record of the board.

    Truth is injected so the check is testable without a competition file, and so the one place that
    knows how to read the board stays the board's own files.
    """

    def __init__(self, entries: dict[str, float], standing: dict[str, object], corpus: list[BoardDocument]) -> None:
        self.entries = entries
        self.standing = standing
        self.corpus = corpus
        self.stems = {name: BoardClaims.token(name) for name in entries}
        self.bare = {name: BoardClaims.token(VERSION.sub("", name)) for name in entries}

    @staticmethod
    def token(text: str) -> re.Pattern[str]:
        """`text` as a whole word: no letter, digit or underscore on either side."""
        return re.compile(rf"(?<![A-Za-z0-9_])({re.escape(text)})(?![A-Za-z0-9_])")

    @staticmethod
    def scored(path: Path) -> dict[str, float]:
        """Every entry the board record knows, by id, with the score it actually got.

        Rows marked `scope: excluded` are kept: an excluded row still has a board score, and a sentence
        quoting it wrong is wrong in the same way.
        """
        if not path.exists():
            return {}
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {str(row["id"]): float(row["board"]) for row in rows if row.get("id") and row.get("board")}

    @staticmethod
    def latest(path: Path) -> dict[str, object]:
        """The newest leaderboard read, or an empty standing when none has been taken."""
        if not path.exists():
            return {}
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return max(rows, key=lambda row: str(row.get("taken", ""))) if rows else {}

    @staticmethod
    def documents(roots: tuple[Path, ...], beads: Path) -> list[BoardDocument]:
        """The prose corpus: the same roots `claims_check` reads, plus every issue."""
        found = [BoardDocument.of(path) for path in ClaimsCheck.documents(roots)]
        return found + BoardDocument.beads(beads)

    @staticmethod
    def quoted(line: str) -> list[tuple[int, int, str]]:
        """Where on a line each board-score number sits, and what it says."""
        if RANGE.search(line):
            return []
        found = [(match.start(), match.end(), match.group()) for match in SCORE.finditer(line)]
        return [span for span in found if BOARD_FLOOR <= float(span[2]) <= BOARD_CEILING]

    def named(self, line: str) -> str | None:
        """The single board entry a line names, or None when it names none or several.

        A bare stem never names an entry (too many are words: `band`, `rate`, `vote`), but it does make a
        line a comparison, because prose writes the comparator that way (`s6half_v1 0.955 (s6third 0.956)`).
        """
        found = [name for name, token in self.stems.items() if token.search(line)]
        if len(found) != 1:
            return None
        compared = [name for name, token in self.bare.items() if name != found[0] and token.search(line)]
        return None if compared else found[0]

    def about(self, line: str, name: str) -> set[str]:
        """The board scores written near enough to `name` on this line to be claims about that entry."""
        names = [(match.start(), match.end()) for match in self.stems[name].finditer(line)]
        near = [
            (start, text)
            for start, end, text in self.quoted(line)
            if any(
                first - WINDOW <= start
                and end <= last + WINDOW
                and not SENTENCE_END.search(line[min(end, last) : max(start, first)])
                for first, last in names
            )
        ]
        lowered = line.lower()
        return {
            text
            for start, text in near
            if not any(word in lowered[max(0, start - WINDOW) : start] for word in OTHER_RULER)
        }

    def misquoted(self) -> list[tuple[str, int, str, set[str], float]]:
        """Every line quoting one entry, next to its name, at board scores that entry did not get."""
        findings: list[tuple[str, int, str, set[str], float]] = []
        for document in self.corpus:
            for number, line in document.lines:
                name = self.named(line) if self.quoted(line) else None
                if name is None:
                    continue
                board = self.entries[name]
                scores = self.about(line, name)
                if not scores or any(round(float(text), DECIMALS) == round(board, DECIMALS) for text in scores):
                    continue
                findings.append((document.label, number, name, scores, board))
        return findings

    def stale_ranks(self) -> list[tuple[str, int, int, int]]:
        """Every `rank N of M` in the record that disagrees with the newest read of the board."""
        standing = self.standing.get("rank")
        if standing is None:
            return []
        findings: list[tuple[str, int, int, int]] = []
        for document in self.corpus:
            for number, line in document.lines:
                if DATED.search(line) or DATED.search(document.label):
                    continue
                for rank, field in RANK.findall(line):
                    if int(rank) != int(str(standing)):
                        findings.append((document.label, number, int(rank), int(field.replace(",", ""))))
        return findings


def main() -> None:
    """Report prose that quotes an entry at a board score it did not get, or a rank that has moved."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--assert", dest="blocking", action="store_true")
    arguments = command.parsed()

    entries = BoardClaims.scored(BOARD)
    Denominator(len(entries), f"no scored entry in {BOARD}").refuse_empty()
    check = BoardClaims(entries, BoardClaims.latest(LEADERBOARD), BoardClaims.documents(CORPUS, BEADS))

    misquoted = check.misquoted()
    for label, number, name, scores, board in misquoted:
        logger.info(
            "MISQUOTED %s:%d names %s and quotes %s; that entry scored %.3f",
            label,
            number,
            name,
            ", ".join(sorted(scores)),
            board,
        )
    stale = check.stale_ranks()
    for label, number, rank, field in stale:
        logger.info(
            "STALE RANK %s:%d says rank %d of %d; the read taken %s says rank %s of %s",
            label,
            number,
            rank,
            field,
            check.standing.get("taken"),
            check.standing.get("rank"),
            check.standing.get("field"),
        )
    logger.info(
        "%d scored entries, %d documents read, %d misquoted line(s), %d stale rank(s)%s",
        len(entries),
        len(check.corpus),
        len(misquoted),
        len(stale),
        "" if arguments.blocking else " (advisory)",
    )
    if (misquoted or stale) and arguments.blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
