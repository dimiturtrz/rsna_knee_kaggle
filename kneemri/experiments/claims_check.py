"""Every score the prose attributes to a run, against what that run actually recorded:
`python -m kneemri.experiments.claims_check`.

THE DIRECTION NOTHING HERE CHECKED. `ledger_check` walks runs on disk and asks whether each is written
down somewhere findable — disk to prose. Its own docstring names what that cannot see: "whether the
surrounding sentence is true." This walks the other way. For every line of the record that names ONE run
and quotes a number in the AUC band, it asks whether that number is one the run ever recorded. A run
missing from the ledger is an arm thrown away; a run quoted at a number it never scored is worse, because
it reads as evidence and steers the next decision.

THE FAILURE IT IS NAMED FOR HAS HAPPENED AT LEAST FIVE TIMES, always the same way — a number is copied
forward from prose instead of re-read from the thing that produced it:

  - `CLAUDE.md` quoted the champion at 0.925 through two later champions, which is why the rule beside
    that line now reads "Read the entry and its score from that file, never from here".
  - `board_transfer`'s slope was quoted wrong from prose twice (+1.82 fitted over a 34-row board file
    against +1.55 over the complete 58), and the pinned gap read 0.008 while the API said 0.015.
  - knee-8pp8: the line `board = 1.37 x offline - 0.2598` is FABRICATED — no fit produces it, and the
    real slope at our operating point is ~0.95.
  - DINOv2-small sat in the monoculture table at 0.8964 after `recal_holdannot_s2026` re-scored to
    0.8515 under a ruler that had moved with the arm. Both numbers are real; only one is that trunk's.

That last case is why a disagreement here is a QUESTION and not a verdict. A number the record quotes
and the disk does not know is one of: a stale value the run has since been re-scored past, a two-ruler
artefact (the arm moved the evaluation labels along with itself), a hand-computed mean or delta the band
happens to catch, a run whose artefacts were overwritten by a re-run under the same name, or a number
that was never measured at all. Only the last is fabrication. The check reports; the reader adjudicates.

WHY IT CANNOT SIMPLY FLAG EVERY UNKNOWN NUMBER, and this is the lesson `ledger_check` paid for: a gate
that fires on a third of a correct-enough corpus is a gate the reader learns to skip. Three conditions
narrow it to lines that can actually be wrong about ONE run:

  - the line names EXACTLY ONE known run, and by a multi-segment stem. A line comparing two arms quotes
    two scores legitimately, and a one-word stem (`floor`) matches ordinary prose.
  - the number sits in the AUC band this project scores in. Deltas, weights, thresholds and p-values are
    not claims about a run's score and are never counted.
  - the line quotes NONE of the scores that run recorded — not its best, not any epoch it logged, not
    the value stamped in its checkpoint. A line reading "band_dense 0.8578, against small_feat 0.8601"
    carries a number belonging to something else BY DESIGN, and a line that gets the run right is not
    accused over its neighbours.

WHAT IT CANNOT SEE, measured on its own first sweep of the corpus rather than guessed:

  - A TABLE COLUMN. The marker test is per LINE, and a table puts `nearest` in the header row while the
    data rows carry only numbers — so an agreement column or a second-ruler column reads as a row of
    holdout claims. Two of the eighteen findings on the first clean sweep were this.
  - A SECOND RULER NAMED IN PROSE. `calib_span_hold` genuinely reads 0.9152 on the gold 58 and
    `lr5x_d6` genuinely reads 0.8475 when both arms are given the TTA they get at inference. The
    sentence says so; the check cannot, because there is no second ruler on disk to compare against.
  - A FOLD-0 OR FULL-FIT TWIN. The seating tables are written over `_cf` counterparts whose scores are
    not their `_s2026` originals', and only a name distinguishes them.

None of the three is a defect in the record. All three are why this reports and does not block.

ENFORCED as of 2026-09-01, in `nox -s lint` with `--assert`. It shipped advisory — it read 300k words of
prose written before it existed, and its first job was a list somebody works through. That list was
worked through on 2026-09-01: 29 findings, six of them real misquotes corrected against the runs on
disk, six lines rewritten to name the ruler their number sits on, and the rest false because the check
lacked the attribution rule its board sibling already had. At zero it graduates, because an advisory
list nobody sweeps is a defect deferred, not a defect found.

It is enforced in nox and NOT in CI: it needs finished runs under the data root and refuses an empty
denominator rather than printing a clean corpus's sentence, and the runner has no dataset.
"""

import json
import logging
import re
from pathlib import Path

from core import cli
from core.denominator import Denominator
from kneemri.experiments.ledger_check import LedgerCheck
from kneemri.runs.run_metrics import RunMetrics

logger = logging.getLogger(__name__)

CORPUS = (
    Path("experiments/README.md"),
    Path("interpretations"),
    Path("research"),
    Path("docs"),
    Path("comp"),
    Path("CLAUDE.md"),
    Path("AGENTS.md"),
)
"""Everywhere a run's score is quoted in prose. `comp/` is in because the kernel headers argue from
member scores and were among the documents carrying the withdrawn 0.028 member gap; the two agent files
are in because the champion's stale 0.925 lived in one of them for two champions."""

AUC_FLOOR = 0.70
AUC_CEILING = 0.99
"""The band a number has to sit in to be read as a claim about a run's holdout score. The pool's worst
scoring member is 0.7847 and its best solo is 0.8938, so the floor admits every real member with room
underneath and excludes the deltas, weights and shares that share the corpus. Nothing here scores above
0.99; a number that high is a correlation, a fraction or a probability.

THE BAND NEVER APPLIES TO THE RUN'S OWN SCORES. `encoder_adapt` recorded 0.6902 — below the floor, being
the failed SSL init — so the first run of this check read its heading, "0.6902 vs 0.8578 control", saw
only the control's number in band, and accused the line of misquoting the run it was correctly quoting.
A floor that hides a run's real score turns a correct sentence into a finding."""

BOARD = Path("comp/board.jsonl")
"""The board file, read only for its entry ids."""

WINDOW = 40
"""How far from the run's own name a number may sit and still be read as that run's score, in characters.

A claim about a run is written NEXT to it. Measured over the corpus, the dominant remaining false finding
was a number further down a sentence than that — a bar, a subset's total, a control's reading — and forty
characters is the widest span the record's real `name (0.8578)` and `name reads 0.8578` forms need."""

IDENTIFIER = re.compile(r"`([A-Za-z0-9_./-]+)`")
"""A backticked identifier: the record's own mark for "this is the subject". A number standing NEARER to
some other identifier than to the run the line names is that other thing's number, whether or not this
check knows the name — `against \\`native224\\` = 0.8502` names its subject perfectly well."""

QUOTED = re.compile(r"(?<![\d.])0\.\d{4}(?!\d)")
"""A four-decimal score as the record writes them. Bounded on both sides so `0.85` is not read as a
truncated claim and `0.85781` is not read as `0.8578`."""

BOUNDED = re.compile(r"(?<![><~±+\-])(?<![><~±+\-] )(?<![\d.])0\.\d{4}(?!\d)")
"""The same, minus numbers a comparison or a sign has already declared not to be a measurement.
`cnx_own_recipe` was priced by the TEST `peak8 + it > 0.8731`, which is a bar nothing passed rather than
a score anything scored, and a delta written `+0.0053` is not a claim about anyone's holdout. The sign
is excluded whether or not a space follows it (`> 0.8731` is the same test as `>0.8731`) — but a space
ALONE must never exclude, which is the bug the mirror test caught: written `[>< ~±+\\-]`, the class read
every score in ordinary prose as sign-marked and the sweep saw only the ones glued to a bracket."""

NOT_A_MEMBER_SCORE = (
    "loss",
    "greedy",
    "ensemble",
    "members",
    "board",
    "correlation",
    "neighbour",
    "nearest",
    "agreement",
    "twin",
    "closer",
    "spread",
    "distance",
    "distant",
    "champion five",
    "champion seven",
    "control",
    "roster",
    "curve",
    "pair line",
    "tighter",
    "displacing",
    "boundary",
    "tta",
    "bar",
    "gold 58",
)
"""Line markers that put the number's subject somewhere other than one member's holdout. Training loss
lands squarely in the AUC band (`0.7493 -> 0.7113`); a greedy search line reads `greedy 54: 0.8670
+raddino_dense`, where the score is the SUBSET's and the member is what was added to it. Neither is a
claim this check can adjudicate, and both are frequent enough to bury the ones it can.

THE LARGEST CLASS IS PAIRWISE AGREEMENT, and it is the one that most looks like a score: this project
measures how alike two members are on the same 0-1 scale it measures accuracy on, and the seating
arguments are written in it — `ema_dense`'s nearest seated neighbour is 0.9306, `through_dense` sits at
0.9143 from its closest twin. Every one of those is a claim about a PAIR. Read as a holdout score they
are all wrong by a mile, which is exactly how they crowded out the six real findings on the first pass.

THE SECOND HALF OF THIS TUPLE WAS MEASURED on the 2026-09-01 sweep, by reading all 29 findings by hand:
twenty-two of them were a number whose subject the line NAMES, in a word this tuple did not yet carry.
`control`, `roster`, `curve`, `pair line`, `boundary` and `bar` each put the number on something that is
not one member's holdout; `tighter`, `distant` and `displacing` mark a comparison whose number belongs to
the thing compared against; `tta` marks the second ruler this check's own docstring already names as a
blind spot; `champion seven` is `champion five` one champion later. Six real defects survived that sweep
and were corrected in place."""


class ClaimsCheck:
    """Lines of the written record, against the scores the runs they name actually recorded.

    Truth is injected rather than read, so the check is testable without a data root and so the one
    place that knows how to read a run off disk stays `LedgerCheck.finished`.
    """

    def __init__(
        self, runs: dict[str, float], recorded: dict[str, set[str]], documents: list[Path], entries: set[str]
    ) -> None:
        self.runs = runs
        self.recorded = recorded
        """Every score each run ever wrote, as four-decimal strings: its epochs and its kept value. An
        epoch score is a legitimate thing for prose to quote — a trajectory argument does it — so a run
        is not accused over a number it genuinely produced merely because a later epoch beat it."""
        self.corpus = documents
        self.entries = entries
        """Board entry ids. A line naming one is making a claim about an ENTRY's score, and an entry is
        an ensemble of members: `against basedense_v1 (0.8690, five members)` quotes the entry's number
        beside a member's name, correctly, and the check has nothing to say about it."""
        self.stems = {
            name: re.compile(
                rf"(?<![A-Za-z0-9_])(?:{re.escape(name)}|{re.escape(LedgerCheck.trimmed(name))})(?![A-Za-z0-9_])"
            )
            for name in runs
            if "_" in LedgerCheck.trimmed(name)
        }
        """A run's name matched as a WHOLE TOKEN in the raw line, not as a substring of the normalised
        one, and under BOTH spellings the record uses — `calib_dense_hold` and `calib_dense_hold_s2026`.
        Matching only the trimmed one made the suffixed spelling invisible, so a line naming two members
        looked like a line naming one, and the second member's score was reported as the first's.

        Raw rather than normalised because `LedgerCheck.plain` folds `basedense_v1` and `base_dense` to
        strings where one contains the other, so every mention of that board ENTRY was read as a claim
        about the member whose name it spells. The cost of
        the stricter match is a mention the record spells differently from the file, which lands as a
        finding this check simply does not raise."""

    @staticmethod
    def readable(document: Path) -> list[tuple[int, str]]:
        """A document's lines, numbered from one so a finding cites a place a reader can open.

        COLUMNAR LINES ARE NOT READ — a markdown table row, and an indented row of a fixed-width block
        (in prose, and in the comment banners the kernels carry). A row's numbers are named by a HEADER
        this line-scanner never sees, and the header is usually a ruler this check does not own: an
        entry's joint offline, a gold-58 reading, a with-member holdout, a nearest-neighbour distance.
        Measured over the whole corpus, every columnar finding read by hand was a correct number under
        another column's name — 48 of 97 pipe rows on the first widened sweep, and the indented blocks
        again after that. Dropping a class the check is structurally unable to adjudicate is the trade
        the ledger's own design lesson asks for: a gate that fires on a correct corpus is a gate nobody
        reads. It is a BLIND SPOT, not a clean skip — a genuinely stale number inside a table passes
        here, and only a checker that parses the header can ever catch it.
        """
        if not document.exists():
            return []
        numbered = enumerate(document.read_text(encoding="utf-8").splitlines(), start=1)
        return [(number, line) for number, line in numbered if not ClaimsCheck.columnar(line)]

    @staticmethod
    def columnar(line: str) -> bool:
        """Whether a line is a table row or a fixed-width block row rather than a sentence."""
        body = line.lstrip("#") if line.lstrip().startswith("#") else line
        return body.lstrip().startswith("|") or body.startswith(("    ", "\t")) or "   " in body.strip()

    def named(self, line: str) -> str | None:
        """The single run a line names, or None when it names none, several, or a board entry too.

        Several is the common case and the one that must not be accused: an interpretation's whole job
        is putting an arm beside its control, and both scores on that line are correct.
        """
        if any(entry in line for entry in self.entries):
            return None
        found = [name for name, token in self.stems.items() if token.search(line)]
        return found[0] if len(found) == 1 else None

    @staticmethod
    def quoted(line: str) -> set[str]:
        """The four-decimal numbers on a line that read as a claim about somebody's holdout score."""
        if any(marker in line.lower() for marker in NOT_A_MEMBER_SCORE):
            return set()
        return {text for text in BOUNDED.findall(line) if AUC_FLOOR <= float(text) <= AUC_CEILING}

    def about(self, line: str, name: str) -> set[str]:
        """The quoted scores the line actually attributes to the run it names.

        Two ways a number belongs to something else even on a line naming exactly one run: it sits too
        far from the name to be its (`WINDOW`), or some other backticked identifier stands nearer to it
        (`IDENTIFIER`). Both were read off the corpus rather than guessed.
        """
        token = self.stems[name]
        mine = [match.span() for match in token.finditer(line)]
        others = [match.span(1) for match in IDENTIFIER.finditer(line) if not token.search(match.group(1))]
        kept: set[str] = set()
        for match in BOUNDED.finditer(line):
            if not AUC_FLOOR <= float(match.group()) <= AUC_CEILING:
                continue
            nearest = min((self.gap(match.start(), span) for span in mine), default=WINDOW + 1)
            if nearest > WINDOW or any(self.gap(match.start(), span) < nearest for span in others):
                continue
            kept.add(match.group())
        return kept

    @staticmethod
    def gap(start: int, span: tuple[int, int]) -> int:
        """Characters between a number's first digit and the nearest edge of a name, zero if inside it."""
        if span[0] <= start <= span[1]:
            return 0
        return span[0] - start if start < span[0] else start - span[1]

    def disputed(self) -> list[tuple[Path, int, str, set[str], str]]:
        """Every line naming one run and quoting only scores that run never recorded.

        The run's own scores are looked for over the WHOLE line rather than over the in-band claims,
        because a run may have scored outside the band and a line that quotes it correctly must never
        become a finding on account of where the band was drawn.

        The run's own best comes back with the finding so the report can state both halves — what the
        prose says and what the disk says — on one line, which is the form a reader can act on without
        opening anything.
        """
        findings: list[tuple[Path, int, str, set[str], str]] = []
        for document in self.corpus:
            for number, line in self.readable(document):
                scores = self.quoted(line)
                name = self.named(line) if scores else None
                if name is None or set(QUOTED.findall(line)) & self.recorded.get(name, set()):
                    continue
                claimed = self.about(line, name)
                if not claimed:
                    continue
                findings.append((document, number, name, claimed, f"{self.runs[name]:.4f}"))
        return findings

    def held_by(self, score: str) -> list[str]:
        """The runs whose own records hold this number, which sorts a finding into one of two kinds.

        A quoted score another run genuinely scored is a MIS-ATTRIBUTION: the number is real and the
        name beside it is not, which is how `warm_calib_dense`'s negative verdict stayed unwritten for a
        day while six unrelated lines vouched for its score. A quoted score NO run ever recorded is the
        worse kind — stale, hand-computed, or never measured — and it is the shape knee-8pp8's
        fabricated slope had. The check does not decide which; it says which question to ask.
        """
        return sorted(name for name, known in self.recorded.items() if score in known)

    def sourced(self, scores: set[str]) -> str:
        """Each quoted score beside whoever, if anyone, actually recorded it."""
        parts = []
        for score in sorted(scores):
            held = self.held_by(score)
            parts.append(f"{score} (held by {', '.join(held[:2])})" if held else f"{score} (held by NO run)")
        return ", ".join(parts)

    @staticmethod
    def documents(roots: tuple[Path, ...]) -> list[Path]:
        """Every markdown, python and kernel file under the corpus roots, files and directories alike."""
        found: list[Path] = []
        for root in roots:
            if root.is_dir():
                found.extend(sorted(path for pattern in ("*.md", "*.py") for path in root.rglob(pattern)))
            elif root.exists():
                found.append(root)
        return found

    @staticmethod
    def recorded_scores(processed: Path, runs: dict[str, float]) -> dict[str, set[str]]:
        """Every score each run wrote, as the strings the record would quote.

        Epochs come from the metrics file where one exists; the kept value is always included, because 50 of
        the runs on disk wrote no metrics file at all and their checkpoint is the only account they left.
        """
        scores: dict[str, set[str]] = {}
        for name, best in runs.items():
            known = {f"{best:.4f}"}
            metrics = RunMetrics(processed / f"{name}.pt")
            if metrics.exists():
                known.update(f"{float(row['auc']):.4f}" for row in metrics.rows() if row.get("auc") is not None)
            scores[name] = known
        return scores

    @staticmethod
    def board_entries(path: Path) -> set[str]:
        """Every submission id the board knows, so a claim about an ENTRY is never read as one about a member."""
        if not path.exists():
            return set()
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {str(row["id"]) for row in rows if row.get("id")}


def main() -> None:
    """Report prose that quotes a run at a score it never recorded."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--assert", dest="blocking", action="store_true")
    arguments = command.parsed()

    processed = command.rooted(arguments).models()
    runs = LedgerCheck(*CORPUS[:1]).finished(processed)
    # Same guard as the ledger's, for the same reason: an unmounted data root makes a check that read no
    # run at all print the sentence a clean corpus prints.
    Denominator(len(runs), f"no finished run under {processed}").refuse_empty()

    check = ClaimsCheck(
        runs,
        ClaimsCheck.recorded_scores(processed, runs),
        ClaimsCheck.documents(CORPUS),
        ClaimsCheck.board_entries(BOARD),
    )
    findings = check.disputed()
    for document, number, name, scores, best in findings:
        logger.info(
            "DISPUTED %s:%d names %s and quotes %s; that run's records hold %s",
            document,
            number,
            name,
            check.sourced(scores),
            best,
        )
    logger.info(
        "%d runs on disk, %d documents read, %d disputed line(s)%s",
        len(runs),
        len(check.corpus),
        len(findings),
        "" if arguments.blocking else " (advisory)",
    )
    if findings and arguments.blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
