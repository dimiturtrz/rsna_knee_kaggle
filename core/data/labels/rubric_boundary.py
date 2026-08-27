"""Where the HOST'S rubric and our report reader disagree: `python -m core.data.labels.rubric_boundary`.

EVERY TARGET THIS PROJECT TRAINS ON IS DERIVED FROM THE REPORT, AND THE GROUND TRUTH IS NOT THE REPORT.
The host stated the rule in discussion topic 733491 (read 2026-08-27): the labels come from a
standardized image review, two readers with a third adjudicating, and *"a report may therefore use
language that appears inconsistent with the corresponding image-based label because the original report
and the image-based labels were generated through different processes."* Three divergences are named
outright — a meniscal tear must be DEFINITE (intrasubstance degeneration that does not reach the
articular surface is negative), and marrow-edema wording and cartilage wording *"do not by themselves
determine the contusion, osteoarthritis, or other challenge labels."*

`ReportReading`'s system prompt says to report what THE REPORT SAID and not to diagnose. That is the
right instruction for transcription and the wrong one for this competition, so the reader was never told
the rubric it is being graded by. This module does not fix that. It SIZES it: how many cells of the
table we train on are positive where the host's stated rule says negative.

IT NEEDS NO GROUND TRUTH, WHICH IS WHY IT COSTS NOTHING. The host's rule supplies the correct answer for
the excluded wording directly, so this reads report text against our own targets — no annotations, no
card, no submission slot. It is therefore a measurement over the WHOLE corpus rather than over the 58.

ENGLISH ONLY, AND EVERY RATE CARRIES THE DENOMINATOR IT WAS COMPUTED INSIDE. 59.9% of reports are not
English (`report_language`), so an excluded-wording count is a count inside the English subset and NOT a
rate over the corpus. `severity_boundary` carries the same discipline for the effusion threshold; the
rule it enforces here is that both numbers are printed — the share of the English subset and the share
of the corpus — because this project has three times read the first as the second.

WHAT IT CANNOT SAY. A sentence carrying BOTH excluded and qualifying wording ("degenerative tear
extending to the articular surface") is genuinely positive under the rubric, and one carrying neither is
outside the rule entirely. Both get their own bucket rather than a guess.

AND THE HOST'S OWN PHRASING LANDS IN THE DENIAL BUCKET, WHICH IS THE LARGEST REASON THE COUNT IS A
FLOOR. A report writes the named meniscus case as "degeneration WITHOUT a definite tear", which carries
a negation word and is binned as a denial rather than as excluded wording. Those cells are still
printed — a positive in the denial row is a reader error under any reading — but they are not in the
contradiction line. So that line is a floor on the disagreement and never an estimate of it.
"""

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from core import cli
from core.data.labels.reader_agreement import ENGLISH, MIDPOINT
from core.data.labels.report_language import ReportLanguages

logger = logging.getLogger(__name__)

TABLE = "merged_blend_v1.csv"
"""The table the champion's seats were fitted on, which is the one whose cells this is about.

The eval ruler (`report_labels_v2`) is deliberately NOT the default: a contradiction in the ruler moves
what a holdout number means, and a contradiction in the training table moves what the model learned.
This asks the second question."""

SENTENCE = 130
"""How much of the report sentence is printed beside a contradicted cell — enough to judge it by eye."""

BREAK = re.compile(r"[.;\n]")

DENIED = re.compile(
    r"\b(no|without|absent|negative\s+for|unremarkable|intact|preserved|well[\s-]?maintained|normal)\b",
    re.IGNORECASE,
)
"""A denial anywhere in the sentence takes it out of the cross-tab.

"No meniscal tear" carries `tear`, which is QUALIFYING wording, so an unguarded regex reads a denial as
a definite tear and reports a contradiction the tool invented. `severity_boundary` learned this on
'no significant effusion'; the same trap sits one word away here.

THE SECOND HALF OF THIS LIST WAS ADDED AFTER READING THE OUTPUT, and that is the point of printing the
sentences. `intact` / `preserved` / `well-maintained` / `normal` are how a knee report says negative
WITHOUT a negation word — 'articular cartilage of the patellofemoral joint is well-maintained' appeared
among the first fourteen contradicted cells, carrying `cartilage` into the excluded bucket while saying
the opposite. Counting it inflated the very number this module exists to state."""


@dataclass(frozen=True)
class Rubric:
    """One family of findings the host has said the report does not settle, and the wording either way.

    `mentions` is keyed by TARGET COLUMN rather than being one pattern for the family, because a knee
    report grades the compartments separately in one paragraph — 'medial meniscus tear, lateral meniscus
    intact' is one sentence per column and a family-wide regex would carry the tear to both.
    """

    name: str
    mentions: Mapping[str, tuple[str, ...]]
    excluded: tuple[str, ...]
    qualifying: tuple[str, ...]
    stated: str


MENISCUS = Rubric(
    name="meniscus",
    mentions={
        "Medial Meniscus": (r"medial\s+meniscus", r"medial\s+meniscal"),
        "Lateral Meniscus": (r"lateral\s+meniscus", r"lateral\s+meniscal"),
    },
    excluded=(r"degenerat", r"intrasubstance", r"intra-substance", r"mucoid", r"myxoid", r"grade\s*[12]\b"),
    qualifying=(r"\btear", r"\btorn", r"rupture", r"macerat", r"bucket[\s-]?handle", r"displaced\s+fragment"),
    stated="a definite tear only; intrasubstance degeneration not reaching the articular surface is negative",
)

CONTUSION = Rubric(
    name="contusion",
    mentions={"Contusion": (r"contusion", r"bone\s+bruise", r"marrow\s+o?edema", r"marrow\s+signal")},
    excluded=(r"marrow\s+o?edema", r"marrow\s+signal", r"stress\s+reaction", r"o?edema[\s-]?like"),
    qualifying=(r"contusion", r"bone\s+bruise", r"traumatic\s+marrow"),
    stated="marrow edema wording does not by itself determine the contusion label",
)

OSTEOARTHRITIS = Rubric(
    name="osteoarthritis",
    mentions={
        "Medial OA": (r"medial\s+compartment", r"medial\s+femorotibial"),
        "Lateral OA": (r"lateral\s+compartment", r"lateral\s+femorotibial"),
        "PF OA": (r"patellofemoral", r"patello-femoral", r"retropatellar"),
    },
    excluded=(r"chondral", r"cartilage", r"chondromalacia", r"chondropathy"),
    qualifying=(r"osteoarthr", r"degenerative\s+joint", r"osteophyt", r"joint\s+space\s+narrow", r"\bspur"),
    stated="cartilage findings do not by themselves determine the osteoarthritis labels",
)

RUBRICS: tuple[Rubric, ...] = (MENISCUS, CONTUSION, OSTEOARTHRITIS)
"""Declaration-only, so a fourth family the host names later is an entry and never a branch."""

QUALIFIED = "qualifying wording"
EXCLUDED = "excluded wording only"
BOTH = "both, e.g. degenerative tear reaching the surface"
UNQUALIFIED = "mentioned, neither wording"
DENIAL = "denied"
SILENT = "no sentence for this column"
BUCKETS: tuple[str, ...] = (EXCLUDED, BOTH, QUALIFIED, UNQUALIFIED, DENIAL, SILENT)
"""Ordered with the bucket the host's rule DECIDES first, because that is the one row this tool is for."""


class RubricBoundary:
    """Each report's wording for a finding, read against what our training table says about that cell."""

    def __init__(self, languages: ReportLanguages | None = None) -> None:
        self.languages = languages or ReportLanguages()

    @staticmethod
    def matched(patterns: Sequence[str], text: str) -> bool:
        """Whether any of the patterns appears — compiled here rather than held, so a `Rubric` stays data."""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    def sentences(self, report: str, patterns: tuple[str, ...]) -> list[str]:
        """The report's sentences that name this column's structure, and only those.

        Sentence-scoped for the reason `severity_boundary` is: one paragraph grades every compartment,
        and a report-wide regex reads the medial meniscus's tear onto the lateral one.
        """
        return [part for part in BREAK.split(report) if self.matched(patterns, part)]

    def bucket(self, report: str, rubric: Rubric, column: str) -> str:
        """Which side of the host's stated rule this report's own wording puts this cell on."""
        seen = self.sentences(report, rubric.mentions[column])
        if not seen:
            return SILENT
        found = " ".join(part for part in seen if not DENIED.search(part))
        if not found:
            return DENIAL
        excluded = self.matched(rubric.excluded, found)
        qualifying = self.matched(rubric.qualifying, found)
        if excluded and qualifying:
            return BOTH
        if excluded:
            return EXCLUDED
        return QUALIFIED if qualifying else UNQUALIFIED

    def table(self, reports: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
        """One row per (English study, column): its bucket, and whether our table calls that cell positive.

        The join is on study UID and the row set is the REPORTS', not the targets' — a target table that
        covers a merged corpus carries studies with no report here, and counting those as silent would
        put a corpus difference inside a reader measurement.
        """
        indexed = targets.set_index(targets.columns[0])
        rows = []
        for _, study in reports.iterrows():
            report = str(study["Report"])
            identifier = study["StudyInstanceUID"]
            if self.languages.language(report) != ENGLISH or identifier not in indexed.index:
                continue
            for rubric in RUBRICS:
                for column in rubric.mentions:
                    value = pd.to_numeric(indexed.at[identifier, column], errors="coerce")
                    if pd.isna(value):
                        continue
                    bucket = self.bucket(report, rubric, column)
                    said = " ".join(self.sentences(report, rubric.mentions[column])).strip()
                    anywhere = self.matched(rubric.qualifying, report)
                    rows.append((identifier, rubric.name, column, bucket, float(value) > MIDPOINT, said, anywhere))
        return pd.DataFrame(rows, columns=["study", "rubric", "column", "bucket", "positive", "said", "anywhere"])

    def report(self, table: pd.DataFrame, reports: int, english: int) -> list[str]:
        """The cross-tab per family, with BOTH denominators on the first line.

        A share of the excluded bucket and a share of the corpus are different numbers and this project
        has read the first as the second three times in one session. Both are printed, always.
        """
        lines = [f"{english} English reports of {reports} carrying a report ({len(table)} cells cross-tabbed)"]
        for rubric in RUBRICS:
            family = table[table["rubric"] == rubric.name]
            lines.append(f"{rubric.name} — host: {rubric.stated}")
            for bucket in BUCKETS:
                rows = family[family["bucket"] == bucket]
                if rows.empty:
                    continue
                positives = int(rows["positive"].sum())
                share = positives / len(rows)
                lines.append(f"  {bucket:44s} {len(rows):5d} cells   our table positive {positives:5d}  ({share:.1%})")
            decided = family[family["bucket"] == EXCLUDED]
            wrong = int(decided["positive"].sum())
            cells = len(family)
            lines.append(
                f"  CONTRADICTED by the host's rule: {wrong} cells"
                f" — {wrong / len(decided):.1%} of the excluded bucket,"
                f" {wrong / cells:.2%} of this family's {cells} English cells"
                if len(decided)
                else "  CONTRADICTED by the host's rule: no cell carries excluded wording alone"
            )
        return lines

    def attribution(self, table: pd.DataFrame) -> list[str]:
        """For the cells NO sentence names, where a positive came from: another column, a global sentence, or nothing.

        THIS IS A DIFFERENT QUESTION FROM THE HOST'S RULE and it belongs beside it because it shares the
        cross-tab. The rubric asks 'the report names cartilage, does that make OA positive'. This asks
        'the report names this compartment NOWHERE, and our table says positive anyway — where did the
        value come from'. Three mechanisms, and they are not the same thing: SPILLOVER graded every
        compartment off one compartment's finding, which is an attribution error; GLOBAL read an
        unlocalised sentence ('tricompartmental degenerative change') onto all three columns, which is
        CORRECT; UNSUPPORTED put a positive where the report carries no wording of that family anywhere.

        THE GLOBAL ROW IS WHY THIS IS NOT A SUBTRACTION. Without it every unlocalised report reads as an
        invented label, and the headline would be the compartment regex's blind spot rather than a fact
        about the table. UNSUPPORTED is still not automatically an error either — the ledger's
        `confident224` result (-0.0071) measured real signal in unknown cells at 0.2875 weight, so a
        prior may carry what the report withholds. This says which pile a cell is in, not which is wrong.
        """
        lines = []
        for rubric in RUBRICS:
            family = table[table["rubric"] == rubric.name]
            silent = family[(family["bucket"] == SILENT) & family["positive"]]
            if silent.empty:
                lines.append(f"{rubric.name} — no positive cell is unnamed by the report")
                continue
            supported = set(family[(family["bucket"] != SILENT) & family["positive"]]["study"])
            spilled = silent["study"].isin(supported)
            unlocalised = int((~spilled & silent["anywhere"]).sum())
            unsupported = int((~spilled & ~silent["anywhere"]).sum())
            lines.append(
                f"{rubric.name} — {len(silent)} positives on columns no sentence names:"
                f" {int(spilled.sum())} spillover ({spilled.mean():.1%}, another column of the same study"
                f" is named AND positive), {unlocalised} global (this family's wording is in the report,"
                f" localised to no compartment), {unsupported} unsupported (the wording is nowhere in it)"
            )
        return lines

    def contradictions(self, table: pd.DataFrame, limit: int) -> list[str]:
        """The cells a person should open: excluded wording, and our table calls them positive anyway.

        Separate from `report` for the reason `severity_boundary` separates its own — the count is the
        claim and the identifiers are what makes it checkable instead of a rate quoted at second hand.
        """
        wrong = table[(table["bucket"] == EXCLUDED) & table["positive"]]
        rows = wrong.head(limit).itertuples()
        return [f"  {row.column:12s} {row.study[-12:]}  {row.said[:SENTENCE]}" for row in rows]


def main() -> None:
    """Size the population where the host's stated rubric and our training table disagree."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", default=TABLE, help="the target table to cross-tab, under labels/")
    command.parser.add_argument("--limit", type=int, default=20, help="how many contradicted cells to name")
    arguments = command.parsed()

    root = command.rooted(arguments)
    reports = pd.read_csv(root.raw() / "train.csv")[["StudyInstanceUID", "Report"]].dropna()
    targets = pd.read_csv(Path(root.processed()) / "labels" / arguments.labels)

    boundary = RubricBoundary()
    table = boundary.table(reports, targets)
    english = table["study"].nunique()
    command.report(boundary.report(table, len(reports), english))
    command.report(["positives on columns the report never names — spillover, global wording, or nothing:"])
    command.report(boundary.attribution(table))
    command.report(["cells the host's rule says are negative and our table calls positive:"])
    command.report(boundary.contradictions(table, arguments.limit))


if __name__ == "__main__":
    main()
