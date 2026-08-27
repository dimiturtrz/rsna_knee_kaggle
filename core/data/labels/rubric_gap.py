"""How many derived targets does the host's published rubric say are wrong? `python -m core.data.labels.rubric_gap`.

THE HOST STATED THE RULES THE LABELS WERE MADE UNDER, and they are not the rules our reader was given.
Discussion topic 733343 publishes the image-review criteria for all twelve findings, and topic 733491
answers with the meniscus one restated:

    "In each case, ambiguous or borderline findings ('on the fence') were graded as negative to favor
     specificity. ... A report may therefore use language that appears inconsistent with the
     corresponding image-based label because the original report and the image-based labels were
     generated through different processes."

Every target this project trains on is derived from the REPORT, by a reader whose system prompt says
"say what THE REPORT SAID about it. Do not diagnose, do not infer". That is the right instruction for
transcription and the wrong one here: a report reading "intrasubstance degeneration without definite
tear", "mild ACL signal change without discontinuity" or "low-grade MCL sprain" earns a grade above
absent, and the competition's own label for each is 0.

THREE OF THE TWELVE RULES HAVE THE SAME SHAPE, so one ruler measures them all: a MENTION of the
structure, a vocabulary the rubric ACCEPTS, a vocabulary it names and calls NEGATIVE anyway, and a
denial that must be subtracted before either is searched for. `RUBRIC` holds one `RubricRule` per
finding; adding a fourth is a rule, not a module.

THE RULES SUPPLY THE ANSWER, SO THIS NEEDS NO GROUND TRUTH. Unlike `severity_boundary`, which can only
cross-tab inside the 58 annotated studies, the subthreshold class is negative BY THE STATED RULE. So the
population at risk is the whole corpus, and the measurement is: how many studies carry that wording, and
how many of those our own targets call positive. The annotated studies are read beside it as a check on
the rules themselves, not as their denominator.

ENGLISH ONLY, AND THE DENOMINATOR SAYS SO. 59.9% of the corpus is not English (`report_language`), and
inventing a subthreshold vocabulary for eight more languages would put the tool's own defect inside the
number it reports. Every line names how many studies it rests on, and the headline rate is over the
CORPUS, not over the subset that parsed — a rate inside the subset that happened to parse is not a rate
over the corpus.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from core import cli
from core.data.labels.reader_agreement import ENGLISH, MIDPOINT
from core.data.labels.report_language import ReportLanguages
from core.findings import COMPARTMENT_PAIRS, STUDY, TARGETS

logger = logging.getLogger(__name__)

BREAK = re.compile(r"[.;\n]")

DENIAL_WINDOW = r"\b(?:no|without|not|negative\s+for|absent|denies)\b[^,]{0,40}?(?:definite\s+)?(?:"
"""A negation, then at most a clause, then whatever the rule calls positive.

"No definite tear" and "without discontinuity" are the exact phrases the host's rules turn on, and a
bare search for the positive vocabulary reads both as positive — which would file the very studies this
module exists to count under the accepted class. Bounded by a comma and a short window so a denial of
one clause cannot silence a finding reported later in the same sentence.
"""

MEETS = "meets the stated threshold"
SUBTHRESHOLD = "named, but under the stated threshold"
DENIED = "denied outright"
UNQUALIFIED = "mentioned, neither vocabulary"
UNSIDED = "sentence names no side"
SILENT = "no sentence about it"
BUCKETS: tuple[str, ...] = (SUBTHRESHOLD, MEETS, DENIED, UNQUALIFIED, UNSIDED, SILENT)
"""Ordered with the class the rules call negative first, since it is the one the count is about."""

MENTIONED = 20
"""How many contradicted cells to print. Enough to check the patterns by eye, not the whole list."""


@dataclass
class RubricRule:
    """One finding's published criterion, as the three vocabularies a report can satisfy it with.

    Beside `RubricGap` rather than in a rules module because a rule is not a type shared by several
    producers — it is this measurement's argument, and it means nothing without the ruler that reads it.
    """

    columns: tuple[str, ...]
    mention: str
    accepted: str
    subthreshold: str
    criterion: str
    mentions: re.Pattern[str] = field(init=False)
    positive: re.Pattern[str] = field(init=False)
    under: re.Pattern[str] = field(init=False)
    denies: re.Pattern[str] = field(init=False)

    def __post_init__(self) -> None:
        self.mentions = re.compile(self.mention, re.IGNORECASE)
        self.positive = re.compile(self.accepted, re.IGNORECASE)
        self.under = re.compile(self.subthreshold, re.IGNORECASE)
        self.denies = re.compile(DENIAL_WINDOW + self.accepted + r")", re.IGNORECASE)

    def sided(self) -> bool:
        """Whether a sentence has to name WHICH column it speaks about before it counts for one.

        True only for a compartment pair, where the two columns are one finding located to one side:
        "medial meniscal tear, lateral meniscus intact" is one sentence carrying one positive and one
        negative, and a sentence-only scope would grade both columns torn. ACL and MCL have one column
        each, so any sentence mentioning them speaks about that column.
        """
        return len(self.columns) > 1


MENISCUS_RULE = RubricRule(
    columns=COMPARTMENT_PAIRS[0],
    mention=r"menisc",
    accepted=r"\btear|\btorn|rupture|macerat|bucket.?handle|(?:reach|extend|contact)\w*\s+(?:to\s+|the\s+)*"
    r"(?:articular\s+)?surface",
    subthreshold=r"degenerat|intrasubstance|intra-substance|mucoid|myxoid",
    criterion="signal definitely contacting the articular surface on two or more images, or a morphologic "
    "abnormality; intrasubstance degeneration is negative",
)
ACL_RULE = RubricRule(
    columns=("ACL",),
    mention=r"\bACL\b|anterior\s+cruciate",
    accepted=r"\btear|\btorn|rupture|discontinu|disrupt|full.thickness|complete",
    subthreshold=r"sprain|thicken|signal\s+change|increased\s+signal|mild|partial|degenerat|attenuat",
    criterion="more than half the fibres, or discontinuity; mild signal change, degeneration or thickening "
    "without discontinuity is negative",
)
MCL_RULE = RubricRule(
    columns=("MCL",),
    mention=r"\bMCL\b|medial\s+collateral",
    accepted=r"high.?grade|grade\s*(?:3|III)\b|complete|\brupture|full.thickness|\btorn",
    subthreshold=r"low.?grade|grade\s*(?:1|2|I{1,2})\b|sprain|strain|chronic|remote|thicken|oedema|edema",
    criterion="high-grade ACUTE injury only; low-grade sprains and chronic or remote stress change are negative",
)
RUBRIC: tuple[RubricRule, ...] = (MENISCUS_RULE, ACL_RULE, MCL_RULE)
"""The three published criteria whose negative class the report's own words name. The other nine findings
turn on a size or an image feature ("moderate or large", "one centimetre of >50%-thickness loss") that a
report states in its own units — `severity_boundary` is that shape, and it needs the annotations."""


class RubricGap:
    """Each report's sentences about a finding, bucketed by the host's rule and read beside our target."""

    def __init__(self, languages: ReportLanguages | None = None) -> None:
        self.languages = languages or ReportLanguages()

    def _cells(self, table: pd.DataFrame, columns: tuple[str, ...]) -> dict[str, dict[str, float]]:
        """The rule's columns as plain floats keyed by study, for the derived table and the truth alike.

        A dict rather than `frame.at[study, column]` because the frames come from `read_csv` with no
        declared dtypes, so every cell reads back as an untyped scalar the pair-key indexer cannot check.
        """
        indexed = table.set_index(STUDY)
        return {
            column: {str(study): float(value) for study, value in indexed[column].to_dict().items()}
            for column in columns
        }

    def sentences(self, rule: RubricRule, report: str, column: str) -> list[str]:
        """The report's sentences about this finding, scoped to this column when the rule is sided."""
        parts = [part for part in BREAK.split(report) if rule.mentions.search(part)]
        if not rule.sided():
            return parts
        wanted = re.compile(column.split()[0], re.IGNORECASE)
        return [part for part in parts if wanted.search(part)]

    def bucket(self, rule: RubricRule, report: str, column: str) -> str:
        """Which class of the stated rule this report's sentences put this column in."""
        seen = self.sentences(rule, report, column)
        if not seen:
            return UNSIDED if rule.mentions.search(report) else SILENT
        found = " ".join(seen)
        if rule.positive.search(rule.denies.sub("", found)):
            return MEETS
        if rule.under.search(found):
            return SUBTHRESHOLD
        return DENIED if rule.denies.search(found) else UNQUALIFIED

    def table(self, rule: RubricRule, reports: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        """One row per English study per column of this rule: its bucket, and what our target says."""
        derived = self._cells(labels, rule.columns)
        first = rule.columns[0]
        rows = []
        for _, study in reports.iterrows():
            report, name = str(study["Report"]), str(study[STUDY])
            if self.languages.language(report) != ENGLISH or name not in derived[first]:
                continue
            rows.extend(
                (name, column, self.bucket(rule, report, column), derived[column][name] > MIDPOINT)
                for column in rule.columns
            )
        return pd.DataFrame(rows, columns=["study", "column", "bucket", "positive"])

    def report(self, rule: RubricRule, table: pd.DataFrame, corpus: int) -> list[str]:
        """The cross-tab, with the corpus denominator on the first line and the headline rate over it.

        The rate that matters is the share of the WHOLE corpus whose target the stated rule contradicts.
        Reporting it over the English subset would overstate it by the reciprocal of the parse rate,
        which is the arithmetic this project has got wrong three times in one session.
        """
        studies = table["study"].nunique()
        cells = corpus * len(rule.columns)
        lines = [
            f"{'/'.join(rule.columns)} — {rule.criterion}",
            f"  {studies} English studies of {corpus} in the corpus ({studies / corpus:.1%})",
        ]
        for bucket in BUCKETS:
            rows = table[table["bucket"] == bucket]
            if rows.empty:
                continue
            lines.append(
                f"    {bucket:34s} {len(rows):4d} cells   our target positive {int(rows['positive'].sum()):4d}"
            )
        wrong = self.contradicted(table)
        lines.append(
            f"  contradicted: {len(wrong)} cells — {len(wrong) / cells:.2%} of the corpus's {cells},"
            f" a FLOOR, since the {1 - studies / corpus:.0%} this tool cannot read carry the same wording"
        )
        return lines

    def contradicted(self, table: pd.DataFrame) -> pd.DataFrame:
        """The rows the host's rule says are mislabelled: named under threshold, our target positive.

        Separate from `report` because these are the cells a person should open, and because a re-read
        under a corrected prompt is scored by whether it moves exactly these.
        """
        return table[(table["bucket"] == SUBTHRESHOLD) & table["positive"]]

    def annotated(self, rule: RubricRule, table: pd.DataFrame, truth: pd.DataFrame) -> list[str]:
        """The same buckets read against the 58 radiologist annotations — a check on the RULE, not on us.

        Small n by construction, so it settles nothing on its own. But if the annotation agrees with the
        stated rule where both exist, the rule is worth re-reading the whole corpus under; if it does not,
        the defect is in the ruler and no better reader recovers it.
        """
        graded = truth.dropna(subset=TARGETS, how="all")
        cells = self._cells(graded, rule.columns)
        rows = table[table["study"].isin(cells[rule.columns[0]])].copy()
        rows["truth"] = [cells[str(row.column)][str(row.study)] > MIDPOINT for row in rows.itertuples()]
        lines = [f"  {rows['study'].nunique()} of the {len(graded)} annotated studies are English:"]
        for bucket in BUCKETS:
            seen = rows[rows["bucket"] == bucket]
            if seen.empty:
                continue
            lines.append(
                f"    {bucket:34s} {len(seen):3d} cells   annotated positive {int(seen['truth'].sum()):3d}"
                f"   our target positive {int(seen['positive'].sum()):3d}"
            )
        return lines


def main() -> None:
    """Count the targets the host's stated rubric contradicts, over the corpus that produced them."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, help="the derived labels table; defaults to report_labels_v2")
    arguments = command.parsed()

    root = command.rooted(arguments)
    truth = pd.read_csv(root.raw() / "train.csv")
    labels = pd.read_csv(arguments.labels or root.processed() / "labels" / "report_labels_v2.csv")

    gap = RubricGap()
    reports = truth[[STUDY, "Report"]]
    total = 0
    for rule in RUBRIC:
        table = gap.table(rule, reports, labels)
        command.report(gap.report(rule, table, len(truth)))
        command.report(gap.annotated(rule, table, truth))
        wrong = gap.contradicted(table)
        total += len(wrong)
        # The identifiers, because a count of contradicted cells is a claim and the studies are evidence.
        command.report([f"    {row.study}  {row.column}" for row in wrong.head(MENTIONED).itertuples()])
    command.report([f"{total} contradicted cells over the three rules whose negative class the report names"])


if __name__ == "__main__":
    main()
