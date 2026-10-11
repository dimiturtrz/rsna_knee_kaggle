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

NINE OF THE TWELVE COLUMNS HAVE THE SAME SHAPE, so one ruler measures them all: a MENTION of the
structure, a vocabulary the rubric ACCEPTS, a vocabulary it names and calls NEGATIVE anyway, and a
denial that must be subtracted before either is searched for. `RUBRIC` holds one `RubricRule` per
finding; adding another is a rule, not a module. They come in two shapes — a TEAR word (meniscus, ACL,
MCL) and a SIZE word (effusion, Baker's, the three OA compartments) — and the size vocabulary is
imported from `severity_boundary`, which owns it.

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
from core.data.labels.severity_boundary import LARGE_WORDS, MENTIONS, SMALL_WORDS
from core.findings import COMPARTMENT_PAIRS, CRITERIA, STUDY, TARGETS

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

    def criterion(self) -> str:
        """The host's own words for this rule, from the vocabulary both this and the reader read."""
        return CRITERIA[self.columns[0]]


MENISCUS_RULE = RubricRule(
    columns=COMPARTMENT_PAIRS[0],
    mention=r"menisc",
    accepted=r"\btear|\btorn|rupture|macerat|bucket.?handle|truncat|diminutive|displaced|"
    r"(?:reach|extend|contact)\w*\s+(?:to\s+|the\s+)*(?:articular\s+)?surface",
    subthreshold=r"degenerat|intrasubstance|intra-substance|mucoid|myxoid",
)
ACL_RULE = RubricRule(
    columns=("ACL",),
    mention=r"\bACL\b|anterior\s+cruciate",
    accepted=r"\btear|\btorn|rupture|discontinu|disrupt|full.thickness|complete",
    subthreshold=r"sprain|thicken|signal\s+change|increased\s+signal|mild|partial|degenerat|attenuat",
)
MCL_RULE = RubricRule(
    columns=("MCL",),
    mention=r"\bMCL\b|medial\s+collateral",
    accepted=r"high.?grade|grade\s*(?:3|III)\b|complete|\brupture|full.thickness|\btorn|disrupt",
    subthreshold=r"low.?grade|grade\s*(?:1|2|I{1,2})\b|sprain|strain|chronic|remote",
)
"""EDEMA IS NOT IN THE MCL SUBTHRESHOLD VOCABULARY, and the first version of this rule had it there.

The published criterion reads "a high-grade partial or complete acute tear ... WITH DISRUPTED FIBERS AND
EDEMA WITHIN AND ADJACENT TO THE LIGAMENT" — so oedema is part of what makes an MCL injury POSITIVE, and
listing it as subthreshold read the positive class as the negative one. It cost 3 of 7 annotated cells
disagreeing with the ruler before the thread was read verbatim rather than from a summary of it.
"""
SIZE_ACCEPTED = "|".join(LARGE_WORDS)
SIZE_SUBTHRESHOLD = "|".join(SMALL_WORDS)
"""The size threshold, from `severity_boundary`, which owns this vocabulary and cross-tabs it on Effusion.

Five more of the twelve criteria are stated as a SIZE — "a moderate or large amount of fluid", "a
moderate or large fluid collection", "a moderate or large area of high-grade cartilage loss" — and a size
word is the same shape as a tear word: a vocabulary the rule accepts, one it names and calls negative, and
a denial that must come out first. So they are `RubricRule`s here rather than a second module, and the
words are IMPORTED rather than restated, because two homes for one threshold is how the two drift apart.
"""

CARTILAGE_ACCEPTED = r"|severe|advanced|high.?grade|full.?thickness|bone.on.bone|grade\s*(?:3|4|III|IV)\b"
CARTILAGE_SUBTHRESHOLD = r"|early|low.?grade|fibrillat|fissur|grade\s*(?:1|2|I{1,2})\b"
"""What the OA criterion adds to the size words: it names a cartilage-loss GRADE as well as an area.

"A moderate or large area (roughly 1 cm or greater) of high-grade cartilage loss, defined as greater than
50 percent of cartilage thickness" — so a report saying "grade 2 chondromalacia" or "early degenerative
change" is under the stated threshold however it words the area, and "full-thickness cartilage loss" meets
it however it words the size.
"""

EFFUSION_RULE = RubricRule(
    columns=("Effusion",),
    mention="|".join(MENTIONS),
    accepted=SIZE_ACCEPTED,
    subthreshold=SIZE_SUBTHRESHOLD,
)
BAKERS_RULE = RubricRule(
    columns=("Baker's",),
    mention=r"baker|popliteal\s+cyst|gastrocnemio",
    accepted=SIZE_ACCEPTED,
    subthreshold=SIZE_SUBTHRESHOLD,
)
MEDIAL_OA_RULE = RubricRule(
    columns=("Medial OA",),
    mention=r"medial\s+(?:compartment|femorotibial|tibiofemoral|joint\s+space)",
    accepted=SIZE_ACCEPTED + CARTILAGE_ACCEPTED,
    subthreshold=SIZE_SUBTHRESHOLD + CARTILAGE_SUBTHRESHOLD,
)
LATERAL_OA_RULE = RubricRule(
    columns=("Lateral OA",),
    mention=r"lateral\s+(?:compartment|femorotibial|tibiofemoral|joint\s+space)",
    accepted=SIZE_ACCEPTED + CARTILAGE_ACCEPTED,
    subthreshold=SIZE_SUBTHRESHOLD + CARTILAGE_SUBTHRESHOLD,
)
PF_OA_RULE = RubricRule(
    columns=("PF OA",),
    mention=r"patellofemoral|patellar\s+cartilage|trochlea|retropatellar",
    accepted=SIZE_ACCEPTED + CARTILAGE_ACCEPTED,
    subthreshold=SIZE_SUBTHRESHOLD + CARTILAGE_SUBTHRESHOLD,
)

RUBRIC: tuple[RubricRule, ...] = (
    MENISCUS_RULE,
    ACL_RULE,
    MCL_RULE,
    EFFUSION_RULE,
    BAKERS_RULE,
    MEDIAL_OA_RULE,
    LATERAL_OA_RULE,
    PF_OA_RULE,
)
"""Nine of the twelve columns, in the two shapes a report states an image criterion in: a TEAR word and a
SIZE word. The other three — Synovitis, Contusion, Fracture — name an image feature the report either
asserts or does not, with no threshold vocabulary for the rule to read, so this ruler is blind to them.

Every one of these is worth exactly as much as its annotation cross-check says (`backed`), which is what
caught MCL and what disqualifies the SIZE family — the report's size word does not predict the
radiologist's size judgement, and the largest arm in the table is the least trustworthy one. A rule being
in this tuple is not a claim that its cells are mislabelled; `main` prints the two totals apart.
"""


@dataclass(frozen=True, eq=False)
class RuleEvidence:
    """One rule's bucketed table read beside the radiologists' annotations — what `backed` is judged on.

    Produced by pairing `RubricGap.table` with the train table: the rule says what each bucket SHOULD be,
    the table says where each report fell, and `truth` is the only witness that can disagree with both.
    """

    rule: RubricRule
    table: pd.DataFrame
    truth: pd.DataFrame


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
            f"{'/'.join(rule.columns)} — {rule.criterion()}",
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

    def backed(self, evidence: RuleEvidence) -> bool:
        """Do the radiologists' own annotations agree that this rule's subthreshold class is negative?

        THE COUNT IS ONLY WORTH WHAT THIS SAYS. A rule earns its cells by predicting the annotation where
        both exist: every subthreshold cell should be annotated 0, because that is what the stated rule
        claims. A rule whose subthreshold cells are annotated positive half the time is not measuring a
        mislabelled corpus — it is measuring its own disagreement with the radiologist, and no re-read
        recovers those cells because there is nothing wrong with them.

        ZERO COUNTEREXAMPLES, NOT A MAJORITY. The rule's claim is that this class is negative — a rate of
        0, not a rate under a half — so one annotated positive refutes it and no threshold has to be
        invented. A majority test was written first and called MCL backed at 3 of 7, which is the arm this
        project had already flagged by hand; that disagreement is what showed the test, not the arm, was
        wrong. A rule with no annotated subthreshold cell is NOT backed either: unmeasured is not agreed.

        It disqualifies the whole SIZE family. Effusion — the largest arm of all at 524 cells — has 6 of
        its 12 annotated subthreshold cells annotated POSITIVE. `severity_boundary` was built for exactly
        that boundary, and two competitors report the same effusion sentence annotated both ways (topic
        735826). A size word in a report does not predict the radiologist's size judgement on the image.

        n is 1 to 12 per rule, so a pass here is weak evidence and a fail is strong evidence. `annotated`
        prints the cell counts beside the verdict for that reason — read them, do not read the word alone.
        """
        sub = self.subthreshold(evidence)
        return bool(len(sub)) and not int(sub["truth"].sum())

    def subthreshold(self, evidence: RuleEvidence) -> pd.DataFrame:
        """The annotated cells this rule calls subthreshold — the only class its claim is falsifiable on."""
        rows = self._annotated_rows(evidence)
        return rows[rows["bucket"] == SUBTHRESHOLD]

    def _annotated_rows(self, evidence: RuleEvidence) -> pd.DataFrame:
        """This rule's buckets over the annotated studies, with the radiologist's verdict beside ours."""
        graded = evidence.truth.dropna(subset=TARGETS, how="all")
        columns = evidence.rule.columns
        cells = self._cells(graded, columns)
        rows = evidence.table[evidence.table["study"].isin(cells[columns[0]])].copy()
        rows["truth"] = [cells[str(row.column)][str(row.study)] > MIDPOINT for row in rows.itertuples()]
        return rows

    def annotated(self, evidence: RuleEvidence) -> list[str]:
        """The same buckets read against the 58 radiologist annotations — a check on the RULE, not on us.

        Small n by construction, so it settles nothing on its own. But if the annotation agrees with the
        stated rule where both exist, the rule is worth re-reading the whole corpus under; if it does not,
        the defect is in the ruler and no better reader recovers it.
        """
        rows = self._annotated_rows(evidence)
        graded = len(evidence.truth.dropna(subset=TARGETS, how="all"))
        sub = self.subthreshold(evidence)
        verdict = "BACKED" if self.backed(evidence) else "NOT BACKED — do not bank this rule's cells"
        lines = [
            f"  {rows['study'].nunique()} of the {graded} annotated studies are English: {verdict}"
            f" ({int(sub['truth'].sum())} of {len(sub)} subthreshold cells annotated positive)"
        ]
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
    truth = pd.read_csv(root.train_table())
    labels = pd.read_csv(arguments.labels or root.labels())

    gap = RubricGap()
    reports = truth[[STUDY, "Report"]]
    totals = {True: 0, False: 0}
    for rule in RUBRIC:
        table = gap.table(rule, reports, labels)
        command.report(gap.report(rule, table, len(truth)))
        evidence = RuleEvidence(rule, table, truth)
        command.report(gap.annotated(evidence))
        wrong = gap.contradicted(table)
        totals[gap.backed(evidence)] += len(wrong)
        # The identifiers, because a count of contradicted cells is a claim and the studies are evidence.
        command.report([f"    {row.study}  {row.column}" for row in wrong.head(MENTIONED).itertuples()])
    # Two totals, never one: summing a rule the annotations refuse into the headline is how 448 became 1600.
    command.report(
        [
            f"{totals[True]} contradicted cells over the rules the annotations BACK",
            f"{totals[False]} more under rules they do NOT — measuring the ruler, not the corpus",
        ]
    )


if __name__ == "__main__":
    main()
