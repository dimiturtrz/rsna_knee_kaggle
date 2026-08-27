"""How many derived meniscus targets does the host's rubric say are wrong? `python -m core.data.labels.tear_rubric`.

THE HOST STATED THE RULE THE LABELS WERE MADE UNDER, and it is not the rule our reader was given.
Discussion topic 733491, answered 2026-08-27:

    "For meniscal tears, the target is a definite tear; intrasubstance degenerative signals that do not
     reach the articular surface are negative. ... A report may therefore use language that appears
     inconsistent with the corresponding image-based label because the original report and the
     image-based labels were generated through different processes."

Every target this project trains on is derived from the REPORT, by a reader whose system prompt says
"say what THE REPORT SAID about it. Do not diagnose, do not infer". That is the right instruction for
transcription and the wrong one here: a report that says "intrasubstance degeneration without definite
tear" gets a meniscus grade above absent, and the competition's own label for it is 0.

THE RULE SUPPLIES THE ANSWER, SO THIS NEEDS NO GROUND TRUTH. Unlike `severity_boundary`, which can only
cross-tab inside the 58 annotated studies, the degenerate-signal-without-a-tear class is negative BY THE
STATED RULE. So the population at risk is the whole corpus, and the measurement is: how many studies
carry that wording, and how many of those our own targets call positive. The annotated studies are read
beside it as a check on the rule itself, not as its denominator.

ENGLISH ONLY, AND THE DENOMINATOR SAYS SO. 59.9% of the corpus is not English (`report_language`), and
inventing a degeneration vocabulary for eight more languages would put the tool's own defect inside the
number it reports. Every line names how many studies it rests on, and the headline rate is over the
CORPUS, not over the subset that parsed — a rate inside the subset that happened to parse is not a rate
over the corpus.
"""

import logging
import re
from pathlib import Path

import pandas as pd

from core import cli
from core.data.labels.reader_agreement import ENGLISH, MIDPOINT
from core.data.labels.report_language import ReportLanguages
from core.findings import COMPARTMENT_PAIRS, STUDY, TARGETS

logger = logging.getLogger(__name__)

MENISCUS: tuple[str, str] = COMPARTMENT_PAIRS[0]
"""The two columns the stated rule speaks about, taken from the pair vocabulary rather than spelled again."""

MENTIONS = re.compile(r"menisc", re.IGNORECASE)
BREAK = re.compile(r"[.;\n]")

DEGENERATIVE = re.compile(r"degenerat|intrasubstance|intra-substance|mucoid|myxoid", re.IGNORECASE)
"""The signal the rule calls negative: substance changes that do not reach the articular surface."""

TEAR = re.compile(r"\btear|\btorn|rupture|macerat|bucket.?handle", re.IGNORECASE)
"""What the rule calls positive. `\\btear` rather than `tear` so `wear` and `shear` do not read as one."""

DENIED = re.compile(
    r"\b(no|without|not|negative\s+for|absent)\b[^,]{0,40}?(definite\s+)?(tear|torn|rupture)", re.IGNORECASE
)
"""A tear word inside a denial is not a tear.

"No definite tear" and "without definite meniscal tear" are the exact phrases the host's rule turns on,
and a bare `TEAR` search reads both as positive — which would put the very studies this module exists to
count into the tear bucket. Bounded by a comma and a short window so a denial of one clause cannot
silence a tear reported later in the same sentence.
"""

SURFACE = re.compile(r"(reach|extend|contact)\w*\s+(to\s+|the\s+)*(articular\s+)?surface", re.IGNORECASE)
"""Surface contact is what promotes a degenerative signal to a tear, so a sentence claiming it is not
the negative class even when the denial regex has taken its tear word away."""

DEGENERATE_ONLY = "degenerative signal, no definite tear"
TORN = "definite tear"
DENIAL = "tear denied, no degeneration"
UNQUALIFIED = "meniscus mentioned, neither word"
UNSIDED = "meniscus sentence names no side"
SILENT = "no meniscus sentence"
BUCKETS: tuple[str, ...] = (DEGENERATE_ONLY, TORN, DENIAL, UNQUALIFIED, UNSIDED, SILENT)
"""Ordered with the class the rule calls negative first, since it is the one the count is about."""

MENTIONED = 20
"""How many contradicted cells to print. Enough to check the regex by eye, not the whole list."""


class TearRubric:
    """Each report's meniscus sentence, bucketed by the host's rule and read beside our derived target."""

    def __init__(self, languages: ReportLanguages | None = None) -> None:
        self.languages = languages or ReportLanguages()

    def _cells(self, table: pd.DataFrame) -> dict[str, dict[str, float]]:
        """The two meniscus columns as plain floats keyed by study, for the derived table and the truth alike.

        A dict rather than `frame.at[study, side]` because the frames come from `read_csv` with no
        declared dtypes, so every cell reads back as an untyped scalar the pair-key indexer cannot check.
        """
        indexed = table.set_index(STUDY)
        return {
            side: {str(study): float(value) for study, value in indexed[side].to_dict().items()} for side in MENISCUS
        }

    def sentences(self, report: str, side: str) -> list[str]:
        """The report's meniscus sentences naming this side, and only those.

        Side-scoped as well as sentence-scoped because the two meniscus columns are the same finding
        located to one compartment: "medial meniscal tear, lateral meniscus intact" is one sentence
        carrying one positive and one negative, and a sentence-only scope would grade both columns torn.
        """
        wanted = re.compile(side.split()[0], re.IGNORECASE)
        return [part for part in BREAK.split(report) if MENTIONS.search(part) and wanted.search(part)]

    def bucket(self, report: str, side: str) -> str:
        """Which class of the stated rule this report's meniscus sentence puts this side in."""
        seen = self.sentences(report, side)
        if not seen:
            return UNSIDED if MENTIONS.search(report) else SILENT
        found = " ".join(seen)
        torn = TEAR.search(DENIED.sub("", found)) or SURFACE.search(found)
        if torn:
            return TORN
        if DEGENERATIVE.search(found):
            return DEGENERATE_ONLY
        return DENIAL if DENIED.search(found) else UNQUALIFIED

    def table(self, reports: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
        """One row per English study per meniscus side: its bucket, and what our derived target says."""
        derived = self._cells(labels)
        rows = []
        for _, study in reports.iterrows():
            report, name = str(study["Report"]), str(study[STUDY])
            if self.languages.language(report) != ENGLISH or name not in derived[MENISCUS[0]]:
                continue
            rows.extend((name, side, self.bucket(report, side), derived[side][name] > MIDPOINT) for side in MENISCUS)
        return pd.DataFrame(rows, columns=["study", "side", "bucket", "positive"])

    def report(self, table: pd.DataFrame, corpus: int) -> list[str]:
        """The cross-tab, with the corpus denominator on the first line and the headline rate over it.

        The rate that matters is the share of the WHOLE corpus whose meniscus target the stated rule
        contradicts. Reporting it over the English subset would overstate it by the reciprocal of the
        parse rate, which is the arithmetic this project has got wrong three times in one session.
        """
        studies = table["study"].nunique()
        lines = [f"{studies} English studies of {corpus} in the corpus ({studies / corpus:.1%}), both meniscus sides"]
        for bucket in BUCKETS:
            rows = table[table["bucket"] == bucket]
            if rows.empty:
                continue
            positives = int(rows["positive"].sum())
            lines.append(f"  {bucket:36s} {len(rows):4d} sides   our target positive {positives:4d}")
        wrong = self.contradicted(table)
        lines.append(
            f"sides the stated rule calls negative and our target calls positive: {len(wrong)}"
            f" — {len(wrong) / (corpus * len(MENISCUS)):.2%} of the corpus's {corpus * len(MENISCUS)} meniscus cells,"
            f" and a FLOOR, since the {1 - studies / corpus:.0%} this tool cannot read carry the same wording"
        )
        return lines

    def contradicted(self, table: pd.DataFrame) -> pd.DataFrame:
        """The rows the host's rule says are mislabelled: degeneration without a tear, our target positive.

        Separate from `report` because these are the cells a person should open, and because a re-read
        under a corrected prompt is scored by whether it moves exactly these.
        """
        return table[(table["bucket"] == DEGENERATE_ONLY) & table["positive"]]

    def annotated(self, table: pd.DataFrame, truth: pd.DataFrame) -> list[str]:
        """The same buckets read against the 58 radiologist annotations — a check on the RULE, not on us.

        Small n by construction, so it settles nothing on its own. But if the annotation agrees with the
        stated rule where both exist, the rule is worth re-reading the whole corpus under; if it does not,
        the defect is in the ruler and no better reader recovers it.
        """
        graded = truth.dropna(subset=TARGETS, how="all")
        cells = self._cells(graded)
        rows = table[table["study"].isin(cells[MENISCUS[0]])].copy()
        rows["truth"] = [cells[str(row.side)][str(row.study)] > MIDPOINT for row in rows.itertuples()]
        lines = [f"{rows['study'].nunique()} of the {len(graded)} annotated studies are English:"]
        for bucket in BUCKETS:
            seen = rows[rows["bucket"] == bucket]
            if seen.empty:
                continue
            lines.append(
                f"  {bucket:36s} {len(seen):3d} sides   annotated positive {int(seen['truth'].sum()):3d}"
                f"   our target positive {int(seen['positive'].sum()):3d}"
            )
        return lines


def main() -> None:
    """Count the meniscus targets the host's stated rubric contradicts, over the corpus that produced them."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, help="the derived labels table; defaults to report_labels_v2")
    arguments = command.parsed()

    root = command.rooted(arguments)
    truth = pd.read_csv(root.raw() / "train.csv")
    labels = pd.read_csv(arguments.labels or root.processed() / "labels" / "report_labels_v2.csv")

    rubric = TearRubric()
    table = rubric.table(truth[[STUDY, "Report"]], labels)
    command.report(rubric.report(table, len(truth)))
    command.report(rubric.annotated(table, truth))
    # The identifiers, because a count of contradicted cells is a claim and the studies are the evidence.
    command.report(
        ["cells the stated rule contradicts:"]
        + [f"  {row.study}  {row.side}" for row in rubric.contradicted(table).head(MENTIONED).itertuples()]
    )


if __name__ == "__main__":
    main()
