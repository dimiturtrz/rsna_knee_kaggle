"""Does the 58-study ruler obey its own severity rule? `python -m core.data.labels.severity_boundary`.

EVERY LABEL-COLUMN NUMBER THIS PROJECT HOLDS IS SCORED AGAINST THOSE 58 ANNOTATIONS.
`kneemri.evaluation.supervision_gap` prices each finding twice against them — labels 0.8658 against
model 0.8392 in aggregate, Effusion 0.8149 against 0.9453 — and `reader_agreement` accepts or rejects a replacement
reader by them. If the annotation itself does not follow the rule the competition states, then part of
the reader's measured error is not error, and no better reader recovers it.

The rule is a THRESHOLD: the competition's introduction grades an effusion positive at moderate or
above. So the check is one cross-tab. Bucket each annotated report by the size qualifier in its own
effusion sentence, and read the annotation beside it. A bucket holding BOTH classes means the boundary
is where two radiologists would disagree, which is exactly what two competitors report seeing
(discussion topic 735826, read 2026-08-25) — one of them finding the same English template and the same
sentence annotated both ways.

THIS IS A CHECK ON THE RULER, NOT ON THE READER. It asks nothing about our LLM's verdicts; it reads
the report text and the radiologist's annotation only, so its answer is independent of every label
pipeline we have built.

ENGLISH ONLY, AND THE DENOMINATOR SAYS SO. 59.9% of the corpus is not English (`report_language`) and
inventing a size vocabulary for seven more languages to answer a yes/no existence question would put
the ruler's own defect inside the tool measuring it. The report states how many of the 58 it could read
and how many it could not — a rate inside the subset that happened to parse is not a rate over the 58.
"""

import logging
import re
from pathlib import Path

import pandas as pd

from core import cli
from core.data.labels.reader_agreement import ENGLISH, MIDPOINT
from core.data.labels.report_language import ReportLanguages
from core.findings import TARGETS

logger = logging.getLogger(__name__)

FINDING = "Effusion"
"""The one finding the competition grades by a size word the report almost always carries."""

MENTIONS: tuple[str, ...] = (r"effusion", r"joint\s+fluid", r"suprapatellar\s+fluid")
SMALL_WORDS: tuple[str, ...] = (r"\bsmall", r"\bminimal", r"\btrace", r"\btiny", r"\bscant", r"\bmild")
LARGE_WORDS: tuple[str, ...] = (r"\bmoderate", r"\blarge", r"\bmarked", r"\bsignificant", r"\bgross")
"""The two sides of the stated threshold. `mild` sits with small because the rule names moderate as the
first positive grade, so anything below it is meant to read as negative."""

BREAK = re.compile(r"[.;\n]")

DENIED = re.compile(r"\b(no|without|absent|negative\s+for)\b", re.IGNORECASE)
"""A denial anywhere in the effusion sentence takes it out of the cross-tab entirely.

'No significant knee effusion' carries `significant`, which is a LARGE word, so a size regex reads a
denial as a moderate effusion — and beside its Effusion=0 that is a contradiction the tool INVENTED. It
was one of the eight this module first reported, found by printing the sentences rather than the counts.
Sentence-scoped like everything else here, so a denial of one finding cannot silence another's."""

SILENT = "no effusion sentence"
DENIAL = "effusion denied"
UNQUALIFIED = "mentioned, no size word"
SMALL = "small / minimal / trace"
LARGE = "moderate or larger"
MIXED = "both, e.g. small to moderate"
BUCKETS: tuple[str, ...] = (SMALL, MIXED, LARGE, UNQUALIFIED, DENIAL, SILENT)
"""Ordered as the rule reads them, so the two rows that must not both hold positives are adjacent."""


class SeverityBoundary:
    """Each annotated report's effusion size word, read against the radiologist's own annotation."""

    def __init__(self, languages: ReportLanguages | None = None) -> None:
        self.languages = languages or ReportLanguages()
        self.mentions = [re.compile(pattern, re.IGNORECASE) for pattern in MENTIONS]
        self.small = [re.compile(pattern, re.IGNORECASE) for pattern in SMALL_WORDS]
        self.large = [re.compile(pattern, re.IGNORECASE) for pattern in LARGE_WORDS]

    def sentences(self, report: str) -> list[str]:
        """The report's effusion sentences, and only those.

        Sentence-scoped rather than report-scoped because a knee report grades several findings in one
        paragraph: 'moderate medial compartment osteoarthritis. small joint effusion.' carries both size
        words, and a report-wide regex would read that as a moderate effusion.
        """
        return [part for part in BREAK.split(report) if any(word.search(part) for word in self.mentions)]

    def bucket(self, report: str) -> str:
        """Which side of the stated threshold this report's effusion sentence puts it on."""
        seen = self.sentences(report)
        if not seen:
            return SILENT
        found = " ".join(part for part in seen if not DENIED.search(part))
        if not found:
            return DENIAL
        small = any(word.search(found) for word in self.small)
        large = any(word.search(found) for word in self.large)
        if small and large:
            return MIXED
        if small:
            return SMALL
        return LARGE if large else UNQUALIFIED

    def table(self, annotated: pd.DataFrame) -> pd.DataFrame:
        """One row per annotated English study: its bucket and the radiologist's Effusion annotation."""
        rows = []
        for _, study in annotated.iterrows():
            report = str(study["Report"])
            if self.languages.language(report) != ENGLISH or pd.isna(study.get(FINDING)):
                continue
            rows.append((study["StudyInstanceUID"], self.bucket(report), float(study[FINDING]) > MIDPOINT))
        return pd.DataFrame(rows, columns=["study", "bucket", "positive"])

    def report(self, table: pd.DataFrame, annotated: int) -> list[str]:
        """The cross-tab, with the denominator it was computed inside stated on the first line.

        A bucket that holds both classes is the finding. A bucket holding one class says nothing on its
        own at these counts — 58 studies annotated, a minority of them English — so every row carries its
        own n and the verdict line names how many studies it rests on.
        """
        lines = [f"{len(table)} English annotated studies of {annotated} annotated ({FINDING} read from the report)"]
        mixed = []
        for bucket in BUCKETS:
            rows = table[table["bucket"] == bucket]
            if rows.empty:
                continue
            positives = int(rows["positive"].sum())
            lines.append(f"  {bucket:32s} {len(rows):3d} studies   annotated positive {positives:3d}")
            if 0 < positives < len(rows):
                mixed.append(f"{bucket} ({positives} of {len(rows)})")
        lines.append(
            f"buckets holding BOTH classes: {', '.join(mixed) if mixed else 'none'}"
            + ("" if mixed else " — on this subset the annotation follows the stated threshold")
        )
        return lines

    def contradictions(self, table: pd.DataFrame) -> list[str]:
        """The studies the stated rule gets WRONG: a small effusion annotated 1, a moderate one annotated 0.

        Separate from `report` because these are the rows a person should open. The count is the claim;
        the identifiers are what makes it checkable rather than a rate to be quoted at second hand.
        """
        small = table[(table["bucket"] == SMALL) & table["positive"]]
        large = table[(table["bucket"] == LARGE) & ~table["positive"]]
        found = [(study, f"{SMALL}, annotated 1") for study in small["study"]]
        found += [(study, f"{LARGE}, annotated 0") for study in large["study"]]
        return [f"  {study}  {why}" for study, why in found]


def main() -> None:
    """Cross-tab the annotated studies' effusion size word against the radiologist's own annotation."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--truth", type=Path, help="the annotated table; defaults to raw train.csv")
    arguments = command.parsed()

    root = command.rooted(arguments)
    truth = pd.read_csv(arguments.truth or root.raw() / "train.csv")
    annotated = truth.dropna(subset=TARGETS, how="all")[["StudyInstanceUID", "Report", *TARGETS]]

    boundary = SeverityBoundary()
    table = boundary.table(annotated)
    command.report(boundary.report(table, len(annotated)))
    # The identifiers, because a count of contradictions is a claim and the studies are the evidence.
    command.report(["studies the stated threshold gets wrong:", *boundary.contradictions(table)])


if __name__ == "__main__":
    main()
