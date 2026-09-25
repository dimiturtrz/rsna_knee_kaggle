"""Are the reader's CONFIDENT false positives extraction failures, or real reader-vs-image disagreement?

Run as `python -m core.data.labels.overcall_audit`.

THE QUESTION AND WHY IT DECIDES A LANE. `reader_agreement.errors` measures the label reader's dominant
defect: 108 confident false positives over the annotated studies, all verdict YES, against 32 misses in
the other direction, and `supervision_gap` prices the same thing as 65.3% agreement where the report says
YES. Every label mechanism this project ships treats that as READER error and responds by SOFTENING. If a
material share is instead the extractor mis-reading text the radiologist and the report actually agree
about, the fix is the extractor over all 4,407 studies — upstream of every member ever trained here.

WHAT THIS ADDS TO WORK ALREADY DONE, because most of the question is not new and a rediscovery would be
worse than nothing:

  * `interpretations/kneemri/2026-08-13_what-the-labels-actually-contain.md` measured NEGATION on
    report-YES cells and found it real and small: negated cells agree 42.3% against 73.9% unnegated
    (z = 3.00), but only 1,057 of 16,129 positives — 6.6% — are in that state, on an ENGLISH detector
    over 44.1% located cells.
  * `rubric_gap` and the `2026-08-27_the-rubric-our-reader-was-never-given.md` interpretation measured
    the SUBTHRESHOLD story per finding and BACKED it for meniscus, ACL and Baker's while REFUSING it for
    MCL and the whole SIZE family: pooled, 10 of 23 annotated subthreshold cells under a size rule are
    annotated POSITIVE.
  * `severity_boundary` and `2026-08-25_effusion-threshold-ruler.md` went further on Effusion and
    retired "read the report better" for it outright — the size qualifier is not even MONOTONE in the
    gold classes (small 50% positive, moderate-or-larger 67%, unqualified mention 86%).
  * `2026-08-20_label_contest_pilot.md` is the one manual per-cell audit: 1,200 cells re-read by an Opus
    subagent in their original languages, 92.9% agreed, and of 48 contested cells only SIX are in the
    over-call direction. Its summary is that "the reader fabricates DENIALS, never positives" — a strong
    prior AGAINST the extractor-failure reading of the 108.

None of those classified THE 108 THEMSELVES, and every text instrument above is English-only. Of the 108,
45 are English. That is what this module is for: the same question asked of the disputed cells directly,
in all eight languages the corpus is written in, with the control that makes it a measurement.

THE CONTROL IS THE POINT. Every category here can fire on a report merely because of how radiologists
write, so the identical pipeline is run over the reader's confident TRUE positives — cells where the
radiologist AGREED. A cue that fires as often on those is detecting STYLE, not error, and the report
prints the two rates side by side with their own denominators so no share can be quoted without one.

STATE THE INSTRUMENT'S CLASS. This is regex over reconstructed spans. The extraction recorded no offset
and no evidence sentence (`report_labels_v2.csv` carries value, verdict and a confidence that is a
deterministic function of the verdict — nothing else), so the span a cue is scoped to is this module's
GUESS at the one the reader used, not the span the reader used. `limits` prints what the rules cannot
see, and every classified cell is written out with its span so the calls can be checked by hand rather
than trusted.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core import cli
from core.data.labels.finding_mentions import FindingMentions
from core.data.labels.negation_scope import NegationScope
from core.data.labels.reader_agreement import MIDPOINT, POSITIVE, ReaderAgreement
from core.data.labels.report_language import ReportLanguages
from core.paths import DataRoot

logger = logging.getLogger(__name__)

MISSED_NEGATION = "missed_negation"
LATERALITY = "laterality"
HEDGE_DROPPED = "hedge_dropped"
HISTORICAL = "historical"
SUBTHRESHOLD = "subthreshold"
GENUINE = "genuine"
UNLOCATED = "unlocated"
"""The seven outcomes a cell can take. Six are the bead's categories plus `subthreshold`, which the
rubric work above showed is a distinct mechanism from all five; `UNLOCATED` is the residue and is NEVER
folded into `genuine` — a cell whose finding this vocabulary cannot find in the report is a fact about
the vocabulary, and counting it as a plain assertion would put the tool's blind spot in the tally."""

PRECEDENCE = (MISSED_NEGATION, HISTORICAL, HEDGE_DROPPED, SUBTHRESHOLD)
"""Which cue wins when a span carries several, most-decisive first.

A CHOICE, not a fact, and it moves the tally: "status post ACL repair, possible re-tear" carries three.
Negation leads because it flips the claim outright rather than qualifying it; `subthreshold` trails
because a report can qualify severity while still plainly asserting the finding. Every cue that fired is
written to the artifact beside the chosen category, so a reader can re-derive the tally under a
different order instead of taking this one on trust.
"""

CATEGORIES = (MISSED_NEGATION, LATERALITY, HEDGE_DROPPED, HISTORICAL, SUBTHRESHOLD, GENUINE, UNLOCATED)

RESAMPLES = 2000
SEED = 2026
INTERVAL = (2.5, 97.5)
"""Bootstrap settings, matching `noise_binding` so two readings of these same 108 cells are comparable."""


HEDGE: dict[str, tuple[str, ...]] = {
    "en": (
        r"possib",
        r"suspect",
        r"cannot\s+(be\s+)?exclude",
        r"can\s*not\s+exclude",
        r"equivocal",
        r"indetermin",
        r"may\s+repres",
        r"question",
        r"differential",
        r"\bDD\b",
        r"\bversus\b",
        r"\bvs\b",
        r"probabl",
        r"likely",
        r"appears?\s+to\s+repres",
        r"concerning\s+for",
    ),
    "es": (
        r"posib",
        r"probabl",
        r"sugestiv",
        r"sospech",
        r"dudos",
        r"impresiona",
        r"no\s+se\s+puede\s+excluir",
        r"puede\s+corresponder",
        r"indetermin",
    ),
    "tr": (
        r"olasılık",
        r"muhtemel",
        r"şüphe",
        r"olabilir",
        r"düşündür",
        r"ayırıcı",
        r"lehine",
        r"uyumlu\s+olabil",
        r"ekarte\s+edilemez",
    ),
    "hr": (
        r"moguć",
        r"vjerojatn",
        r"može\s+odgovarati",
        r"sumnj",
        r"suspektn",
        r"ne\s+može\s+se\s+isključiti",
        r"\bDD\b",
        r"vjerojatno",
    ),
    "de": (
        r"möglich",
        r"fraglich",
        r"verdächtig",
        r"\bDD\b",
        r"\bwohl\b",
        r"am\s+ehesten",
        r"nicht\s+sicher",
        r"vermutlich",
        r"\bV\.?a\.?\b",
        r"suspekt",
    ),
    "bg": (r"възможн", r"вероятн", r"съмнени", r"суспектн", r"диференциал", r"не\s+може\s+да\s+се\s+изключи"),
    "el": (r"πιθαν", r"ενδεχοµέν", r"ενδεχομέν", r"ύποπτ", r"δεν\s+αποκλείεται", r"µπορούσε", r"μπορούσε"),
    "nl": (r"mogelijk", r"verdacht", r"waarschijnlijk", r"\bDD\b", r"vermoedelijk", r"suspect"),
}
"""Words a report uses when it is NOT sure. English-only vocabulary existed before this module
(`report_language.HEDGES`) and was deliberately kept English-only there because the flag came back
BACKWARDS as a supervision weight. It is written out for all eight here because the question is
different: not "does hedging predict error" but "did the extractor score a hedged sentence confident"."""

HISTORY: dict[str, tuple[str, ...]] = {
    "en": (
        r"status\s+post",
        r"\bs/p\b",
        r"\bprior\b",
        r"previous",
        r"history\s+of",
        r"\bremote\b",
        r"chronic",
        r"postoperative",
        r"post[- ]?surgical",
        r"\bold\b",
        r"sequela",
        r"known\b",
    ),
    "es": (
        r"antecedent",
        r"\bprevi",
        r"postquirúrg",
        r"crónic",
        r"secuela",
        r"\bantigu",
        r"intervenid",
        r"historia\s+de",
    ),
    "tr": (r"öykü", r"geçirilmiş", r"\beski\b", r"kronik", r"opere", r"postoperatif", r"sekel", r"bilinen", r"anamnez"),
    "hr": (
        r"stanje\s+nakon",
        r"ranij",
        r"prijašnj",
        r"kronič",
        r"postoperativ",
        r"\bstar",
        r"\bstatus\s+post",
        r"anamnest",
    ),
    "de": (
        r"anamnestisch",
        r"\bbek\.?\b",
        r"bekannt",
        r"\bZ\.?n\.?\b",
        r"zustand\s+nach",
        r"chronisch",
        r"\balt\b",
        r"vorbekannt",
        r"vorschädigung",
        r"postoperativ",
        r"voroperiert",
    ),
    "bg": (r"анамнез", r"\bстар", r"хронич", r"състояние\s+след", r"опериран", r"предшеств"),
    "el": (r"ιστορικό", r"παλαι", r"χρόνι", r"µετεγχειρητικ", r"μετεγχειρητικ", r"προηγούµεν", r"προηγούμεν"),
    "nl": (r"status\s+na", r"voorgeschiedenis", r"\boud\b", r"chronisch", r"postoperatief", r"\bbekend", r"eerder"),
}
"""References to a PRIOR study, an operation, or clinical history rather than a current image finding."""

MINIMISER: dict[str, tuple[str, ...]] = {
    "en": (
        r"\bmild",
        r"\bminimal",
        r"\bsmall\b",
        r"\bminor\b",
        r"\bslight",
        r"\btrace\b",
        r"\btiny\b",
        r"\bsubtle",
        r"low[- ]grade",
        r"grade\s*[i1]\b",
        r"grade\s*(ii|2)\b",
        r"degenerative\s+signal",
        r"intrasubstance",
        r"without\s+surfacing",
        r"\bfocal\b",
        r"early\b",
    ),
    "es": (
        r"\bleve",
        r"mínim",
        r"minim",
        r"discret",
        r"pequeñ",
        r"\bligero",
        r"grado\s*[i1]\b",
        r"grado\s*(ii|2)\b",
        r"incipiente",
        r"\bfin[ao]\b",
        r"escas",
    ),
    "tr": (
        r"\bhafif",
        r"minimal",
        r"küçük",
        r"grade?\s*[i1]\b",
        r"grade?\s*(ii|2)\b",
        r"derece\s*[i1]\b",
        r"\baz\b",
        r"hafif\s+derece",
        r"silik",
        r"başlangıç",
    ),
    "hr": (
        r"\bblag",
        r"minimaln",
        r"\bmanj",
        r"\bmal[aeoi]\b",
        r"[i1]\.\s*stupnj",
        r"(ii|2)\.\s*stupnj",
        r"\bI°",
        r"\bII°",
        r"početn",
        r"diskret",
        r"neznatn",
    ),
    "de": (
        r"\bgering",
        r"\bklein",
        r"minimal",
        r"\bleicht",
        r"\bzart",
        r"\bkl\.",
        r"initial",
        r"grad\s*[i1]\b",
        r"grad\s*(ii|2)\b",
        r"diskret",
        r"\bmäßig\s+gering",
    ),
    "bg": (r"минимал", r"\bлек", r"лекостепен", r"\bмалк", r"начал", r"слабо", r"незначител"),
    "el": (
        r"ήπι",
        r"\bµικρ",
        r"\bμικρ",
        r"ελαφρ",
        r"αρχόµεν",
        r"αρχόμεν",
        r"\bµόλις",
        r"\bμόλις",
        r"grade\s*(ii|2)\b",
        r"απλή\s+εκφύλιση",
    ),
    "nl": (r"\blicht", r"\bklein", r"minimaal", r"\bgering", r"minstens", r"beginnende", r"subtiel"),
}
"""Severity words BELOW the annotation rubric's bar, which `core.findings.CRITERIA` states explicitly:
ACL wants "more than 50 percent of fibers disrupted", MCL excludes "low-grade sprains", the meniscus
excludes "intrasubstance degeneration that does not reach the surface", the OA columns want ">50 percent
of cartilage thickness" over "roughly 1 cm or greater", and Effusion wants "a moderate or large amount".

DELIBERATELY EXCLUDES `moderate`/`mäßig`/`orta`/`umjeren`/`moderad`/`matig`/`умерен`/`µέτρι`, which sit AT
the bar for Effusion rather than below it.

READ THIS CATEGORY AGAINST THE PRIOR ART, NOT ALONE. `severity_boundary` measured the size qualifier
against the gold classes for Effusion and found it does not separate them and is not monotone in them
(small 50% positive, moderate-or-larger 67%, unqualified 86%), and `rubric_gap` REFUSED the subthreshold
reading for MCL and the whole size family on annotated cells. So a large `subthreshold` count here is
NOT on its own evidence that fixing the extractor's severity handling would fix the label — for several
of these findings that specific claim has already been tested and failed."""

CUES: dict[str, dict[str, tuple[str, ...]]] = {
    HISTORICAL: HISTORY,
    HEDGE_DROPPED: HEDGE,
    SUBTHRESHOLD: MINIMISER,
}
"""The categories decided by a cue appearing ANYWHERE in the span. `MISSED_NEGATION` is deliberately not
among them: it is the one category whose correctness depends on WHAT the cue scopes over, so it is
decided by `OvercallAudit.denied` instead."""


ARTIFACT = "overcall_audit.csv"
"""Every classified cell with its span, so the calls can be reviewed rather than only tallied."""


@dataclass(frozen=True)
class Judgement:
    """One cell's category, the span it was read from, and every cue that fired in that span."""

    category: str
    span: str
    cues: tuple[str, ...]

    def row(self) -> dict[str, str]:
        """The judgement as artifact columns, cues flattened so the CSV stays one row per cell."""
        return {"category": self.category, "cues": " ".join(self.cues), "span": self.span}


class OvercallAudit:
    """Why the reader called a finding present where the radiologist called it absent."""

    def __init__(
        self,
        mentions: FindingMentions | None = None,
        languages: ReportLanguages | None = None,
        resamples: int = RESAMPLES,
    ) -> None:
        self.mentions = mentions or FindingMentions()
        self.languages = languages or ReportLanguages()
        self.resamples = resamples
        self.cues = {
            category: {
                language: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
                for language, patterns in vocabulary.items()
            }
            for category, vocabulary in CUES.items()
        }
        self.scope = NegationScope(self.mentions)

    def fired(self, span: str, language: str, finding: str) -> tuple[str, ...]:
        """Every cue category this span carries for this finding, in `PRECEDENCE` order.

        Returns all of them rather than the first, because a span carrying three cues is a DIFFERENT
        object from one carrying a single unambiguous negation, and a tally that only ever saw the
        winner could not show the difference.
        """
        carried = {
            category
            for category, vocabulary in self.cues.items()
            if any(pattern.search(span) for pattern in vocabulary.get(language, []))
        }
        if self.scope.denied(span, language, finding):
            carried.add(MISSED_NEGATION)
        return tuple(category for category in PRECEDENCE if category in carried)

    def classify(self, report: str, language: str, finding: str) -> Judgement:
        """The category for one (report, finding) pair, with the span the call was read from.

        A finding named nowhere in the report is `UNLOCATED` unless its COMPARTMENT PARTNER is named and
        it is not, which is `LATERALITY`. Where the finding IS named, the FIRST span carrying any cue
        decides; a mention with no cue anywhere is `GENUINE` — the report plainly asserts the finding as
        present and current, so the disagreement is with the IMAGE rather than with the text.
        """
        spans = self.mentions.mentions(report, language, finding)
        if not spans:
            partner = self.mentions.elsewhere(report, language, finding)
            if partner:
                return Judgement(LATERALITY, partner[0], ())
            return Judgement(UNLOCATED, "", ())
        for span in spans:
            cues = self.fired(span, language, finding)
            if cues:
                return Judgement(cues[0], span, cues)
        return Judgement(GENUINE, spans[0], ())

    def audit(self, cells: pd.DataFrame, reports: dict[str, str]) -> pd.DataFrame:
        """Classify every cell of the given frame, carrying the study, finding and language through.

        Takes the cells rather than reading them so the SAME method runs over the false positives and
        over the true-positive control — a control classified by a second code path would not be one.
        """
        rows = []
        for _, cell in cells.iterrows():
            report = reports[str(cell["study"])]
            judgement = self.classify(report, str(cell["language"]), str(cell["target"]))
            rows.append(
                {
                    "study": cell["study"],
                    "target": cell["target"],
                    "language": cell["language"],
                    **judgement.row(),
                }
            )
        return pd.DataFrame(rows, columns=["study", "target", "language", "category", "cues", "span"])

    def census(self, audited: pd.DataFrame) -> dict[str, int]:
        """How many cells fell in each category, every category present even at zero.

        Zeroes are kept so the seven categories always print in the same order with the same names: a
        table whose rows appear and vanish between runs cannot be compared against another run of it,
        and the control comparison is exactly that.
        """
        counts = audited["category"].value_counts().to_dict()
        return {category: int(counts.get(category, 0)) for category in CATEGORIES}

    def compare(self, over: pd.DataFrame, agreed: pd.DataFrame) -> list[str]:
        """The category split for the disputed cells beside the SAME split for the agreed ones.

        THE CONTROL, and the reason this module is a measurement rather than a tally. Each rate carries
        its own denominator on the same line, and the last column is the difference. A category that
        fires as often where the radiologist AGREED is detecting how radiologists WRITE, and a raw count
        of it says nothing about extraction at all.
        """
        disputed, control = self.census(over), self.census(agreed)
        wide, tall = len(over), len(agreed)
        lines = [
            f"{'category':<16} {'false positive':>18} {'true positive':>18}   difference [95% study-clustered]",
            f"{'':16} {'(n=' + str(wide) + ')':>18} {'(n=' + str(tall) + ')':>18}",
        ]
        for category in CATEGORIES:
            first = disputed[category] / wide if wide else float("nan")
            second = control[category] / tall if tall else float("nan")
            low, high = self.interval(over, agreed, category)
            lines.append(
                f"{category:<16} {disputed[category]:>6} {first:>10.1%} {control[category]:>6} "
                f"{second:>10.1%}   {first - second:+6.1%} [{low:+.1%}, {high:+.1%}]"
            )
        return lines

    def interval(self, over: pd.DataFrame, agreed: pd.DataFrame, category: str) -> tuple[float, float]:
        """The false-positive-minus-true-positive difference's interval, resampling STUDIES.

        Cells within a study share one report, one radiologist and one writing style, so they are not
        independent draws and a cell-level interval would be too narrow — the same discipline
        `reader_agreement` and `noise_binding` adopt, for the same reason. Studies are drawn once and
        BOTH rates recomputed from the same draw, because the two groups overlap in studies and
        resampling them separately would break that pairing.
        """
        generator = np.random.default_rng(SEED)
        pooled = pd.concat([over.assign(side=0), agreed.assign(side=1)])
        studies = pooled["study"].unique()
        by_study = {study: frame for study, frame in pooled.groupby("study")}
        drawn = []
        for _ in range(self.resamples):
            picked = generator.choice(studies, size=len(studies), replace=True)
            sample = pd.concat([by_study[study] for study in picked])
            first, second = sample[sample["side"] == 0], sample[sample["side"] == 1]
            if first.empty or second.empty:
                continue
            drawn.append(first["category"].eq(category).mean() - second["category"].eq(category).mean())
        if not drawn:
            return (float("nan"), float("nan"))
        low, high = np.percentile(np.array(drawn), INTERVAL)
        return (float(low), float(high))

    def incidence(self, over: pd.DataFrame, agreed: pd.DataFrame) -> list[str]:
        """How often each cue fired at all, IGNORING `PRECEDENCE`, on both groups.

        The table above assigns every cell to one category, so a span carrying a hedge AND a minimiser
        is invisible in whichever lost. This one counts a cue wherever it appears, so the four rates sum
        past 100% by design, and it is the check on whether a category's small count is a real absence
        or an artefact of the order chosen in `PRECEDENCE`.
        """
        lines = [
            f"{'cue':<16} {'false positive':>18} {'true positive':>18}   (precedence ignored; a cell may carry several)"
        ]
        for category in PRECEDENCE:
            fired = over["cues"].fillna("").str.contains(category)
            also = agreed["cues"].fillna("").str.contains(category)
            first = fired.mean() if len(over) else float("nan")
            second = also.mean() if len(agreed) else float("nan")
            lines.append(f"{category:<16} {int(fired.sum()):>6} {first:>10.1%} {int(also.sum()):>6} {second:>10.1%}")
        return lines

    def coverage(self, audited: pd.DataFrame) -> list[str]:
        """What fraction of cells the vocabulary could locate, as both numbers and never as a rate alone.

        Printed before any category rate because every one of those is computed INSIDE the located
        subset, and this project has three times read a rate over a parsed subset as a rate over the
        corpus. The unlocated count is the size of the hole in every line that follows.
        """
        located = int((audited["category"] != UNLOCATED).sum())
        total = len(audited)
        lines = [f"located the finding in {located} of {total} cells ({located / total:.1%})" if total else "no cells"]
        missing = audited[audited["category"] == UNLOCATED]
        if not missing.empty:
            languages = missing["language"].value_counts().items()
            findings = missing["target"].value_counts().head(4).items()
            by_language = ", ".join(f"{language} {count}" for language, count in languages)
            by_finding = ", ".join(f"{target} {count}" for target, count in findings)
            lines.append(f"  unlocated by language: {by_language}")
            lines.append(f"  unlocated by finding:  {by_finding}")
        return lines

    @staticmethod
    def limits() -> list[str]:
        """What these rules CANNOT see, printed with the result rather than kept in a docstring.

        A regex negation detector auditing negation errors is a weak instrument measuring its own
        failure mode, and the honest response is to say where it is blind in the same output as the
        number, so the two are never separated by a copy-paste.
        """
        return [
            "what these rules would MISS:",
            "  * SCOPE, PARTLY HANDLED AND STILL THE WEAKEST JOINT. A negation now counts only where the",
            "    text it governs names this finding's anatomy or lesion, which is what stops 'without",
            "    extrusion' reading as a denied meniscus tear. It is still not a parse: 'ACL intact,",
            "    medial meniscus torn' is one span, and a denial governing a LATER clause about another",
            "    finding is credited here. The 2026-08-13 interpretation hit the same defect.",
            "  * LOCATION. Where several sentences name a finding, the first carrying any cue decides. A",
            "    report that denies a finding in one line and asserts it in the next is scored on the",
            "    denial. This is the residue left in the control and it is NOT removable by vocabulary.",
            "  * RECONSTRUCTION. The reader recorded no offset or evidence sentence, so the span scored",
            "    here is this module's GUESS at the one it used, not the one it used.",
            "  * VOCABULARY. Eight languages hand-built from reading the annotated reports; a negation",
            "    phrased in a way no pattern covers reads as a plain assertion and lands in `genuine`,",
            "    which biases the tally TOWARD the reader-disagreement answer.",
            "  * PRECEDENCE. A span with several cues is counted once, under `PRECEDENCE`; the `cues`",
            "    column of the artifact holds the rest so a different order can be re-derived.",
            "  * SUBTHRESHOLD IS NOT A VERDICT. `severity_boundary` measured the size qualifier against",
            "    the gold classes for Effusion and found it neither separating nor monotone, and",
            "    `rubric_gap` refused the subthreshold reading for MCL and the size family.",
        ]

    @staticmethod
    def _cells(root: DataRoot, table: Path) -> tuple[pd.DataFrame, dict[str, str]]:
        """The scored cells and the report text they were read from, joined on the study."""
        agreement = ReaderAgreement()
        annotated = agreement.annotated(root)
        read = pd.read_csv(table)
        cells = agreement.cells(annotated, read)
        text = zip(annotated["StudyInstanceUID"], annotated["Report"], strict=True)
        reports = {str(key): str(value) for key, value in text}
        return cells, reports


def main() -> None:
    """Classify the reader's confident false positives, with its confident true positives as control."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, default=None, help="read table; default report_labels_v2")
    command.parser.add_argument("--out", type=Path, default=None, help=f"where to write {ARTIFACT}")
    arguments = command.parsed()

    root = command.rooted(arguments)
    table = arguments.labels or (root.labels())
    audit = OvercallAudit()
    cells, reports = audit._cells(root, table)  # noqa: SLF001 - the module's own entrypoint

    called = cells[cells["verdict"] == POSITIVE]
    over = called[called["truth"] <= MIDPOINT]
    agreed = called[called["truth"] > MIDPOINT]

    audited_over, audited_agreed = audit.audit(over, reports), audit.audit(agreed, reports)

    command.report(
        [
            f"{len(cells)} annotated cells over {cells['study'].nunique()} studies; the reader called "
            f"{len(called)} positive, the radiologist agreed on {len(agreed)} and refused {len(over)}",
            f"the {len(over)} refused cells span {over['study'].nunique()} studies and "
            f"{over['language'].nunique()} languages ({(over['language'] == 'en').sum()} of them English)",
            "",
            *audit.coverage(audited_over),
            "",
            *audit.compare(audited_over, audited_agreed),
            "",
            *audit.incidence(audited_over, audited_agreed),
            "",
            *audit.limits(),
        ]
    )

    destination = arguments.out or (root.processed() / "labels" / ARTIFACT)
    audited_over.assign(group="false positive").pipe(
        lambda frame: pd.concat([frame, audited_agreed.assign(group="true positive")])
    ).to_csv(destination, index=False)
    logger.info("")
    logger.info("every classified cell with its span: %s", destination)


if __name__ == "__main__":
    main()
