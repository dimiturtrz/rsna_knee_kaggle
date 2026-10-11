"""Which sentences of a report talk about a given finding, in the report's own language.

Run as part of `core.data.labels.overcall_audit`; this module is the LOCATOR, not the judgement.

WHY IT EXISTS SEPARATELY FROM THE AUDIT. The reader that produced `report_labels_v2.csv` recorded a
value, a verdict and a confidence per (study, finding) and NOTHING ELSE — no offset, no sentence, no
quoted evidence. So an audit of what the reader got wrong cannot pull up the span the label came from;
it has to RECONSTRUCT a candidate span from the text. That reconstruction is a separate, reusable and
separately-wrong thing from the classification applied to it, and collapsing the two would hide which of
them a bad call came from.

STATE THE INSTRUMENT'S CLASS PLAINLY: this is vocabulary matching, not parsing. A sentence is called a
mention of ACL when an ACL term appears in it. It cannot resolve which of two findings a shared clause
attaches to, and it does not know grammar in any of the eight languages it covers.

THE VOCABULARY IS MULTILINGUAL BECAUSE THE CORPUS IS, and this is the exact blind spot
`report_language` was written to correct. English is 40% of the corpus; of the 108 confident false
positives this audit exists to explain, only 45 are English. An English-only matcher would report a
rate over the 42% it could read and present it as a rate over the whole — which is the failure
`report_language`'s docstring records ("the 16.6% was a property of the tool").
"""

import logging
import re

from core.data.labels.report_language import Language
from core.findings import COMPARTMENT_PAIRS, TARGETS, Finding

logger = logging.getLogger(__name__)

WINDOW = 240
"""Characters kept either side of a term when a report has no usable sentence punctuation.

Turkish and Bulgarian reports in this corpus routinely run a whole examination into one unpunctuated
block, and several close sentences without a following space (`izlenmiştir.Arka`). A paragraph-long
"sentence" would put every cue in the report inside every finding's span and score the whole report at
once, so the span is capped rather than trusted to punctuation that is not there.
"""

MINIMUM_SENTENCE = 3
"""Characters below which a split fragment is dropped rather than kept as a sentence."""

BOUNDARY = re.compile(r"(?<=[.;:!?\n])")
"""Split AFTER terminal punctuation, keeping it. Deliberately not `\\s`-anchored: see `WINDOW`."""

FINDING_TERMS: dict[str, dict[str, tuple[str, ...]]] = {
    Finding.ACL: {
        Language.EN: (r"\bACL\b", r"anterior\s+cruciate"),
        Language.ES: (r"\bLCA\b", r"cruzado\s+anterior"),
        Language.TR: (r"ön\s+çapraz", r"öçb\b", r"anterior\s+çapraz"),
        Language.HR: (r"prednj\w*\s+križn", r"\bPKL\b", r"lig\w*\s+cruciatum\s+anterius"),
        Language.DE: (r"\bVKB\b", r"vorder\w*\s+kreuzband"),
        Language.BG: (r"предн\w*\s+кръстн",),
        Language.EL: (r"πρόσθι\w*\s+χιαστ", r"πρόσθιου\s+χιαστού"),
        Language.NL: (r"voorste\s+kruisband", r"\bVKB\b"),
    },
    Finding.MCL: {
        Language.EN: (r"\bMCL\b", r"medial\s+collateral"),
        Language.ES: (r"\bLCM\b", r"colateral\s+(medial|intern)", r"ligamento\s+lateral\s+intern"),
        Language.TR: (r"medial\s+kollateral", r"\bMKL\b", r"iç\s+yan\s+bağ"),
        Language.HR: (r"medijaln\w*\s+kolateraln", r"\bMKL\b"),
        Language.DE: (r"\bMCL\b", r"innenband", r"medial\w*\s+kollateral"),
        Language.BG: (r"медиалн\w*\s+колатерал",),
        Language.EL: (r"έσω\s+πλάγι",),
        Language.NL: (r"medial\w*\s+collateral",),
    },
    Finding.MEDIAL_MENISCUS: {
        Language.EN: (r"medial\s+meniscus", r"medial\s+menisc"),
        Language.ES: (r"menisco\s+(intern|medial)",),
        Language.TR: (r"(medial|iç)\s+menisküs", r"medyal\s+menisküs"),
        Language.HR: (r"medijaln\w*\s+menisk",),
        Language.DE: (r"innenmeniskus", r"medial\w*\s+meniskus"),
        Language.BG: (r"медиалн\w*\s+менискус",),
        Language.EL: (r"έσω\s+µηνίσκ", r"έσω\s+μηνίσκ"),
        Language.NL: (r"medial\w*\s+meniscus", r"binnenmeniscus"),
    },
    Finding.LATERAL_MENISCUS: {
        Language.EN: (r"lateral\s+meniscus", r"lateral\s+menisc"),
        Language.ES: (r"menisco\s+(extern|lateral)",),
        Language.TR: (r"(lateral|dış)\s+menisküs",),
        Language.HR: (r"lateraln\w*\s+menisk",),
        Language.DE: (r"außenmeniskus", r"aussenmeniskus", r"lateral\w*\s+meniskus"),
        Language.BG: (r"латералн\w*\s+менискус",),
        Language.EL: (r"έξω\s+µηνίσκ", r"έξω\s+μηνίσκ"),
        Language.NL: (r"lateral\w*\s+meniscus", r"buitenmeniscus"),
    },
    Finding.MEDIAL_OA: {
        Language.EN: (r"medial\s+(compartment|femoral|tibial)", r"medial\s+.{0,30}cartilage"),
        Language.ES: (
            r"(femorotibial|compartiment\w*)\s+medial",
            r"cóndilo\s+femoral\s+medial",
            r"artrosis\s+.{0,20}medial",
        ),
        Language.TR: (r"medial\s+(kompartman|femorotibial)", r"medial\s+.{0,25}kıkırdak", r"gonartroz"),
        Language.HR: (r"medijaln\w*\s+(kompartment|femorotibial)", r"medijaln\w*\s+.{0,25}hrskavic"),
        Language.DE: (r"medial\w*\s+(kompartiment|femorotibial)", r"innerer?\s+gelenkspalt"),
        Language.BG: (r"медиалн\w*\s+.{0,25}хрущял", r"медиалното\s+плато"),
        Language.EL: (r"έσω\s+διαµέρισµα", r"έσω\s+διαμέρισμα"),
        Language.NL: (r"medial\w*\s+.{0,25}kraakbeen", r"medial\w*\s+compartiment"),
    },
    Finding.LATERAL_OA: {
        Language.EN: (r"lateral\s+(compartment|femoral|tibial|trochlea)", r"lateral\s+.{0,30}cartilage"),
        Language.ES: (r"(femorotibial|compartiment\w*)\s+lateral", r"cóndilo\s+femoral\s+lateral"),
        Language.TR: (r"lateral\s+(kompartman|femorotibial)", r"lateral\s+.{0,25}kıkırdak"),
        Language.HR: (r"lateraln\w*\s+(kompartment|femorotibial)", r"lateraln\w*\s+.{0,25}hrskavic"),
        Language.DE: (r"lateral\w*\s+(kompartiment|femorotibial)", r"äußerer?\s+gelenkspalt"),
        Language.BG: (r"латералн\w*\s+.{0,25}хрущял",),
        Language.EL: (r"έξω\s+διαµέρισµα", r"έξω\s+διαμέρισμα"),
        Language.NL: (r"lateral\w*\s+.{0,25}kraakbeen", r"lateral\w*\s+compartiment"),
    },
    Finding.PF_OA: {
        Language.EN: (r"patellofemoral", r"patellar\s+(facet|cartilage)", r"trochlea"),
        Language.ES: (
            r"patelofemoral",
            r"femoropatelar",
            r"rotulian\w*\s+.{0,20}cartíl",
            r"cartílago\s+rotulian",
            r"tróclea",
        ),
        Language.TR: (r"patellofemoral", r"patellar\s+.{0,20}kıkırdak", r"troklea"),
        Language.HR: (r"patelofemoral", r"fasete\s+patele", r"patelarn\w*\s+.{0,20}hrskavic"),
        Language.DE: (r"patellofemoral", r"retropatellar", r"patella\w*\s+.{0,20}knorpel", r"trochlea"),
        Language.BG: (r"пателофеморал", r"хрущял\s+на\s+пателата"),
        Language.EL: (r"επιγονατιδοµηριαί", r"επιγονατιδομηριαί"),
        Language.NL: (r"patellofemoral", r"retropatellair", r"trochlea"),
    },
    Finding.EFFUSION: {
        Language.EN: (r"effusion", r"joint\s+fluid", r"suprapatellar\s+(recess|bursa)"),
        Language.ES: (r"derrame", r"líquido\s+articular"),
        Language.TR: (r"efüzyon", r"eklem\s+içi\s+sıvı", r"sıvı\s+(artışı|miktarı)", r"eklem\s+mesafesinde\s+sıvı"),
        Language.HR: (r"izljev", r"izliv", r"tekućin\w*\s+u\s+zglob"),
        Language.DE: (r"erguss", r"gelenkerguß", r"gelenkflüssigkeit"),
        Language.BG: (r"излив", r"ставна\s+течност"),
        Language.EL: (r"υγρού\s+ενδαρθρικά", r"συλλογή\s+υγρού", r"ενδαρθρικ\w*\s+υγρ"),
        Language.NL: (r"vocht", r"effusie", r"opzetting\s+.{0,25}recessus", r"hydrops"),
    },
    Finding.SYNOVITIS: {
        Language.EN: (r"synovit", r"synovial\s+(thickening|proliferation|hypertroph)"),
        Language.ES: (r"sinovit", r"proliferación\s+sinovial", r"engrosamiento\s+sinovial"),
        Language.TR: (r"sinovit", r"sinovyal\s+kalınlaş"),
        Language.HR: (r"sinovit", r"sinovijaln\w*\s+zadebljan"),
        Language.DE: (r"synovialit", r"synovit", r"synovial\w*\s+verdick"),
        Language.BG: (r"синовит", r"синовиалн\w*\s+удебел"),
        Language.EL: (r"υµενίτιδ", r"υμενίτιδ", r"πάχυνση\s+του\s+υµένα", r"πάχυνση\s+του\s+υμένα"),
        Language.NL: (r"synovit", r"synoviale\s+verdikking"),
    },
    Finding.BAKERS: {
        Language.EN: (r"baker", r"popliteal\s+cyst"),
        Language.ES: (r"baker", r"quiste\s+poplíte"),
        Language.TR: (r"baker", r"popliteal\s+kist"),
        Language.HR: (r"baker", r"poplitealn\w*\s+cist"),
        Language.DE: (r"baker", r"poplitealzyste"),
        Language.BG: (r"бейкър", r"бейкер", r"поплитеалн\w*\s+киста"),
        Language.EL: (r"baker", r"κύστη\s+.{0,12}ιγνυακ"),
        Language.NL: (r"baker", r"poplitea\w*\s+cyste"),
    },
    Finding.CONTUSION: {
        Language.EN: (r"contusion", r"bone\s+bruise", r"(bone\s+)?marrow\s+edema", r"subchondral\s+.{0,12}edema"),
        Language.ES: (r"contusi", r"edema\s+óse", r"edema\s+de\s+(la\s+)?médula\s+ósea", r"edema\s+medular"),
        Language.TR: (r"kontüzyon", r"kemik\s+iliği\s+ödem", r"kemik\s+ödem", r"subkondral\s+ödem"),
        Language.HR: (r"kontuzij", r"koštan\w*\s+edem", r"edem\s+koštane\s+srži"),
        Language.DE: (r"kontusion", r"bone\s+bruise", r"knochenödem", r"knochenmarködem", r"knochenmarksödem"),
        Language.BG: (r"контузион", r"костно-?мозъчен\s+едем", r"костномозъчен\s+едем", r"субхондрален\s+едем"),
        Language.EL: (r"οστεοµυελικ\w*\s+οίδηµα", r"οστεομυελικ\w*\s+οίδημα", r"µώλωπ", r"μώλωπ"),
        Language.NL: (r"contusie", r"botcontusie", r"botoedeem", r"beenmergoedeem", r"bone\s+bruise"),
    },
    Finding.FRACTURE: {
        Language.EN: (r"fracture", r"cortical\s+break"),
        Language.ES: (r"fractura",),
        Language.TR: (r"fraktür", r"kırık"),
        Language.HR: (r"fraktur", r"prijelom"),
        Language.DE: (r"fraktur", r"knöcherner?\s+ausriss"),
        Language.BG: (r"фрактур", r"счупван"),
        Language.EL: (r"κάταγµα", r"κάταγμα", r"κατάγµατα", r"κατάγματα"),
        Language.NL: (r"fractuur", r"fractur"),
    },
}
"""Each finding's own words in each of the eight languages the corpus is written in.

HAND-BUILT BY READING THE ANNOTATED REPORTS, which is a bias this module cannot remove and the audit
must control for: terms were collected while looking at the cells the reader got WRONG, so they are
better tuned to those than to the cells it got right. `overcall_audit` runs the identical pipeline over
the reader's correct positives for exactly this reason — if the vocabulary were merely fitted to the
false positives, coverage would be visibly worse on the control, and the report prints both.

Written as regex fragments rather than words because seven of the eight languages inflect the anatomy:
`meniska`/`meniskus`/`meniscus` and `medijalni`/`medijalnog`/`medijalne` are the same term declined, and
a word list would need every case of every noun.
"""


class FindingMentions:
    """Where in a report each of the twelve findings is named, in that report's own language."""

    def __init__(self, window: int = WINDOW) -> None:
        self.window = window
        self.terms = {
            finding: {
                language: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
                for language, patterns in languages.items()
            }
            for finding, languages in FINDING_TERMS.items()
        }
        self.partner = {left: right for left, right in COMPARTMENT_PAIRS} | {
            right: left for left, right in COMPARTMENT_PAIRS
        }

    def covers(self, language: str) -> bool:
        """Whether this module has vocabulary for a language at all.

        Asked BEFORE a cell is classified so that "no vocabulary" and "no mention" stay separate
        outcomes. A report in an uncovered language produces no mentions for a trivial reason, and
        counting that as "the finding is not named" would put the tool's blind spot into the tally as
        if it were a fact about the corpus.
        """
        return any(language in languages for languages in FINDING_TERMS.values())

    def sentences(self, report: str) -> list[str]:
        """The report split into spans a cue may be scoped to.

        Splits on terminal punctuation and newlines, then caps each piece at `WINDOW` characters around
        its own start — several reports here are a single unpunctuated paragraph, and an uncapped span
        would scope every cue in the report to every finding in it.
        """
        pieces = [piece.strip() for piece in BOUNDARY.split(report)]
        spans: list[str] = []
        for piece in pieces:
            if len(piece) < MINIMUM_SENTENCE:
                continue
            spans.extend(piece[start : start + self.window] for start in range(0, len(piece), self.window) or [0])
        return spans

    def mentions(self, report: str, language: str, finding: str) -> list[str]:
        """Every span of the report that names `finding`, in `language`'s vocabulary.

        Empty when the finding is not named, when the language is not covered, or when the finding's
        terms happen to be missing for that language. The three are indistinguishable HERE by design —
        `covers` answers the second and the audit reports the residue as unlocated rather than guessing.
        """
        patterns = self.terms.get(finding, {}).get(language, [])
        if not patterns:
            return []
        return [span for span in self.sentences(report) if any(pattern.search(span) for pattern in patterns)]

    def names(self, text: str, language: str, finding: str) -> bool:
        """Whether this exact stretch of text names the finding, WITHOUT splitting it into sentences.

        `mentions` answers "which sentence talks about ACL"; this answers "does this fragment name ACL",
        which is what a caller scoping a cue needs. The audit uses it to ask whether a negation is
        actually denying the finding or merely a qualifier of it: `without extrusion` and `without
        surfacing tear` both sit beside a meniscus, and only the second one denies a meniscus tear.
        """
        patterns = self.terms.get(finding, {}).get(language, [])
        return any(pattern.search(text) for pattern in patterns)

    def elsewhere(self, report: str, language: str, finding: str) -> list[str]:
        """Spans naming the finding's COMPARTMENT PARTNER but not the finding itself.

        The evidence for a laterality error, and the only case this corpus can pose it: a pair is the
        only place two of the twelve columns describe the same lesion and differ only in which side it
        sits on (`core.findings.COMPARTMENT_PAIRS`). A medial-meniscus positive whose report names only
        the LATERAL meniscus is the extractor putting a real finding on the wrong side.
        """
        partner = self.partner.get(finding)
        if partner is None:
            return []
        own = set(self.mentions(report, language, finding))
        return [span for span in self.mentions(report, language, partner) if span not in own]

    def coverage(self, reports: list[tuple[str, str, str]]) -> dict[str, int]:
        """How many of the given (report, language, finding) triples this vocabulary can locate at all.

        The numerator of the fraction the audit is required to state. Returned as counts rather than a
        rate so the caller prints both numbers and can never quote a share without its denominator.
        """
        located = sum(1 for report, language, finding in reports if self.mentions(report, language, finding))
        partnered = sum(
            1
            for report, language, finding in reports
            if not self.mentions(report, language, finding) and self.elsewhere(report, language, finding)
        )
        return {"total": len(reports), "located": located, "partner_only": partnered}

    def known(self) -> tuple[str, ...]:
        """The findings this vocabulary covers, in the project's canonical order.

        A method rather than a constant so the equality with `TARGETS` is CHECKED on every call instead
        of asserted in a comment: a finding added to `core.findings` and not to `FINDING_TERMS` would
        otherwise be silently unlocatable in every report.
        """
        return tuple(target for target in TARGETS if target in self.terms)
