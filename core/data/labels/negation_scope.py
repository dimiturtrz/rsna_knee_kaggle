"""Does a negation in this sentence actually deny THIS finding, or only one of its attributes?

The scope rule behind `core.data.labels.overcall_audit`, split out because it is separately wrong from
the classification that consumes it and has to be separately testable.

WHY IT IS A MODULE AND NOT AN `IF`. Scoring a negation cue anywhere in the span — the obvious reading,
and the one the 2026-08-13 interpretation used — fired on SIXTEEN of the reader's confident TRUE
positives, cells where the radiologist agreed the finding was there. Every one of them was a report
asserting a finding and denying an attribute of it: `without extrusion`, `without articular surface
collapse`, `zonder significant botoedeem`, `без дислокация`, `no desplazada`, `χωρίς καθίζηση`. The
control is what exposed that, and this module is the correction it forced.

THE RULE, IN TWO PARTS, BOTH LOAD-BEARING:

  * DIRECTION. A prepositional denial governs what FOLLOWS it; a normality predicate governs its whole
    clause, because half this corpus's languages put the predicate last (`kollateral ligaman normaldir`).
  * BOUND. A denial stops at the next comma or coordinating conjunction. Without it, `the ACL is not in
    a straight line and increased signal, in favor of tear` reads as a denied tear on the strength of a
    word two clauses downstream.

Against the 311 positive calls on the annotated studies the two parts together took the negation
category from 18 to 12 on the disputed cells and from 16 to 3 on the agreed ones. The 3 that survive
are a DIFFERENT defect this module cannot fix: the report denies the finding in one sentence and
asserts it in another, and the locator hands over the first.

STATE THE CLASS: regex, not a parse. It has no notion of which of two findings a shared clause attaches
to, and it reads no grammar in any of the eight languages it covers.
"""

import logging
import re

from core.data.labels.finding_mentions import FindingMentions
from core.data.labels.report_language import Language
from core.findings import Finding

logger = logging.getLogger(__name__)

DENIAL: dict[str, tuple[str, ...]] = {
    Language.EN: (r"\bno\b", r"\bnot\b", r"without", r"negative\s+for", r"ruled\s+out", r"denie", r"\bnor\b"),
    Language.ES: (r"\bno\b", r"\bsin\b", r"descarta", r"ausencia", r"\bnegativ"),
    Language.TR: (r"\byok\b", r"yoktur", r"mevcut\s+değil", r"\bdeğil\b", r"etmiyor"),
    Language.HR: (r"\bbez\b", r"\bnema\b", r"\bne\s", r"nije\b"),
    Language.DE: (r"\bkein", r"\bohne\b", r"\bnicht\b"),
    Language.BG: (r"\bняма\b", r"\bбез\b", r"\bне\s"),
    Language.EL: (r"\bδεν\b", r"χωρίς", r"ουδέν", r"\bόχι\b"),
    Language.NL: (r"\bgeen\b", r"zonder", r"\bniet\b"),
}
"""Words that deny WHAT FOLLOWS THEM — prepositions and negative particles. Scoped FORWARD ONLY.

The direction is the whole point and it was learned by hand-checking every call this module made. A
prepositional denial in a radiology report overwhelmingly qualifies the finding rather than removing it:
`without extrusion`, `without articular surface collapse`, `zonder significant botoedeem`, `без
дислокация`, `no desplazada`, `χωρίς καθίζηση` all sit in a sentence that ASSERTS the finding and denies
one of its attributes. Scoring the cue anywhere in the span read every one of those as an extraction
error — sixteen of them on cells where the radiologist AGREED with the reader, which is how the control
caught it."""

NORMALITY: dict[str, tuple[str, ...]] = {
    Language.EN: (r"\bnormal", r"\bintact", r"unremarkable", r"preserved", r"\babsent", r"well[- ]maintained"),
    Language.ES: (r"\bnormal", r"conservad", r"íntegr", r"\bintegr"),
    Language.TR: (r"\bnormal", r"korunmuş", r"izlenmemiş", r"görülmemiş", r"saptanmamış", r"izlenmedi", r"doğal"),
    Language.HR: (r"uredn", r"\buredan\b", r"održan", r"očuvan", r"primjeren", r"intaktn"),
    Language.DE: (r"unauffällig", r"\bintakt", r"regelrecht", r"\bnormal", r"erhalten"),
    Language.BG: (r"нормалн", r"нормално", r"запазен", r"съхранен", r"интактн"),
    Language.EL: (r"φυσιολογικ", r"ακέραι"),
    Language.NL: (r"\bnormaal", r"\bintact", r"onopvallend"),
}
"""Words asserting the structure is NORMAL — predicates, not prepositions, so they scope over their whole
clause in EITHER direction.

Two reasons they are separated from `DENIAL` rather than listed beside it. They are verb-final in half
this corpus's languages (`lateral ve medial kollateral ligaman normaldir` puts the denial after
everything it denies, and forward-only scoping would see nothing but a suffix), and they deny the
STRUCTURE wholesale rather than an attribute of it, so the qualifier failure `DENIAL` is prone to cannot
arise. `preserved` and `\\bnormal` are here on the strength of `rubric_boundary`'s own correction: it
found "articular cartilage of the patellofemoral joint is well-maintained" reading as POSITIVE because a
bare search for the finding vocabulary saw `cartilage` and nothing else."""

CLAUSE: dict[str, str] = {
    Language.EN: r",|\band\b|\bbut\b|\bwith\b|\bhowever\b",
    Language.ES: r",|\by\b|\bpero\b|\bcon\b|\baunque\b",
    Language.TR: r",|\bve\b|\bancak\b|\bile\b|\bfakat\b",
    Language.HR: r",|\bi\b|\bali\b|\bte\b|\buz\b",
    Language.DE: r",|\bund\b|\baber\b|\bmit\b|\bjedoch\b",
    Language.BG: r",|\bи\b|\bно\b|\bсъс?\b|\bобаче\b",
    Language.EL: r",|\bκαι\b|\bαλλά\b|\bµε\b|\bμε\b",
    Language.NL: r",|\ben\b|\bmaar\b|\bmet\b|\bechter\b",
}
"""Where a forward-scoping denial STOPS — the next comma or coordinating conjunction.

A denial governs its own clause and not the one after it. Without the bound, `The contour of ACL is not
in a straight line and increased signal intensity inside, in favor of tear` reads as a denied ACL tear
because the word `tear` appears somewhere downstream — on a cell where the radiologist confirmed the
tear. Bounding is the second half of the scope rule and it is as load-bearing as the direction."""

TEAR, BREAK, EDEMA, FLUID, CYST, WEAR, SYNOVIUM = "tear", "break", "edema", "fluid", "cyst", "wear", "synovium"

FAMILY: dict[str, str] = {
    Finding.ACL: TEAR,
    Finding.MCL: TEAR,
    Finding.MEDIAL_MENISCUS: TEAR,
    Finding.LATERAL_MENISCUS: TEAR,
    Finding.MEDIAL_OA: WEAR,
    Finding.LATERAL_OA: WEAR,
    Finding.PF_OA: WEAR,
    Finding.EFFUSION: FLUID,
    Finding.SYNOVITIS: SYNOVIUM,
    Finding.BAKERS: CYST,
    Finding.CONTUSION: EDEMA,
    Finding.FRACTURE: BREAK,
}
"""Which LESION each finding is, because a denial scopes over the lesion word and not the anatomy word.

`without surfacing tear` denies a meniscus tear while naming no meniscus, and `no subchondral bone
marrow edema at the femur` denies a contusion while naming no contusion. A scope test that only looked
for the anatomy vocabulary of `finding_mentions` would miss both."""

DENIABLE: dict[str, dict[str, tuple[str, ...]]] = {
    TEAR: {
        Language.EN: (r"\btear", r"\btorn\b", r"ruptur", r"disrupt", r"sprain", r"\bavuls"),
        Language.ES: (r"rotura", r"ruptur", r"desgarro", r"\blesión", r"avuls"),
        Language.TR: (r"yırtık", r"rüptür", r"yaralan", r"avuls"),
        Language.HR: (r"ruptur", r"puknuć", r"\brasap", r"avuls"),
        Language.DE: (r"\bruptur", r"\briss", r"zerreiss", r"ausriss"),
        Language.BG: (r"разкъсв", r"руптур", r"скъсв"),
        Language.EL: (r"ρήξ", r"ρήγµα", r"ρήγμα"),
        Language.NL: (r"ruptuur", r"scheur", r"laesie"),
    },
    BREAK: {
        Language.EN: (r"fracture", r"\bbreak"),
        Language.ES: (r"fractura",),
        Language.TR: (r"fraktür", r"kırık"),
        Language.HR: (r"prijelom", r"fraktur"),
        Language.DE: (r"fraktur", r"\bbruch"),
        Language.BG: (r"фрактур", r"счупв"),
        Language.EL: (r"κάταγµ", r"καταγμ", r"κατάγµ"),
        Language.NL: (r"fractuur", r"\bbreuk"),
    },
    EDEMA: {
        Language.EN: (r"edema", r"oedema", r"contusion", r"bruise", r"marrow\s+signal"),
        Language.ES: (r"edema", r"contusi"),
        Language.TR: (r"ödem", r"kontüzyon", r"kontuzyon"),
        Language.HR: (r"edem", r"kontuzij", r"nagnječ"),
        Language.DE: (r"ödem", r"kontusion", r"bone\s+bruise"),
        Language.BG: (r"оток", r"едем", r"контуз"),
        Language.EL: (r"οίδηµ", r"οίδημ", r"θλάσ"),
        Language.NL: (r"oedeem", r"contusie", r"kneuzing"),
    },
    FLUID: {
        Language.EN: (r"effusion", r"\bfluid"),
        Language.ES: (r"derrame", r"líquido", r"liquido"),
        Language.TR: (r"efüzyon", r"sıvı"),
        Language.HR: (r"izljev", r"tekućin"),
        Language.DE: (r"erguss", r"flüssigkeit"),
        Language.BG: (r"излив", r"течност"),
        Language.EL: (r"συλλογ", r"υγρ"),
        Language.NL: (r"vocht", r"hydrops", r"effusie"),
    },
    CYST: {
        Language.EN: (r"\bcyst",),
        Language.ES: (r"quiste",),
        Language.TR: (r"\bkist",),
        Language.HR: (r"\bcist",),
        Language.DE: (r"zyste",),
        Language.BG: (r"\bкист",),
        Language.EL: (r"κύστ",),
        Language.NL: (r"\bcyste",),
    },
    WEAR: {
        Language.EN: (
            r"cartilage",
            r"\bchondral",
            r"osteophyt",
            r"\bdefect",
            r"thinning",
            r"fissur",
            r"arthros",
            r"arthrit",
            r"\bwear\b",
            r"joint\s+space",
        ),
        Language.ES: (r"cartílag", r"cartilag", r"\bcondral", r"osteofit", r"artros", r"adelgaza"),
        Language.TR: (r"kıkırdak", r"kondral", r"osteofit", r"artroz", r"incelme"),
        Language.HR: (r"hrskavic", r"\bhondral", r"osteofit", r"artroz", r"stanjen"),
        Language.DE: (r"knorpel", r"\bchondral", r"osteophyt", r"arthros", r"gonarthros"),
        Language.BG: (r"хрущял", r"\bхондрал", r"остеофит", r"артроз", r"изтънява"),
        Language.EL: (r"χόνδρ", r"οστεόφυτ", r"αρθρίτ", r"αρθρωσ", r"αρθρώσ"),
        Language.NL: (r"kraakbeen", r"\bchondraal", r"osteofyt", r"artros", r"gonartros"),
    },
    SYNOVIUM: {
        Language.EN: (r"synovit", r"synovial"),
        Language.ES: (r"sinovit", r"sinovial"),
        Language.TR: (r"sinovit", r"sinovyal"),
        Language.HR: (r"sinovit", r"sinovijaln"),
        Language.DE: (r"synovit", r"synovial"),
        Language.BG: (r"синовит", r"синовиал"),
        Language.EL: (r"υµενίτ", r"υμενίτ", r"αρθρικ\w*\s+υµέν"),
        Language.NL: (r"synovit", r"synoviaal"),
    },
}
"""The lesion words whose denial actually removes each family's finding.

HAND-BUILT, like every other vocabulary here, and it carries the same bias: it was written by reading
the reports this audit then scores. It is stated as a separate list from `finding_mentions.FINDING_TERMS`
because a denial has to land on ONE of the two — the anatomy or the lesion — and a rule that accepted
either would re-admit `partial radial tear of the lateral meniscus without extrusion`, whose forward
scope names no lesion at all and is exactly the false alarm the split exists to stop."""


class NegationScope:
    """Whether a denial in a span reaches the finding, or stops at one of its attributes."""

    def __init__(self, mentions: FindingMentions | None = None) -> None:
        self.mentions = mentions or FindingMentions()
        self.denials = self._compiled(DENIAL)
        self.normality = self._compiled(NORMALITY)
        self.deniable = {family: self._compiled(vocabulary) for family, vocabulary in DENIABLE.items()}
        self.clause = {language: re.compile(pattern, re.IGNORECASE) for language, pattern in CLAUSE.items()}

    @staticmethod
    def _compiled(vocabulary: dict[str, tuple[str, ...]]) -> dict[str, list[re.Pattern[str]]]:
        """One language-keyed vocabulary compiled once, so no pattern is rebuilt per cell."""
        return {
            language: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for language, patterns in vocabulary.items()
        }

    def governed(self, span: str, language: str, start: int) -> str:
        """The stretch a denial beginning at `start` actually governs: up to the next clause boundary."""
        tail = span[start:]
        boundary = self.clause[language].search(tail) if language in self.clause else None
        return tail[: boundary.start()] if boundary else tail

    def about(self, text: str, language: str, finding: str) -> bool:
        """Whether this stretch of text names the finding — by its ANATOMY or by its LESION.

        Either alone is enough: `no fracture` names the lesion with no anatomy, `medial meniscus is
        normal` names the anatomy with no lesion, and both deny the finding.
        """
        patterns = self.deniable[FAMILY[finding]].get(language, [])
        return self.mentions.names(text, language, finding) or any(pattern.search(text) for pattern in patterns)

    def denied(self, span: str, language: str, finding: str) -> bool:
        """Whether a negation in this span scopes over THIS finding rather than an attribute of it."""
        for pattern in self.denials.get(language, []):
            if any(
                self.about(self.governed(span, language, found.end()), language, finding)
                for found in pattern.finditer(span)
            ):
                return True
        return any(pattern.search(span) for pattern in self.normality.get(language, [])) and self.about(
            span, language, finding
        )
