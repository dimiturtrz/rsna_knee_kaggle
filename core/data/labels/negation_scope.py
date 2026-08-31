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

logger = logging.getLogger(__name__)

DENIAL: dict[str, tuple[str, ...]] = {
    "en": (r"\bno\b", r"\bnot\b", r"without", r"negative\s+for", r"ruled\s+out", r"denie", r"\bnor\b"),
    "es": (r"\bno\b", r"\bsin\b", r"descarta", r"ausencia", r"\bnegativ"),
    "tr": (r"\byok\b", r"yoktur", r"mevcut\s+değil", r"\bdeğil\b", r"etmiyor"),
    "hr": (r"\bbez\b", r"\bnema\b", r"\bne\s", r"nije\b"),
    "de": (r"\bkein", r"\bohne\b", r"\bnicht\b"),
    "bg": (r"\bняма\b", r"\bбез\b", r"\bне\s"),
    "el": (r"\bδεν\b", r"χωρίς", r"ουδέν", r"\bόχι\b"),
    "nl": (r"\bgeen\b", r"zonder", r"\bniet\b"),
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
    "en": (r"\bnormal", r"\bintact", r"unremarkable", r"preserved", r"\babsent", r"well[- ]maintained"),
    "es": (r"\bnormal", r"conservad", r"íntegr", r"\bintegr"),
    "tr": (r"\bnormal", r"korunmuş", r"izlenmemiş", r"görülmemiş", r"saptanmamış", r"izlenmedi", r"doğal"),
    "hr": (r"uredn", r"\buredan\b", r"održan", r"očuvan", r"primjeren", r"intaktn"),
    "de": (r"unauffällig", r"\bintakt", r"regelrecht", r"\bnormal", r"erhalten"),
    "bg": (r"нормалн", r"нормално", r"запазен", r"съхранен", r"интактн"),
    "el": (r"φυσιολογικ", r"ακέραι"),
    "nl": (r"\bnormaal", r"\bintact", r"onopvallend"),
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
    "en": r",|\band\b|\bbut\b|\bwith\b|\bhowever\b",
    "es": r",|\by\b|\bpero\b|\bcon\b|\baunque\b",
    "tr": r",|\bve\b|\bancak\b|\bile\b|\bfakat\b",
    "hr": r",|\bi\b|\bali\b|\bte\b|\buz\b",
    "de": r",|\bund\b|\baber\b|\bmit\b|\bjedoch\b",
    "bg": r",|\bи\b|\bно\b|\bсъс?\b|\bобаче\b",
    "el": r",|\bκαι\b|\bαλλά\b|\bµε\b|\bμε\b",
    "nl": r",|\ben\b|\bmaar\b|\bmet\b|\bechter\b",
}
"""Where a forward-scoping denial STOPS — the next comma or coordinating conjunction.

A denial governs its own clause and not the one after it. Without the bound, `The contour of ACL is not
in a straight line and increased signal intensity inside, in favor of tear` reads as a denied ACL tear
because the word `tear` appears somewhere downstream — on a cell where the radiologist confirmed the
tear. Bounding is the second half of the scope rule and it is as load-bearing as the direction."""

TEAR, BREAK, EDEMA, FLUID, CYST, WEAR, SYNOVIUM = "tear", "break", "edema", "fluid", "cyst", "wear", "synovium"

FAMILY: dict[str, str] = {
    "ACL": TEAR,
    "MCL": TEAR,
    "Medial Meniscus": TEAR,
    "Lateral Meniscus": TEAR,
    "Medial OA": WEAR,
    "Lateral OA": WEAR,
    "PF OA": WEAR,
    "Effusion": FLUID,
    "Synovitis": SYNOVIUM,
    "Baker's": CYST,
    "Contusion": EDEMA,
    "Fracture": BREAK,
}
"""Which LESION each finding is, because a denial scopes over the lesion word and not the anatomy word.

`without surfacing tear` denies a meniscus tear while naming no meniscus, and `no subchondral bone
marrow edema at the femur` denies a contusion while naming no contusion. A scope test that only looked
for the anatomy vocabulary of `finding_mentions` would miss both."""

DENIABLE: dict[str, dict[str, tuple[str, ...]]] = {
    TEAR: {
        "en": (r"\btear", r"\btorn\b", r"ruptur", r"disrupt", r"sprain", r"\bavuls"),
        "es": (r"rotura", r"ruptur", r"desgarro", r"\blesión", r"avuls"),
        "tr": (r"yırtık", r"rüptür", r"yaralan", r"avuls"),
        "hr": (r"ruptur", r"puknuć", r"\brasap", r"avuls"),
        "de": (r"\bruptur", r"\briss", r"zerreiss", r"ausriss"),
        "bg": (r"разкъсв", r"руптур", r"скъсв"),
        "el": (r"ρήξ", r"ρήγµα", r"ρήγμα"),
        "nl": (r"ruptuur", r"scheur", r"laesie"),
    },
    BREAK: {
        "en": (r"fracture", r"\bbreak"),
        "es": (r"fractura",),
        "tr": (r"fraktür", r"kırık"),
        "hr": (r"prijelom", r"fraktur"),
        "de": (r"fraktur", r"\bbruch"),
        "bg": (r"фрактур", r"счупв"),
        "el": (r"κάταγµ", r"καταγμ", r"κατάγµ"),
        "nl": (r"fractuur", r"\bbreuk"),
    },
    EDEMA: {
        "en": (r"edema", r"oedema", r"contusion", r"bruise", r"marrow\s+signal"),
        "es": (r"edema", r"contusi"),
        "tr": (r"ödem", r"kontüzyon", r"kontuzyon"),
        "hr": (r"edem", r"kontuzij", r"nagnječ"),
        "de": (r"ödem", r"kontusion", r"bone\s+bruise"),
        "bg": (r"оток", r"едем", r"контуз"),
        "el": (r"οίδηµ", r"οίδημ", r"θλάσ"),
        "nl": (r"oedeem", r"contusie", r"kneuzing"),
    },
    FLUID: {
        "en": (r"effusion", r"\bfluid"),
        "es": (r"derrame", r"líquido", r"liquido"),
        "tr": (r"efüzyon", r"sıvı"),
        "hr": (r"izljev", r"tekućin"),
        "de": (r"erguss", r"flüssigkeit"),
        "bg": (r"излив", r"течност"),
        "el": (r"συλλογ", r"υγρ"),
        "nl": (r"vocht", r"hydrops", r"effusie"),
    },
    CYST: {
        "en": (r"\bcyst",),
        "es": (r"quiste",),
        "tr": (r"\bkist",),
        "hr": (r"\bcist",),
        "de": (r"zyste",),
        "bg": (r"\bкист",),
        "el": (r"κύστ",),
        "nl": (r"\bcyste",),
    },
    WEAR: {
        "en": (
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
        "es": (r"cartílag", r"cartilag", r"\bcondral", r"osteofit", r"artros", r"adelgaza"),
        "tr": (r"kıkırdak", r"kondral", r"osteofit", r"artroz", r"incelme"),
        "hr": (r"hrskavic", r"\bhondral", r"osteofit", r"artroz", r"stanjen"),
        "de": (r"knorpel", r"\bchondral", r"osteophyt", r"arthros", r"gonarthros"),
        "bg": (r"хрущял", r"\bхондрал", r"остеофит", r"артроз", r"изтънява"),
        "el": (r"χόνδρ", r"οστεόφυτ", r"αρθρίτ", r"αρθρωσ", r"αρθρώσ"),
        "nl": (r"kraakbeen", r"\bchondraal", r"osteofyt", r"artros", r"gonartros"),
    },
    SYNOVIUM: {
        "en": (r"synovit", r"synovial"),
        "es": (r"sinovit", r"sinovial"),
        "tr": (r"sinovit", r"sinovyal"),
        "hr": (r"sinovit", r"sinovijaln"),
        "de": (r"synovit", r"synovial"),
        "bg": (r"синовит", r"синовиал"),
        "el": (r"υµενίτ", r"υμενίτ", r"αρθρικ\w*\s+υµέν"),
        "nl": (r"synovit", r"synoviaal"),
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
