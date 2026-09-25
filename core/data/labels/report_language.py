"""What language each report is written in, and whether it grades its findings.

Run as `python -m core.data.labels.report_language`.

The reports are not an English corpus with a non-English tail. English is 40% of them, and the rest is
Spanish, Turkish, Croatian, Greek, German, Bulgarian and Dutch — every one of them a full radiology
report that grades its findings in its own vocabulary.

THIS EXISTS BECAUSE A MEASUREMENT WAS TAKEN WITH THE WRONG RULER. `interpretations/kneemri/
2026-08-13_what-the-labels-actually-contain.md` reported that 16.6% of non-English reports carry a
severity marker, and flagged it as a LOWER BOUND rather than a measurement, on the grounds that English
patterns against non-English text find little by construction. That caution was right and the correction
is large: read in their own vocabularies, non-English reports carry severity markers 56-89% of the time.
The 16.6% was a property of the tool.

STATE THE DENOMINATOR, because this file exists to correct one — and the first version of this file got
it wrong in its own turn. "Carries a severity marker" is measured over the reports CLASSIFIED as a
language, and 9.3% of the corpus was classified as none. That residue was not noise: all but six of the
412 are CROATIAN, a full eighth radiology language, and the LONGEST reports in the corpus (median 1164
characters against 945 for the classified ones). Read in its own words it grades 86.2% of the time and
to English patterns 0.0% — the largest blind spot the English ruler had, and it sat inside the very
denominator this docstring was warning about. An unclassified residue is a language you have not
named until you have read it.

Nor does a marker being present mean it attaches to a particular finding — measured in English,
per-finding regex extraction adds nothing a level does not already carry.
"""

import logging
import re

import pandas as pd

from core import cli

logger = logging.getLogger(__name__)

MINIMUM_SIGNATURE_WORDS = 2
"""Distinct signature words a language needs before it is claimed, so one shared word cannot decide."""

SIGNATURES: dict[str, tuple[str, ...]] = {
    "en": ("the", "and", "with", "there", "is", "joint", "tear", "normal", "signal"),
    "tr": ("ve", "ile", "izlenmiştir", "izlendi", "görünüm", "kesiminde", "düzeyinde", "mevcut", "bulgular", "diz"),
    "es": ("de", "la", "el", "con", "sin", "rodilla", "menisco", "articular", "se", "del"),
    "de": ("der", "die", "das", "und", "mit", "ohne", "gelenk", "kniegelenk", "nachweis", "im", "kein"),
    "bg": ("на", "мр", "данни", "става", "няма", "се", "костите", "менискус"),
    "nl": ("van", "het", "een", "geen", "met", "knie", "mediaal", "rechts", "links", "meniscus"),
    "el": ("ΓΟΝΑΤΟΣ", "εξέταση", "χωρίς", "του", "της", "ευρηματα", "µε", "και"),
    "hr": ("bez", "znakova", "meniska", "hrskavice", "ligamenta", "zgloba", "promjene", "signala", "intenziteta"),
}
"""Function words and anatomy, never diacritics: a character class of accented letters scored ANY accented
text as Turkish and reported the corpus 94% Turkish, which is the same blind-spot error this file corrects."""

SEVERITY: dict[str, tuple[str, ...]] = {
    "en": (r"grade\s*[i1-4v]", r"\bmild", r"\bmoderate", r"\bsevere", r"full[- ]thickness", r"partial[- ]thickness"),
    "tr": (r"grade?\s*[i1-4v]", r"derece", r"hafif", r"\borta\b", r"ileri", r"ciddi", r"tam\s*kat", r"parsiyel"),
    "es": (r"grado\s*[i1-4v]", r"\bleve", r"moderad", r"\bsever", r"\bgrave", r"espesor\s+completo", r"parcial"),
    "de": (r"grad\s*[i1-4v]", r"gering", r"mäßig", r"mittelgradig", r"\bschwer", r"ausgeprägt", r"vollschichtig"),
    "bg": (r"степен", r"\bлек", r"умерен", r"тежк", r"изразен", r"пълно", r"частичн"),
    "nl": (r"graad\s*[i1-4v]", r"\blicht", r"matig", r"ernstig", r"volledig", r"partieel", r"gering"),
    "el": (r"βαθμ", r"ήπι", r"μέτρι", r"σοβαρ", r"πλήρ", r"μερικ"),
    "hr": (r"stupnj", r"stupanj", r"\bblag", r"umjeren", r"\btešk", r"izražen", r"potpun", r"parcijaln", r"djelomičn"),
}
"""Each language's own grading words — the whole point, since the English set finds 1.7% of Spanish."""

HEDGES: dict[str, tuple[str, ...]] = {
    "en": (
        r"possib",
        r"suspect",
        r"cannot\s+(be\s+)?exclude",
        r"can\s*not\s+exclude",
        r"equivocal",
        r"indetermin",
        r"may\s+repres",
        r"question\s+of",
        r"differential",
        r"\bquer(y|ie)",
        r"\bvs\b",
        r"versus",
    ),
}
"""Words a report uses when it is NOT sure the finding is there — English only, and deliberately so.

The other seven languages are missing because the measurement that would have justified writing them
came back backwards. Split by this flag, the reader's positive calls agree with the radiologist MORE on
hedged reports than on plain ones, at every verdict level (knee-mxj, 2026-08-21), and the reason is
structural rather than a small sample: the reader has already READ the hedge. Its five verdict levels
ARE its confidence in the text, so a regex over the same sentence re-derives a quantity the value column
already carries, and re-derives it worse. Where a report equivocates the reader has already downgraded
the cell, which leaves the residual assertion more reliable, not less.

Kept, and kept English-only, because the flag is a legitimate ACCEPTANCE TEST for a future reader even
though it is not a supervision signal: a replacement that claims to handle uncertainty better has to
move this split, and the module cannot ask the question without the vocabulary."""

UNKNOWN = "unknown"


class ReportLanguages:
    """Which language a report is in, and whether it grades its findings in that language."""

    def __init__(self) -> None:
        self.words = {
            code: [re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE) for word in signature]
            for code, signature in SIGNATURES.items()
        }
        self.markers = self._compiled(SEVERITY)
        self.hedges = self._compiled(HEDGES)

    @staticmethod
    def _compiled(vocabulary: dict[str, tuple[str, ...]]) -> dict[str, list[re.Pattern[str]]]:
        return {
            code: [re.compile(pattern, re.IGNORECASE) for pattern in patterns] for code, patterns in vocabulary.items()
        }

    def language(self, report: str) -> str:
        """The best-matching language, or `unknown` when no language matches enough signature words."""
        matched = {code: sum(1 for word in words if word.search(report)) for code, words in self.words.items()}
        best = max(matched, key=lambda code: matched[code])
        return best if matched[best] >= MINIMUM_SIGNATURE_WORDS else UNKNOWN

    def graded(self, report: str, language: str) -> bool:
        """Whether the report carries an explicit severity marker IN THE GIVEN LANGUAGE.

        The language is passed rather than re-detected so a caller can ask the question the corrected
        measurement turns on: what the ENGLISH patterns find in a Spanish report, against what Spanish
        patterns find in it.
        """
        return any(marker.search(report) for marker in self.markers.get(language, []))

    def hedged(self, report: str, language: str) -> bool:
        """Whether the report qualifies its findings — "possible", "cannot exclude" — IN THE GIVEN LANGUAGE.

        Returns False for every language but English, because only English vocabulary is written; see
        `HEDGES` for the measurement that stopped the other seven from being written. A caller splitting
        the corpus on this must therefore restrict itself to English reports rather than reading the
        False side as "not hedged".
        """
        return any(hedge.search(report) for hedge in self.hedges.get(language, []))

    def census(self, reports: list[str]) -> list[str]:
        """One line per language: how many reports, what share, and how many grade their findings.

        Both rates are reported side by side because the pair is the finding — the own-language rate is
        what the corpus contains, and the English-pattern rate is what the earlier measurement could see.
        """
        languages = [self.language(report) for report in reports]
        lines = []
        for code in sorted(set(languages), key=lambda item: -languages.count(item)):
            texts = [report for report, found in zip(reports, languages, strict=True) if found == code]
            own = sum(1 for text in texts if self.graded(text, code))
            english = sum(1 for text in texts if self.graded(text, "en"))
            share = len(texts) / len(reports)
            lines.append(
                f"{code:>8} {len(texts):>6} {share:>7.1%}   graded {own / len(texts):>6.1%} in its own words, "
                f"{english / len(texts):>6.1%} to English patterns"
            )
        return lines


def main() -> None:
    """Report the corpus's language mix and how much of it grades its findings."""
    command = cli.Command(__doc__, logger)
    arguments = command.parsed()

    root = command.rooted(arguments)
    reports = pd.read_csv(root.train_table())["Report"].fillna("").astype(str).tolist()
    command.report(ReportLanguages().census(reports))


if __name__ == "__main__":
    main()
