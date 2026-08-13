"""What language each report is written in, and whether it grades its findings: `python -m kneemri.report_language`.

The reports are not an English corpus with a non-English tail. English is 40% of them, and the rest is
Spanish, Turkish, Greek, German, Bulgarian and Dutch — every one of them a full radiology report that
grades its findings in its own vocabulary.

THIS EXISTS BECAUSE A MEASUREMENT WAS TAKEN WITH THE WRONG RULER. `interpretations/kneemri/
2026-08-13_what-the-labels-actually-contain.md` reported that 16.6% of non-English reports carry a
severity marker, and flagged it as a LOWER BOUND rather than a measurement, on the grounds that English
patterns against non-English text find little by construction. That caution was right and the correction
is large: read in their own vocabularies, non-English reports carry severity markers 56-89% of the time.
The 16.6% was a property of the tool.

STATE THE DENOMINATOR, because this file exists to correct one. "Carries a severity marker" is measured
over the reports CLASSIFIED as a language, and 9.3% of the corpus is classified as none. Nor does a
marker being present mean it attaches to a particular finding — measured in English, per-finding regex
extraction adds nothing a level does not already carry.
"""

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

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
}
"""Each language's own grading words — the whole point, since the English set finds 1.7% of Spanish."""

UNKNOWN = "unknown"


class ReportLanguages:
    """Which language a report is in, and whether it grades its findings in that language."""

    def __init__(self) -> None:
        self.words = {
            code: [re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE) for word in signature]
            for code, signature in SIGNATURES.items()
        }
        self.markers = {
            code: [re.compile(pattern, re.IGNORECASE) for pattern in patterns] for code, patterns in SEVERITY.items()
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from core.paths import DataRoot  # noqa: PLC0415 - CLI-only, and this module is imported as a library

    root = DataRoot.from_config(arguments.paths)
    reports = pd.read_csv(root.raw() / "train.csv")["Report"].fillna("").astype(str).tolist()
    for line in ReportLanguages().census(reports):
        logger.info(line)


if __name__ == "__main__":
    main()
