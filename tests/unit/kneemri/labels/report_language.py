"""Language detection and severity marking, on one real-shaped sentence per language in the corpus."""

from kneemri.labels.report_language import UNKNOWN, ReportLanguages

SPANISH = "Técnica: RMN de la rodilla. Resultados: Rotura parcial del menisco interno con derrame articular."
GERMAN = "Kniegelenk mit geringgradiger Chondropathie und ohne Nachweis einer Meniskusläsion im medialen Kompartiment."
TURKISH = "Medial menisküs posterior hornu kesiminde grade III dejenerasyon izlendi ve hafif eklem sıvısı mevcut."
ENGLISH = "There is a partial-thickness tear of the medial meniscus with mild joint effusion and normal signal."
CROATIAN = "Bez znakova rupture medijalnog meniska. Hondromalacija patele 2. stupnja uz promjene signala hrskavice."


def test_language():
    """One representative per language present in the corpus, plus text that matches none.

    The Bulgarian case is the boundary that matters: it shares no alphabet with the others, and an
    earlier detector that scored diacritics rather than words called 94% of this corpus Turkish.
    """
    languages = ReportLanguages()

    assert languages.language(ENGLISH) == "en"
    assert languages.language(SPANISH) == "es"
    assert languages.language(GERMAN) == "de"
    assert languages.language(TURKISH) == "tr"
    assert languages.language("МР находка: МР данни за ставен излив. Костите се променени.") == "bg"
    assert languages.language(CROATIAN) == "hr", "the residue that the first census reported as 9.3% unclassified"
    assert languages.language("Rapporto breve.") == UNKNOWN, "one word is not a language"


def test_graded():
    """A severity marker is found in the report's OWN language and missed by the English patterns.

    This is the whole correction: `parcial` and `geringgradig` grade a finding exactly as `partial` and
    `mild` do, and an English-only reading reported 16.6% of non-English reports as carrying severity
    when the real rates run 56-89%.
    """
    languages = ReportLanguages()

    assert languages.graded(SPANISH, "es")
    assert languages.graded(GERMAN, "de")
    assert languages.graded(TURKISH, "tr")
    assert languages.graded(ENGLISH, "en")

    assert languages.graded(CROATIAN, "hr"), "`stupnja` is a grade, and 86.2% of the Croatian block carries one"

    assert not languages.graded(SPANISH, "en"), "the English patterns are what under-counted Spanish"
    assert not languages.graded(GERMAN, "en")
    assert not languages.graded(CROATIAN, "en"), "English patterns find 0.0% of it — the corpus's largest blind spot"
    assert not languages.graded("Menisco sin alteraciones.", "es"), "an ungraded report must not match"


def test_census():
    """One line per language, carrying both rates — what the corpus says and what English patterns see."""
    lines = ReportLanguages().census([ENGLISH, SPANISH, SPANISH, GERMAN])

    assert len(lines) == 3, "three languages among four reports"
    assert lines[0].lstrip().startswith("es"), "the commonest language is reported first"
    assert "100.0% in its own words" in lines[0]
    assert "0.0% to English patterns" in lines[0]
