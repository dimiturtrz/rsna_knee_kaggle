"""The scope rule, on every synthetic span that produced a false alarm before the rule existed.

Each span is written by hand rather than read from the corpus, so these tests touch no data root and pin
BEHAVIOUR rather than a tally. They matter more than the audit's own tests: scoring a negation cue
anywhere in a span fired on SIXTEEN cells where the radiologist AGREED with the reader, and every one of
those shapes is kept here so the defect cannot come back silently.
"""

from core.data.labels.negation_scope import NegationScope

TORN = "There is a full-thickness tear of the medial meniscus."
DENIED = "No fracture is seen."
QUALIFIED = "Radial tear of the lateral meniscus without extrusion."


def test_about():
    """A fragment counts as being about a finding through its ANATOMY or its LESION, either alone."""
    scope = NegationScope()

    assert scope.about("the medial meniscus", "en", "Medial Meniscus"), "anatomy with no lesion word"
    assert scope.about(" surfacing tear.", "en", "Medial Meniscus"), "lesion with no anatomy word"
    assert not scope.about(" extrusion.", "en", "Medial Meniscus"), "a qualifier is neither"
    assert not scope.about(" subchondral bone marrow edema.", "en", "PF OA"), "`subchondral` is not `chondral`"


def test_governed():
    """A denial governs up to the next clause boundary and no further."""
    scope = NegationScope()
    span = "not in a straight line and increased signal, in favor of tear"

    assert scope.governed(span, "en", len("not")).strip() == "in a straight line", "the coordinator ends the scope"
    assert scope.governed("no fracture seen", "en", len("no")) == " fracture seen", "an unbounded clause runs on"


def test_denied():
    """The two-part rule: a forward-scoping denial, a clause-scoping normality predicate."""
    scope = NegationScope()

    assert scope.denied(DENIED, "en", "Fracture"), "a denial governing the lesion itself"
    assert scope.denied("Degenerative signal in the lateral meniscus without surfacing tear.", "en", "Lateral Meniscus")
    assert scope.denied("Lateral ve medial kollateral ligaman normaldir.", "tr", "MCL"), "a verb-final predicate"

    assert not scope.denied(QUALIFIED, "en", "Lateral Meniscus"), "the tear is asserted; the extrusion is denied"
    assert not scope.denied("Fracture without displacement.", "en", "Fracture"), "an attribute, not the finding"
    assert not scope.denied("The ACL is not in a straight line and increased signal, in favor of tear.", "en", "ACL"), (
        "the tear sits past the coordinator, outside what the denial governs"
    )
    assert not scope.denied(TORN, "en", "Medial Meniscus"), "no cue at all"
