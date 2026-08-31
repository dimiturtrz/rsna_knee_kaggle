"""Locating a finding in a report, on synthetic sentences shaped like the corpus's eight languages."""

from core.data.labels.finding_mentions import MINIMUM_SENTENCE, WINDOW, FindingMentions
from core.findings import TARGETS

ENGLISH = "There is a tear of the medial meniscus. The ACL is intact. Small joint effusion."
SPANISH = "Rotura del menisco interno. El LCA se encuentra íntegro."


def test_covers():
    """A language the vocabulary was written for is separated from one it was not.

    The distinction the audit depends on: a report in an uncovered language yields no mentions for a
    trivial reason, and reading that as "the finding is absent" would put the tool's blind spot into
    the tally as a fact about the corpus.
    """
    mentions = FindingMentions()

    assert mentions.covers("en")
    assert mentions.covers("bg"), "Bulgarian is 8% of the corpus and no English pattern reaches it"
    assert not mentions.covers("fr"), "no vocabulary is not the same answer as no mention"


def test_sentences():
    """A report splits on terminal punctuation, and an unpunctuated block is capped rather than kept whole."""
    mentions = FindingMentions()

    assert mentions.sentences(ENGLISH) == [
        "There is a tear of the medial meniscus.",
        "The ACL is intact.",
        "Small joint effusion.",
    ]
    assert mentions.sentences("ok") == [], f"a fragment under {MINIMUM_SENTENCE} characters is not a sentence"

    unpunctuated = "a" * (WINDOW * 2 + 10)
    pieces = mentions.sentences(unpunctuated)
    assert len(pieces) == 3, "an unpunctuated paragraph is cut into windows, not scored as one span"
    assert max(len(piece) for piece in pieces) == WINDOW


def test_mentions():
    """Only the sentences naming the finding come back, in the report's own language."""
    mentions = FindingMentions()

    assert mentions.mentions(ENGLISH, "en", "Medial Meniscus") == ["There is a tear of the medial meniscus."]
    assert mentions.mentions(ENGLISH, "en", "ACL") == ["The ACL is intact."]
    assert mentions.mentions(ENGLISH, "en", "Fracture") == [], "a finding the report never names"
    assert mentions.mentions(SPANISH, "es", "ACL") == ["El LCA se encuentra íntegro."], "`LCA` is the Spanish `ACL`"
    assert mentions.mentions(SPANISH, "en", "ACL") == [], "the English vocabulary cannot read a Spanish report"


def test_names():
    """A bare fragment is tested for the finding WITHOUT being split into sentences.

    What a caller scoping a negation needs: `without extrusion` and `without surfacing tear` both follow
    a meniscus, and only the second one names anything the meniscus columns are about.
    """
    mentions = FindingMentions()

    assert mentions.names("the medial meniscus", "en", "Medial Meniscus")
    assert not mentions.names(" extrusion.", "en", "Medial Meniscus"), "the qualifier names no meniscus"
    assert not mentions.names("the medial meniscus", "en", "ACL")


def test_elsewhere():
    """A compartment partner named where the finding itself is not — the evidence for a laterality error."""
    mentions = FindingMentions()
    report = "Tear of the lateral meniscus."

    assert mentions.elsewhere(report, "en", "Medial Meniscus") == [report], "a medial call on a lateral-only report"
    assert mentions.elsewhere("Tear of the medial meniscus.", "en", "Medial Meniscus") == [], "its own mention wins"
    assert mentions.elsewhere(report, "en", "Fracture") == [], "a finding with no compartment partner"


def test_coverage():
    """Both numbers, never a rate on its own, and partner-only cells counted apart from located ones."""
    counts = FindingMentions().coverage(
        [
            (ENGLISH, "en", "Medial Meniscus"),
            (ENGLISH, "en", "Fracture"),
            ("Tear of the lateral meniscus.", "en", "Medial Meniscus"),
        ]
    )

    assert counts == {"total": 3, "located": 1, "partner_only": 1}
    assert counts["total"] > counts["located"], "the caller can always print the denominator"


def test_known():
    """The vocabulary covers every target, checked on call rather than asserted in a comment."""
    covered = FindingMentions().known()

    assert covered == TARGETS, "a finding added to core.findings and not here would be silently unlocatable"
