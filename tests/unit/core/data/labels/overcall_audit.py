"""Classifying an over-call, on synthetic spans built to pose each rule's known failure.

Every span here is written by hand rather than read from the corpus, so these tests touch no data root
and pin BEHAVIOUR rather than the tally. The negation cases are the ones that matter: the scope rule was
added because a cue scored anywhere in the span fired on sixteen cells where the radiologist AGREED with
the reader, and each of those shapes is kept here so the defect cannot come back silently.
"""

import math

import pandas as pd

from core.data.labels.overcall_audit import (
    GENUINE,
    HEDGE_DROPPED,
    HISTORICAL,
    LATERALITY,
    MISSED_NEGATION,
    SUBTHRESHOLD,
    UNLOCATED,
    Judgement,
    OvercallAudit,
    OvercallCensus,
)

TORN = "There is a full-thickness tear of the medial meniscus."
DENIED = "No fracture is seen."


def _frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Cells in the shape `audit` consumes, so no test reaches for a real reader table."""
    return pd.DataFrame(rows, columns=["study", "target", "language"])


def test_row():
    """A judgement flattens to one artifact row, cues joined so the CSV stays one line per cell."""
    row = Judgement(MISSED_NEGATION, DENIED, (MISSED_NEGATION, SUBTHRESHOLD)).row()

    assert row == {"category": MISSED_NEGATION, "cues": "missed_negation subthreshold", "span": DENIED}


def test_fired():
    """Every cue a span carries comes back, not only the winner, so precedence can be re-derived."""
    audit = OvercallAudit()

    assert audit.fired(TORN, "en", "Medial Meniscus") == ()
    assert audit.fired(DENIED, "en", "Fracture") == (MISSED_NEGATION,)
    assert audit.fired("Small joint effusion.", "en", "Effusion") == (SUBTHRESHOLD,)

    both = audit.fired("Status post repair, possible small re-tear of the medial meniscus.", "en", "Medial Meniscus")
    assert both == (HISTORICAL, HEDGE_DROPPED, SUBTHRESHOLD), "three cues, in precedence order, none dropped"


def test_classify():
    """One cell's category and the span it was read from, including both residue outcomes."""
    audit = OvercallAudit()

    assert audit.classify(TORN, "en", "Medial Meniscus").category == GENUINE, "a plain current assertion"
    assert audit.classify(DENIED, "en", "Fracture").category == MISSED_NEGATION
    assert audit.classify("Trace joint fluid.", "en", "Effusion").category == SUBTHRESHOLD
    assert audit.classify("Tear of the lateral meniscus.", "en", "Medial Meniscus").category == LATERALITY

    unlocated = audit.classify("Unremarkable examination.", "en", "Baker's")
    assert unlocated.category == UNLOCATED, "a finding this vocabulary cannot find is never folded into genuine"
    assert unlocated.span == "", "an unlocated cell carries no span to review"


def test_audit():
    """Cells are classified through the caller's frame, carrying study, finding and language through."""
    audited = OvercallAudit().audit(
        _frame([("s1", "Fracture", "en"), ("s1", "Medial Meniscus", "en")]),
        {"s1": f"{DENIED} {TORN}"},
    )

    assert list(audited.columns) == ["study", "target", "language", "category", "cues", "span"]
    assert list(audited["category"]) == [MISSED_NEGATION, GENUINE]
    assert audited.loc[0, "span"] == DENIED, "the span is written out so the call can be checked by hand"


def test_census():
    """Every category prints even at zero, so two runs of the table can be compared line for line."""
    counts = OvercallCensus().census(pd.DataFrame({"category": [GENUINE, GENUINE, MISSED_NEGATION]}))

    assert counts[GENUINE] == 2
    assert counts[LATERALITY] == 0, "a category absent from the data still appears in the census"
    assert sum(counts.values()) == 3


def test_compare():
    """The control table: each rate beside its own denominator, and the difference with an interval."""
    audit = OvercallCensus()
    over = pd.DataFrame({"study": ["s1", "s2"], "category": [MISSED_NEGATION, GENUINE]})
    agreed = pd.DataFrame({"study": ["s1", "s2"], "category": [GENUINE, GENUINE]})

    lines = audit.compare(over, agreed)

    assert "(n=2)" in lines[1], "no share is ever printed without the count it was taken over"
    negation = next(line for line in lines if line.startswith(MISSED_NEGATION))
    assert "50.0%" in negation and "0.0%" in negation, "both groups' rates on one line"


def test_interval():
    """The bootstrap draws STUDIES, so a difference driven by one study is not read as a difference."""
    audit = OvercallCensus(resamples=200)
    over = pd.DataFrame({"study": ["s1", "s1", "s2"], "category": [MISSED_NEGATION, MISSED_NEGATION, GENUINE]})
    agreed = pd.DataFrame({"study": ["s1", "s2"], "category": [GENUINE, GENUINE]})

    low, high = audit.interval(over, agreed, MISSED_NEGATION)

    assert low <= high
    assert low >= -1.0 and high <= 1.0, "a difference of two rates cannot leave [-1, 1]"
    assert low < high, "three cells over two studies cannot produce a point estimate"


def test_interval_is_pinned():
    """Bit-equal to the hand-written study loop, and (nan, nan) when no draw had both groups."""
    audit = OvercallCensus(resamples=200)
    over = pd.DataFrame(
        {"study": ["s1", "s1", "s2", "s3"], "category": [MISSED_NEGATION, MISSED_NEGATION, GENUINE, MISSED_NEGATION]}
    )
    agreed = pd.DataFrame({"study": ["s1", "s2", "s3"], "category": [GENUINE, GENUINE, MISSED_NEGATION]})

    assert audit.interval(over, agreed, MISSED_NEGATION) == (0.0, 1.0)
    assert all(math.isnan(bound) for bound in audit.interval(over.iloc[:0], agreed, MISSED_NEGATION))


def test_incidence():
    """Cue firing counted without precedence, so a category buried by the order is still visible."""
    audit = OvercallCensus()
    over = pd.DataFrame({"cues": ["missed_negation subthreshold", ""]})
    agreed = pd.DataFrame({"cues": ["subthreshold"]})

    lines = audit.incidence(over, agreed)
    subthreshold = next(line for line in lines if line.startswith(SUBTHRESHOLD))

    assert "50.0%" in subthreshold, "counted on the over-called cell whose category went to negation"
    assert "100.0%" in subthreshold


def test_coverage():
    """Located and total, always both, because every rate below is computed inside the located subset."""
    lines = OvercallCensus().coverage(
        pd.DataFrame({"category": [GENUINE, UNLOCATED], "language": ["en", "tr"], "target": ["ACL", "ACL"]})
    )

    assert lines[0] == "located the finding in 1 of 2 cells (50.0%)"
    assert "tr 1" in lines[1], "the hole is broken out by language, since the vocabulary is per-language"


def test_limits():
    """The caveats print WITH the number, so a copy-paste cannot separate them."""
    lines = OvercallCensus.limits()

    assert any("SCOPE" in line for line in lines)
    assert any("VOCABULARY" in line for line in lines)
    assert any("SUBTHRESHOLD IS NOT A VERDICT" in line for line in lines), "the prior art that refused it"
