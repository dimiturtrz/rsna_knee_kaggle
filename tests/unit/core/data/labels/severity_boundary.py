"""The stated effusion threshold cross-tabbed against the annotation, on reports written here."""

import pandas as pd

from core.data.labels.severity_boundary import (
    DENIAL,
    LARGE,
    MIXED,
    SILENT,
    SMALL,
    UNQUALIFIED,
    SeverityBoundary,
)
from core.findings import TARGETS

ENGLISH_REPORT = (
    "There is a small joint effusion in the knee with mild synovial thickening. "
    "The anterior cruciate ligament and the medial meniscus are intact and there is no fracture."
)
MODERATE_REPORT = (
    "There is a moderate joint effusion with a distended suprapatellar bursa in the knee. "
    "The anterior cruciate ligament and the lateral meniscus are intact and there is no fracture."
)


def _annotated(*reports: str) -> pd.DataFrame:
    rows = []
    for index, report in enumerate(reports):
        row: dict[str, object] = dict.fromkeys(TARGETS, 0.0)
        row["StudyInstanceUID"] = f"study-{index}"
        row["Report"] = report
        rows.append(row)
    return pd.DataFrame(rows)


def test_sentences():
    """Sentence-scoped, because one paragraph grades several findings and each carries its own size word."""
    boundary = SeverityBoundary()
    found = boundary.sentences("Moderate medial compartment osteoarthritis. Small joint effusion. No fracture")

    assert found == [" Small joint effusion"], "the moderate belongs to the osteoarthritis, not the effusion"
    assert boundary.sentences("Intact cruciate ligaments") == []


def test_bucket():
    """Each side of the stated threshold, plus the two rows that are not evidence about it at all."""
    boundary = SeverityBoundary()

    assert boundary.bucket("Small knee effusion") == SMALL
    assert boundary.bucket("Moderate joint effusion") == LARGE
    assert boundary.bucket("Small to moderate effusion") == MIXED
    assert boundary.bucket("Joint effusion is present") == UNQUALIFIED
    assert boundary.bucket("Intact ligaments, no fracture") == SILENT
    assert boundary.bucket("No significant knee effusion") == DENIAL, "a denial is not a moderate effusion"


def test_table():
    """One row per annotated English study, the annotation read as the boolean the threshold predicts."""
    boundary = SeverityBoundary()
    annotated = _annotated(ENGLISH_REPORT, MODERATE_REPORT)
    annotated.loc[0, "Effusion"] = 1.0

    table = boundary.table(annotated)

    assert list(table["bucket"]) == [SMALL, LARGE], "both reports are English and both were read"
    assert list(table["positive"]) == [True, False]


def test_report():
    """The denominator leads, because a rate inside the subset that parsed is not a rate over the 58."""
    boundary = SeverityBoundary()
    annotated = _annotated(ENGLISH_REPORT, MODERATE_REPORT)
    annotated.loc[0, "Effusion"] = 1.0

    lines = boundary.report(boundary.table(annotated), annotated=58)

    assert lines[0].startswith("2 English annotated studies of 58 annotated"), "the subset never hides the whole"
    assert lines[-1].endswith("threshold"), "neither bucket holds both classes at n=1 apiece"


def test_contradictions():
    """The identifiers, because a count is a claim and the studies are what makes it checkable."""
    boundary = SeverityBoundary()
    annotated = _annotated(ENGLISH_REPORT, MODERATE_REPORT)
    annotated.loc[0, "Effusion"] = 1.0

    found = boundary.contradictions(boundary.table(annotated))

    assert found == [
        f"  study-0  {SMALL}, annotated 1",
        f"  study-1  {LARGE}, annotated 0",
    ], "the rule is a threshold, so it is wrong in BOTH directions and both are named"
