"""The host's definite-tear rule cross-tabbed against our derived targets, on reports written here."""

import pandas as pd

from core.data.labels.tear_rubric import (
    DEGENERATE_ONLY,
    DENIAL,
    MENISCUS,
    SILENT,
    TORN,
    UNQUALIFIED,
    TearRubric,
)
from core.findings import STUDY, TARGETS

DEGENERATE = (
    "There is intrasubstance degeneration of the medial meniscus without a definite tear. "
    "The lateral meniscus is intact and there is no fracture in the knee."
)
TEAR_REPORT = (
    "There is an oblique tear of the medial meniscus reaching the inferior articular surface. "
    "The lateral meniscus appears intact and there is no joint effusion in the knee."
)
SILENT_REPORT = "The cruciate ligaments are intact and there is no fracture or joint effusion in the knee."

MEDIAL, LATERAL = MENISCUS
POSITIVE, ABSENT = 0.82, 0.08


def _reports(*reports: str) -> pd.DataFrame:
    rows = [{STUDY: f"study-{index}", "Report": report} for index, report in enumerate(reports)]
    return pd.DataFrame(rows)


def _labels(*medial: float) -> pd.DataFrame:
    rows = []
    for index, grade in enumerate(medial):
        row: dict[str, object] = dict.fromkeys(TARGETS, ABSENT)
        row[STUDY] = f"study-{index}"
        row[MEDIAL] = grade
        rows.append(row)
    return pd.DataFrame(rows)


def test_sentences():
    """Side-scoped, because one sentence can report a medial tear beside an intact lateral meniscus."""
    rubric = TearRubric()
    both = "Medial meniscal tear, lateral meniscus intact. No fracture"

    assert rubric.sentences(both, MEDIAL) == ["Medial meniscal tear, lateral meniscus intact"]
    assert rubric.sentences(both, LATERAL) == ["Medial meniscal tear, lateral meniscus intact"]
    assert rubric.sentences(SILENT_REPORT, MEDIAL) == []


def test_bucket():
    """The four classes the stated rule turns on, one report per class.

    The first assertion is the whole module: "degeneration without a definite tear" carries a tear WORD,
    and reading it as a tear would file the very cells this tool exists to count under `definite tear`.
    """
    rubric = TearRubric()

    assert rubric.bucket(DEGENERATE, MEDIAL) == DEGENERATE_ONLY
    assert rubric.bucket(TEAR_REPORT, MEDIAL) == TORN
    assert rubric.bucket(DEGENERATE, LATERAL) == UNQUALIFIED
    assert rubric.bucket(SILENT_REPORT, MEDIAL) == SILENT


def test_table():
    """One row per English study per side, carrying our derived target's verdict beside the bucket."""
    table = TearRubric().table(_reports(DEGENERATE, TEAR_REPORT), _labels(POSITIVE, POSITIVE))

    assert len(table) == len(_reports(DEGENERATE, TEAR_REPORT)) * len(MENISCUS)
    medial = table[table["side"] == MEDIAL].set_index("study")
    assert medial.at["study-0", "bucket"] == DEGENERATE_ONLY
    assert bool(medial.at["study-0", "positive"])


def test_report():
    """The headline rate is over the CORPUS, not over the English subset that parsed."""
    corpus = 100
    table = TearRubric().table(_reports(DEGENERATE), _labels(POSITIVE))
    lines = TearRubric().report(table, corpus)

    assert f"of {corpus} in the corpus" in lines[0]
    assert any(DEGENERATE_ONLY in line for line in lines)
    assert lines[-1].startswith("sides the stated rule calls negative and our target calls positive: 1")


def test_contradicted():
    """A degenerative signal our target calls positive is the mislabelled cell; an absent one is not."""
    rubric = TearRubric()

    assert len(rubric.contradicted(rubric.table(_reports(DEGENERATE), _labels(POSITIVE)))) == 1
    assert rubric.contradicted(rubric.table(_reports(DEGENERATE), _labels(ABSENT))).empty
    assert rubric.contradicted(rubric.table(_reports(TEAR_REPORT), _labels(POSITIVE))).empty


def test_annotated():
    """The annotation read beside the same buckets — the check on the rule rather than on us."""
    rubric = TearRubric()
    truth = _labels(ABSENT)
    truth["Report"] = [DEGENERATE]
    lines = rubric.annotated(rubric.table(_reports(DEGENERATE), _labels(POSITIVE)), truth)

    assert lines[0].startswith("1 of the 1 annotated studies are English")
    degenerate = next(line for line in lines if DEGENERATE_ONLY in line)
    assert "annotated positive   0" in degenerate
    assert "our target positive   1" in degenerate


def test_denial_is_not_a_tear():
    """`DENIED` is what separates the two classes, so it is asserted apart from the bucket table.

    Every report in the degenerate class carries a tear word inside a denial, and the module reads the
    right one only because the denial is subtracted before the tear search. A regression here would not
    fail `test_bucket` loudly — it would move 163 measured cells into `definite tear` and report zero.
    """
    rubric = TearRubric()

    assert rubric.bucket("Medial meniscus shows degeneration with no definite tear", MEDIAL) == DEGENERATE_ONLY
    assert rubric.bucket("Medial meniscus intact, no tear", MEDIAL) == DENIAL
    assert rubric.bucket("No lateral meniscal tear, but a medial meniscal tear is seen", MEDIAL) == TORN
