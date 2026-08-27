"""The host's published criteria cross-tabbed against our derived targets, on reports written here."""

import pandas as pd

from core.data.labels.rubric_gap import (
    ACL_RULE,
    DENIED,
    MCL_RULE,
    MEETS,
    MENISCUS_RULE,
    RUBRIC,
    SILENT,
    SUBTHRESHOLD,
    UNQUALIFIED,
    RubricGap,
)
from core.findings import CRITERIA, STUDY, TARGETS

DEGENERATE = (
    "There is intrasubstance degeneration of the medial meniscus without a definite tear. "
    "The lateral meniscus is intact and there is no fracture in the knee."
)
TEAR_REPORT = (
    "There is an oblique tear of the medial meniscus reaching the inferior articular surface. "
    "The lateral meniscus appears intact and there is no joint effusion in the knee."
)
SILENT_REPORT = "The cruciate ligaments are intact and there is no fracture or joint effusion in the knee."

MEDIAL, LATERAL = MENISCUS_RULE.columns
ACL, MCL = ACL_RULE.columns[0], MCL_RULE.columns[0]
POSITIVE, ABSENT = 0.82, 0.08


def _reports(*reports: str) -> pd.DataFrame:
    rows = [{STUDY: f"study-{index}", "Report": report} for index, report in enumerate(reports)]
    return pd.DataFrame(rows)


def _labels(column: str, *grades: float) -> pd.DataFrame:
    rows = []
    for index, grade in enumerate(grades):
        row: dict[str, object] = dict.fromkeys(TARGETS, ABSENT)
        row[STUDY] = f"study-{index}"
        row[column] = grade
        rows.append(row)
    return pd.DataFrame(rows)


def test_sided():
    """Only a compartment pair needs a side; a one-column rule takes any sentence naming it."""
    assert MENISCUS_RULE.sided()
    assert not ACL_RULE.sided()
    assert not MCL_RULE.sided()


def test_criterion():
    """A rule quotes the host's published criterion rather than restating it.

    Reading topic 733343 verbatim is what caught `oedema` sitting in MCL's SUBTHRESHOLD vocabulary when
    the published criterion names edema as part of the POSITIVE one. Sourcing the text from one place
    keeps a rule and the reader's prompt from drifting apart.
    """
    assert MCL_RULE.criterion() == CRITERIA[MCL]
    assert MENISCUS_RULE.criterion() == CRITERIA[MEDIAL]
    assert "edema within and adjacent to the ligament" in MCL_RULE.criterion()


def test_sentences():
    """Side-scoped for the pair, because one sentence can report a medial tear beside an intact lateral."""
    gap = RubricGap()
    both = "Medial meniscal tear, lateral meniscus intact. No fracture"

    assert gap.sentences(MENISCUS_RULE, both, MEDIAL) == ["Medial meniscal tear, lateral meniscus intact"]
    assert gap.sentences(MENISCUS_RULE, SILENT_REPORT, MEDIAL) == []
    assert gap.sentences(ACL_RULE, "The ACL is intact. No fracture", ACL) == ["The ACL is intact"]


def test_bucket():
    """The four classes the stated rules turn on, one report per class.

    The first assertion is the whole module: "degeneration without a definite tear" carries a tear WORD,
    and reading it as a tear would file the very cells this tool exists to count under the accepted class.
    """
    gap = RubricGap()

    assert gap.bucket(MENISCUS_RULE, DEGENERATE, MEDIAL) == SUBTHRESHOLD
    assert gap.bucket(MENISCUS_RULE, TEAR_REPORT, MEDIAL) == MEETS
    assert gap.bucket(MENISCUS_RULE, DEGENERATE, LATERAL) == UNQUALIFIED
    assert gap.bucket(MENISCUS_RULE, SILENT_REPORT, MEDIAL) == SILENT


def test_bucket_reads_every_rule():
    """ACL and MCL carry their own subthreshold vocabulary, and it is not the meniscus one."""
    gap = RubricGap()

    assert gap.bucket(ACL_RULE, "Mild ACL signal change without discontinuity", ACL) == SUBTHRESHOLD
    assert gap.bucket(ACL_RULE, "Complete ACL rupture with fibre discontinuity", ACL) == MEETS
    assert gap.bucket(MCL_RULE, "Low-grade MCL sprain, chronic in appearance", MCL) == SUBTHRESHOLD
    assert gap.bucket(MCL_RULE, "High-grade MCL tear with full-thickness disruption", MCL) == MEETS


def test_table():
    """One row per English study per column, carrying our derived target's verdict beside the bucket."""
    gap = RubricGap()
    table = gap.table(MENISCUS_RULE, _reports(DEGENERATE, TEAR_REPORT), _labels(MEDIAL, POSITIVE, POSITIVE))

    assert len(table) == len(_reports(DEGENERATE, TEAR_REPORT)) * len(MENISCUS_RULE.columns)
    medial = table[table["column"] == MEDIAL].set_index("study")
    assert medial.at["study-0", "bucket"] == SUBTHRESHOLD
    assert bool(medial.at["study-0", "positive"])


def test_report():
    """The headline rate is over the CORPUS, not over the English subset that parsed."""
    gap = RubricGap()
    corpus = 100
    table = gap.table(MENISCUS_RULE, _reports(DEGENERATE), _labels(MEDIAL, POSITIVE))
    lines = gap.report(MENISCUS_RULE, table, corpus)

    assert MENISCUS_RULE.criterion() in lines[0]
    assert f"of {corpus} in the corpus" in lines[1]
    assert any(SUBTHRESHOLD in line for line in lines)
    assert lines[-1].startswith("  contradicted: 1 cells")


def test_contradicted():
    """A subthreshold finding our target calls positive is the mislabelled cell; an absent one is not."""
    gap = RubricGap()

    def rows(report: str, grade: float) -> int:
        return len(gap.contradicted(gap.table(MENISCUS_RULE, _reports(report), _labels(MEDIAL, grade))))

    assert rows(DEGENERATE, POSITIVE) == 1
    assert rows(DEGENERATE, ABSENT) == 0
    assert rows(TEAR_REPORT, POSITIVE) == 0


def test_annotated():
    """The annotation read beside the same buckets — the check on the rule rather than on us."""
    gap = RubricGap()
    truth = _labels(MEDIAL, ABSENT)
    truth["Report"] = [DEGENERATE]
    table = gap.table(MENISCUS_RULE, _reports(DEGENERATE), _labels(MEDIAL, POSITIVE))
    lines = gap.annotated(MENISCUS_RULE, table, truth)

    assert lines[0].strip().startswith("1 of the 1 annotated studies are English")
    degenerate = next(line for line in lines if SUBTHRESHOLD in line)
    assert "annotated positive   0" in degenerate
    assert "our target positive   1" in degenerate


def test_denial_is_not_a_finding():
    """The denial is what separates the two classes, so it is asserted apart from the bucket table.

    Every report in the subthreshold class carries an accepted word inside a denial, and the module reads
    the right one only because the denial is subtracted before the positive search. A regression here
    would not fail `test_bucket` loudly — it would move every measured cell into the accepted class and
    report zero contradicted.
    """
    gap = RubricGap()

    assert gap.bucket(MENISCUS_RULE, "Medial meniscus shows degeneration with no definite tear", MEDIAL) == SUBTHRESHOLD
    assert gap.bucket(MENISCUS_RULE, "Medial meniscus intact, no tear", MEDIAL) == DENIED
    assert gap.bucket(MENISCUS_RULE, "No lateral meniscal tear, but a medial meniscal tear is seen", MEDIAL) == MEETS
    assert gap.bucket(ACL_RULE, "ACL signal change, without discontinuity of the fibres", ACL) == SUBTHRESHOLD


def test_rubric_covers_only_rules_the_report_names():
    """Nine of the twelve columns; the other three name an image feature with no threshold to read."""
    assert len(RUBRIC) == len({rule.columns for rule in RUBRIC})
    covered = [column for rule in RUBRIC for column in rule.columns]
    assert len(covered) == len(set(covered)) == len(TARGETS) - 3
    assert set(covered) <= set(TARGETS)


def test_backed():
    """A rule earns its count only where the radiologist agrees its subthreshold class is negative.

    The gate on every number this module prints. Without it Effusion's 524 cells — 6 of its 12 annotated
    subthreshold cells annotated POSITIVE — would sum into the headline beside meniscus's measured 165 as
    if they were the same kind of fact. The bar is zero counterexamples, because the rule's claim is a
    rate of 0 rather than a rate under a half.
    """
    gap = RubricGap()
    table = gap.table(MENISCUS_RULE, _reports(DEGENERATE), _labels(MEDIAL, POSITIVE))

    agrees = _labels(MEDIAL, ABSENT)
    agrees["Report"] = [DEGENERATE]
    assert gap.backed(MENISCUS_RULE, table, agrees)

    disagrees = _labels(MEDIAL, POSITIVE)
    disagrees["Report"] = [DEGENERATE]
    assert not gap.backed(MENISCUS_RULE, table, disagrees)


def test_subthreshold():
    """The only class the rule's claim is falsifiable on, carrying the radiologist's verdict beside ours."""
    gap = RubricGap()
    table = gap.table(MENISCUS_RULE, _reports(DEGENERATE, TEAR_REPORT), _labels(MEDIAL, POSITIVE, POSITIVE))
    truth = _labels(MEDIAL, ABSENT, ABSENT)
    truth["Report"] = [DEGENERATE, TEAR_REPORT]
    sub = gap.subthreshold(MENISCUS_RULE, table, truth)

    assert list(sub["study"]) == ["study-0"]
    assert not bool(sub["truth"].iloc[0])


def test_backed_is_false_when_nothing_is_annotated():
    """An unmeasured rule has not earned a count either — silence is not agreement."""
    gap = RubricGap()
    table = gap.table(ACL_RULE, _reports(SILENT_REPORT), _labels("ACL", POSITIVE))
    truth = _labels("ACL", ABSENT)
    truth["Report"] = [SILENT_REPORT]
    assert not gap.backed(ACL_RULE, table, truth)
