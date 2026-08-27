"""The host's stated rubric cross-tabbed against our training table, on reports written here."""

import pandas as pd

from core.data.labels.rubric_boundary import (
    BOTH,
    DENIAL,
    EXCLUDED,
    MENISCUS,
    OSTEOARTHRITIS,
    QUALIFIED,
    SILENT,
    UNQUALIFIED,
    RubricBoundary,
)
from core.findings import TARGETS

DEGENERATION = (
    "There is intrasubstance degeneration of the medial meniscus in the knee. "
    "The lateral meniscus is normal and there is no fracture or joint effusion."
)
NEGATED = "Intrasubstance degeneration of the medial meniscus without a definite tear"
DEFINITE = (
    "There is a radial tear of the medial meniscus reaching the articular surface in the knee. "
    "The lateral meniscus is normal and there is no fracture or joint effusion."
)
MIXED = (
    "There is a degenerative tear of the medial meniscus reaching the articular surface in the knee. "
    "The lateral meniscus is normal and there is no fracture or joint effusion."
)
GLOBAL = (
    "There is tricompartmental osteoarthritis of the knee with marginal osteophytes. "
    "The medial meniscus and the lateral meniscus are normal and there is no fracture."
)
CARTILAGE = (
    "Severe chondromalacia patella is noted in the patellofemoral compartment of the knee. "
    "The medial meniscus and the lateral meniscus are normal and there is no fracture."
)


def _reports(*reports: str) -> pd.DataFrame:
    rows = [{"StudyInstanceUID": f"study-{index}", "Report": text} for index, text in enumerate(reports)]
    return pd.DataFrame(rows)


def _targets(*values: float) -> pd.DataFrame:
    rows = []
    for index, value in enumerate(values):
        row: dict[str, object] = dict.fromkeys(TARGETS, value)
        row["StudyInstanceUID"] = f"study-{index}"
        rows.append(row)
    return pd.DataFrame(rows, columns=["StudyInstanceUID", *TARGETS])


def test_matched() -> None:
    """Patterns are searched case-insensitively, so a report's SHOUTED section headers still match."""
    boundary = RubricBoundary()
    assert boundary.matched((r"chondral",), "PATELLOFEMORAL: Chondral thinning")
    assert not boundary.matched((r"chondral",), "The meniscus is normal")


def test_sentences() -> None:
    """Sentence-scoped, because one paragraph grades every compartment and a report-wide regex would
    carry the medial meniscus's tear onto the lateral one."""
    boundary = RubricBoundary()
    found = boundary.sentences(DEFINITE, MENISCUS.mentions["Lateral Meniscus"])
    assert len(found) == 1
    assert "tear" not in found[0]


def test_bucket() -> None:
    """The buckets the host's rule distinguishes, on wording it names outright.

    BOTH is its own bucket and not a contradiction: 'a DEGENERATIVE tear REACHING the surface' carries
    excluded and qualifying wording in one sentence and is positive under the rule, so binning it with
    the excluded-only cells would inflate the single number this module exists to state."""
    boundary = RubricBoundary()
    assert boundary.bucket(DEGENERATION, MENISCUS, "Medial Meniscus") == EXCLUDED
    assert boundary.bucket(DEFINITE, MENISCUS, "Medial Meniscus") == QUALIFIED
    assert boundary.bucket(MIXED, MENISCUS, "Medial Meniscus") == BOTH
    assert boundary.bucket(CARTILAGE, OSTEOARTHRITIS, "PF OA") == EXCLUDED
    assert boundary.bucket(CARTILAGE, OSTEOARTHRITIS, "Medial OA") == SILENT


def test_bucket_reads_an_unnegated_denial_as_a_denial() -> None:
    """A KNEE REPORT SAYS NEGATIVE WITHOUT A NEGATION WORD. 'The cartilage is well-maintained' carries
    `cartilage` into the excluded bucket while saying the opposite, and counting it inflates the one
    number this module exists to state — which is how it was found, by printing the sentences."""
    boundary = RubricBoundary()
    said = "Articular cartilage of the patellofemoral joint is well-maintained in the knee"
    assert boundary.bucket(said, OSTEOARTHRITIS, "PF OA") == DENIAL


def test_bucket_sends_the_hosts_own_phrasing_to_denial_which_is_why_the_count_is_a_FLOOR() -> None:
    """THE MOST IDIOMATIC FORM OF THE NAMED CASE DOES NOT REACH THE CONTRADICTION COUNT. The host wrote
    'degenerative signals that do not reach the articular surface', and a report writes it as
    'degeneration WITHOUT a definite tear' — which carries a negation word and bins as a denial. The
    cell is still counted, in the denial row, where a positive is a reader error by any reading. So the
    contradiction line is a floor on the disagreement and never an estimate of it."""
    boundary = RubricBoundary()
    assert boundary.bucket(NEGATED, MENISCUS, "Medial Meniscus") == DENIAL


def test_bucket_names_wording_the_rule_does_not_reach() -> None:
    """Neither wording is not a guess in either direction — the host's rule simply does not decide it."""
    boundary = RubricBoundary()
    assert boundary.bucket("The medial meniscus is somewhat prominent in the knee", MENISCUS, "Medial Meniscus") == (
        UNQUALIFIED
    )


def test_table() -> None:
    """One row per (English study, column), joined on the REPORTS' row set rather than the targets'."""
    boundary = RubricBoundary()
    table = boundary.table(_reports(DEGENERATION), _targets(1.0))
    medial = table[table["column"] == "Medial Meniscus"].iloc[0]
    assert medial["bucket"] == EXCLUDED
    assert bool(medial["positive"])
    assert "degeneration" in medial["said"]


def test_table_skips_a_study_the_target_table_does_not_carry() -> None:
    """A target table over a merged corpus carries studies with no report, and the reverse holds too;
    counting either as silent would put a corpus difference inside a reader measurement."""
    boundary = RubricBoundary()
    assert boundary.table(_reports(DEGENERATION), _targets()).empty


def test_report() -> None:
    """Both denominators on the first line, because a share of a bucket is not a share of the corpus."""
    boundary = RubricBoundary()
    table = boundary.table(_reports(CARTILAGE), _targets(1.0))
    lines = boundary.report(table, reports=9, english=1)
    assert "1 English reports of 9" in lines[0]
    assert any("CONTRADICTED" in line and "% of this family" in line for line in lines)


def test_attribution() -> None:
    """SPILLOVER SPLITS ON WHETHER ANY NAMED COLUMN OF THE SAME STUDY IS POSITIVE. The cartilage report
    names only the patellofemoral compartment, so with every OA column positive the two silent ones have
    a named positive to have spilled from — and the count has to say so rather than filing them with
    cells no sentence in the report supports at all."""
    boundary = RubricBoundary()
    table = boundary.table(_reports(CARTILAGE), _targets(1.0))
    line = next(text for text in boundary.attribution(table) if text.startswith(OSTEOARTHRITIS.name))
    assert "2 positives on columns no sentence names" in line
    assert "2 spillover" in line


def test_attribution_separates_an_unlocalised_sentence_from_an_invented_label() -> None:
    """WITHOUT THIS ROW THE HEADLINE WOULD BE THE COMPARTMENT REGEX'S BLIND SPOT. 'Tricompartmental
    osteoarthritis' names no compartment, so all three columns are silent and all three positives are
    CORRECT — counting them as invented would report a measurement artefact as a table defect."""
    boundary = RubricBoundary()
    table = boundary.table(_reports(GLOBAL), _targets(1.0))
    line = next(text for text in boundary.attribution(table) if text.startswith(OSTEOARTHRITIS.name))
    assert "3 global" in line
    assert "0 unsupported" in line


def test_attribution_says_so_when_no_positive_is_unnamed() -> None:
    """A family with nothing to explain must not print a rate over an empty denominator."""
    boundary = RubricBoundary()
    table = boundary.table(_reports(CARTILAGE), _targets(0.0))
    assert any(text.startswith(f"{OSTEOARTHRITIS.name} — no positive cell") for text in boundary.attribution(table))


def test_contradictions() -> None:
    """The count is the claim and the sentence is the evidence — a rate quoted alone is not checkable."""
    boundary = RubricBoundary()
    table = boundary.table(_reports(CARTILAGE), _targets(1.0))
    found = boundary.contradictions(table, limit=10)
    assert any("PF OA" in line and "chondromalacia" in line for line in found)


def test_contradictions_is_empty_when_our_table_agrees_with_the_rule() -> None:
    boundary = RubricBoundary()
    table = boundary.table(_reports(CARTILAGE), _targets(0.0))
    assert boundary.contradictions(table, limit=10) == []
