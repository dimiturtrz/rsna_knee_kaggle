"""One calibration constant per level, checked against per-finding rates built here."""

import pandas as pd

from core.data.labels.level_agreement import STUDY, THIN, LevelAgreement
from core.findings import TARGETS

MILD = 0.68
CONFIDENT = 0.94
MEASURED: tuple[tuple[float, float], ...] = ((MILD, 0.378), (CONFIDENT, 0.958))
FINDING = TARGETS[0]
OTHER = TARGETS[1]


def _tables(rows: list[tuple[float, float]], finding: str = FINDING) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A reader table and an annotation table over the same studies, one finding populated."""
    read: list[dict[str, object]] = []
    gold: list[dict[str, object]] = []
    for index, (value, annotation) in enumerate(rows):
        identifier = f"study-{index}"
        read.append({STUDY: identifier, **dict.fromkeys(TARGETS, 0.08), finding: value})
        gold.append({STUDY: identifier, **dict.fromkeys(TARGETS, 0.0), finding: annotation})
    return pd.DataFrame(read).set_index(STUDY), pd.DataFrame(gold).set_index(STUDY)


def test_rates():
    """The rate is the annotated positive fraction among the cells the reader put at that level."""
    agreement = LevelAgreement(MEASURED)
    reader, annotated = _tables([(MILD, 1.0), (MILD, 0.0), (MILD, 1.0), (MILD, 1.0), (CONFIDENT, 1.0)])

    rates = agreement.rates(reader, annotated)
    mild = rates[(rates["finding"] == FINDING) & (rates["level"] == MILD)].iloc[0]

    assert (mild["rate"], mild["studies"]) == (0.75, 4), "three of four annotated positive at the mild level"
    assert set(rates["finding"]) == set(TARGETS), "every finding gets a row, so a silent column is visible"
    assert rates[(rates["finding"] == OTHER) & (rates["level"] == MILD)].iloc[0]["studies"] == 0


def test_spread():
    """The pooled constant beside the range it stands in for — thin rows kept out of the range."""
    agreement = LevelAgreement(MEASURED)
    rows = [(MILD, 1.0)] * THIN + [(MILD, 0.0)] * THIN
    reader, annotated = _tables(rows)
    other, gold = _tables([(MILD, 0.0)] * (THIN * 2), finding=OTHER)
    reader.loc[other.index, OTHER] = other[OTHER]
    annotated.loc[gold.index, OTHER] = gold[OTHER]

    lines = agreement.spread(agreement.rates(reader, annotated))

    assert any("pooled constant 0.378" in line and "0.000 to 0.500" in line for line in lines), lines
    assert all("level 0.94" not in line for line in lines), "no finding has THIN cells at 0.94 here"


def test_inverted():
    """Only a rate on the OTHER side of 0.5 from its constant is named — overshooting is not inverting."""
    agreement = LevelAgreement(MEASURED)
    reader, annotated = _tables([(MILD, 1.0)] * (THIN + 1))

    found = agreement.inverted(agreement.rates(reader, annotated))

    assert len(found) == 1, "the mild cells are annotated 1.000 while the constant calls them 0.378"
    assert FINDING in found[0] and "own rate 1.000" in found[0]

    softened, gold = _tables([(MILD, 1.0), (MILD, 0.0), (MILD, 0.0), (MILD, 0.0), (MILD, 0.0)])

    assert agreement.inverted(agreement.rates(softened, gold)) == [], "0.200 and 0.378 share a side of 0.5"
