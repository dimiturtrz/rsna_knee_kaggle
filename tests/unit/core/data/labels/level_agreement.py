"""One calibration constant per level, checked against per-finding rates built here."""

import numpy as np
import pandas as pd

from core.data.labels.level_agreement import HALVES, SEED, STUDY, THIN, LevelAgreement
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


def _two_findings(first: list[tuple[float, float]], second: list[tuple[float, float]]):
    """Two findings populated over disjoint studies, so each carries its own cells at the mild level."""
    reader, annotated = _tables(first)
    other, gold = _tables(second, finding=OTHER)
    other.index = pd.Index([f"other-{name}" for name in other.index], name=STUDY)
    gold.index = other.index
    return pd.concat([reader, other]), pd.concat([annotated, gold])


def test_heterogeneity():
    """A spread only thin cells explain reads as pooling; one they cannot explain reads as beyond it."""
    agreement = LevelAgreement(MEASURED)
    generator = np.random.default_rng(SEED)
    apart = _two_findings([(MILD, 1.0)] * (THIN * 4), [(MILD, 0.0)] * (THIN * 4))

    split = next(line for line in agreement.heterogeneity(agreement.rates(*apart), generator) if "0.68" in line)

    assert "BEYOND POOLING" in split, "twenty all-positive cells beside twenty all-negative is not sampling"
    assert "range 1.000" in split, "the range is reported beside the range chance produces, never alone"

    thin = _two_findings([(MILD, 1.0)] * HALVES, [(MILD, 0.0)] * HALVES)

    quiet = next(line for line in agreement.heterogeneity(agreement.rates(*thin), generator) if "0.68" in line)

    assert "consistent with pooling" in quiet, "two cells each cannot distinguish a finding from the constant"
    assert "range" not in quiet, "no finding reaches THIN cells, so no range is quoted"


def test_halves():
    """The annotated studies land in disjoint interleaved groups, covering all of them exactly once."""
    agreement = LevelAgreement(MEASURED)
    reader, annotated = _tables([(MILD, 1.0)] * (THIN * 2))

    groups = agreement.halves(reader, annotated)

    assert len(groups) == HALVES, "one group per half"
    assert set(groups[0]).isdisjoint(set(groups[1])), "a study cannot fit its own held-out constant"
    assert sorted(list(groups[0]) + list(groups[1])) == sorted(reader.index), "every study is placed once"


def test_refit():
    """The constant a subset alone would produce, beside the cells it rests on."""
    agreement = LevelAgreement(MEASURED)
    reader, annotated = _tables([(MILD, 1.0), (MILD, 1.0), (MILD, 0.0), (MILD, 0.0)])

    whole = agreement.refit(reader, annotated, reader.index)
    part = agreement.refit(reader, annotated, reader.index[:2])

    assert whole[MILD] == (0.5, 4), "two of four annotated positive at the mild level"
    assert part[MILD] == (1.0, 2), "the subset holds only the positive pair, and says so with its n"
    assert whole[CONFIDENT][1] == 0, "a level nobody was assigned rests on no cells"


def test_gap_sigma():
    """How far two independent estimates drift by chance — smaller as the cells behind them grow."""
    tight = LevelAgreement.gap_sigma(0.5, [100, 100])
    loose = LevelAgreement.gap_sigma(0.5, [10, 10])

    assert tight < loose, "more cells, less room for the halves to disagree by chance"
    assert LevelAgreement.gap_sigma(0.5, [0, 0]) == 0.0, "no cells contribute no variance rather than dividing"


def test_stability():
    """A level whose halves sit apart by less than their own noise reads as settled, not as moving."""
    agreement = LevelAgreement(MEASURED)
    balanced = [(MILD, 1.0), (MILD, 1.0), (MILD, 0.0), (MILD, 0.0)] * THIN
    reader, annotated = _tables(balanced)

    mild = next(line for line in agreement.stability(reader, annotated) if "level 0.68" in line)

    assert "sigma" in mild, "the gap is priced in its own standard errors, never in a flat distance"
    assert "settled" in mild, "a period-four fixture puts both annotations in both halves, so neither drifts"

    split, gold = _tables([(MILD, 1.0), (MILD, 0.0)] * THIN)

    moved = next(line for line in agreement.stability(split, gold) if "level 0.68" in line)

    assert "MOVES" in moved, "alternating rows interleave into an all-positive half and an all-negative one"
