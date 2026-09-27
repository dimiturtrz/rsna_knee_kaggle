from pathlib import Path

import numpy as np
import pytest

from core.data.corpus.anatomy import ANATOMY, Extent
from core.findings import TARGETS
from kneemri.evaluation.alignment_damage import AlignmentDamage, HoldoutJoin
from kneemri.evaluation.scores import ScoredArm

FOCAL = [target for target in TARGETS if ANATOMY[target].extent == Extent.FOCAL]
REGIONAL = [target for target in TARGETS if ANATOMY[target].extent == Extent.REGIONAL]


def separable(studies: int) -> tuple[np.ndarray, np.ndarray]:
    """Alternating truth with a perfectly ranked prediction, so every label's AUC is 1.0."""
    truth = np.tile((np.arange(studies) % 2).astype(np.float32)[:, None], (1, len(TARGETS)))
    return truth.astype(np.float64), truth


def test_split():
    """The cut is at the median, both halves are populated, and a nan abstains from both."""
    spread = np.array([1.0, 2.0, 3.0, 4.0, np.nan], dtype=np.float32)
    split = AlignmentDamage().split(spread)
    assert split.boundary == 2.5
    assert split.low.tolist() == [True, True, False, False, False]
    assert split.high.tolist() == [False, False, True, True, False]
    assert not split.low[4] and not split.high[4], "an unmeasurable study must not be read as well-agreeing"


def test_sizes():
    """The halves report their own denominators, and the abstention is in neither."""
    split = AlignmentDamage().split(np.array([1.0, 2.0, 3.0, 4.0, np.nan], dtype=np.float32))
    assert split.sizes() == (2, 2)


def test_label_gaps():
    """A finding scored equally well on both halves has a zero gap; a damaged one has a positive gap."""
    studies = 40
    predictions, targets = separable(studies)
    spread = np.linspace(1.0, 10.0, studies).astype(np.float32)
    damage = AlignmentDamage()
    flat = damage.label_gaps(ScoredArm(predictions, targets), damage.split(spread))
    assert set(flat) == set(TARGETS)
    assert flat["ACL"] == pytest.approx(0.0), "identical skill on both halves must read as no gap"

    broken = predictions.copy()
    column = TARGETS.index("ACL")
    broken[studies // 2 :, column] = 0.5
    hurt = damage.label_gaps(ScoredArm(broken, targets), damage.split(spread))
    assert hurt["ACL"] > 0.0, "scoring worse where the slots disagree must give a POSITIVE gap"


def test_by_extent():
    """The two structural classes are averaged separately, and an undefined finding is skipped."""
    gaps = dict.fromkeys(FOCAL, 0.2) | dict.fromkeys(REGIONAL, 0.0)
    grouped = AlignmentDamage().by_extent(gaps)
    assert grouped[Extent.FOCAL] == pytest.approx(0.2)
    assert grouped[Extent.REGIONAL] == pytest.approx(0.0)

    with_hole = gaps | {FOCAL[0]: float("nan")}
    assert AlignmentDamage().by_extent(with_hole)[Extent.FOCAL] == pytest.approx(0.2), (
        "an undefined label must be skipped, not poison its whole class to nan"
    )


def test_differential():
    """The confound-controlled quantity is focal minus regional, and a uniform shift cancels."""
    damage = AlignmentDamage()
    assert damage.differential({Extent.FOCAL: 0.2, Extent.REGIONAL: 0.0}) == pytest.approx(0.2)
    assert damage.differential({Extent.FOCAL: 0.3, Extent.REGIONAL: 0.3}) == pytest.approx(0.0), (
        "a gap that moves both classes equally is difficulty, not misregistration, and must read zero"
    )


def test_interval():
    """The bootstrap brackets a genuinely zero differential and is seeded, so it repeats exactly."""
    studies = 60
    predictions, targets = separable(studies)
    spread = np.linspace(1.0, 10.0, studies).astype(np.float32)
    first = AlignmentDamage().interval(ScoredArm(predictions, targets, 40), spread)
    assert first[0] <= 0.0 <= first[1], "no planted damage, so the interval must contain zero"
    assert AlignmentDamage().interval(ScoredArm(predictions, targets, 40), spread) == first


def test_report():
    """Every finding gets a row, and the denominators and the differential are both stated."""
    studies = 40
    predictions, targets = separable(studies)
    spread = np.linspace(1.0, 10.0, studies).astype(np.float32)
    lines = AlignmentDamage().report(ScoredArm(predictions, targets, 20), spread)
    assert any("DIFFERENTIAL" in line for line in lines)
    assert all(any(target in line for line in lines) for target in TARGETS)
    assert any("well-agreeing studies" in line for line in lines), "the split's denominators must be printed"


def test_combined(tmp_path: Path):
    """Two records holding the same studies in DIFFERENT order must still mix study-for-study."""
    names = np.array(["b", "a", "c"], dtype="<U64")
    truth = np.tile(np.array([[1.0], [0.0], [1.0]], dtype=np.float32), (1, len(TARGETS)))
    predictions = np.tile(np.array([[0.9], [0.1], [0.8]]), (1, len(TARGETS)))
    first = tmp_path / "first.holdout.npz"
    np.savez(first, studies=names, predictions=predictions, truth=truth)

    shuffle = np.array([1, 0, 2])
    second = tmp_path / "second.holdout.npz"
    np.savez(second, studies=names[shuffle], predictions=predictions[shuffle], truth=truth[shuffle])

    mixed, mixed_truth, studies = HoldoutJoin().combined([first, second])
    assert studies.tolist() == ["a", "b", "c"]
    assert mixed_truth[0, 0] == 0.0, "study 'a' is the negative one in both records"
    assert mixed[0, 0] < mixed[1, 0], "'a' ranks below 'b' in both, so the mix must keep that order"


def test_aligned(tmp_path: Path):
    """The spread is reordered onto the record's studies, and an unmeasured one becomes nan."""
    dump = tmp_path / "spread.npz"
    np.savez(
        dump,
        studies=np.array(["c", "a"], dtype="<U64"),
        spread=np.array([3.0, 1.0], dtype=np.float32),
        slots=np.array([2, 2], dtype=np.int32),
    )
    values = HoldoutJoin().aligned(dump, np.array(["a", "b", "c"], dtype="<U64"))
    assert values[0] == pytest.approx(1.0)
    assert np.isnan(values[1]), "a study the alignment pass never decoded must abstain, not default"
    assert values[2] == pytest.approx(3.0)
