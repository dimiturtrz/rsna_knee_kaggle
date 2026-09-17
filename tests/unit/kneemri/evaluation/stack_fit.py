"""The stacking harness's contract: nothing is ever scored on the studies that fitted it."""

from pathlib import Path

import numpy as np
import pytest

from kneemri.evaluation.stack_fit import MemberHoldout, StackFit, StackResult, main


def _pool(members: int, studies: int, findings: int) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(0)
    truth = (rng.random((studies, findings)) > 0.7).astype(np.int64)
    signal = truth.astype(np.float64) * 0.4
    return [np.clip(signal + rng.normal(0.3, 0.15, (studies, findings)), 0.01, 0.99) for _ in range(members)], truth


def test_split() -> None:
    """The fitting and scoring positions are disjoint and cover every study."""
    fitting, scoring = StackFit().split(101)
    assert not set(fitting.tolist()) & set(scoring.tolist())
    assert len(fitting) + len(scoring) == 101


def test_mean_area() -> None:
    """A member that ranks the truth perfectly reads 1.0."""
    truth = np.array([[0], [0], [1], [1]], dtype=np.int64)
    assert StackFit.mean_area(truth.astype(np.float64), truth) == 1.0


def test_chosen_penalty() -> None:
    """The penalty comes from the grid, chosen without touching the caller's scoring split."""
    per_member, truth = _pool(members=4, studies=300, findings=4)
    assert StackFit().chosen_penalty(per_member, truth) in {10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0}


def test_scored() -> None:
    """Every rule is judged on the same held-out studies, and none of them is judged on its own fit."""
    per_member, truth = _pool(members=3, studies=400, findings=4)
    members = [
        MemberHoldout(
            name=f"m{index}",
            studies=np.asarray([f"s{position}" for position in range(400)]),
            predictions=array,
            truth=truth,
        )
        for index, array in enumerate(per_member)
    ]
    results = StackFit().scored(members)
    assert [result.rule for result in results] == ["StackedLogit", "LogitAverage", "RankAverage"]
    assert {result.scored_on for result in results} == {200}


def test_aligned() -> None:
    """Members recorded over different studies are cut to the shared set, in one shared order."""
    truth = np.zeros((3, 2), dtype=np.int64)
    first = MemberHoldout("a", np.asarray(["s2", "s1", "s3"]), np.arange(6.0).reshape(3, 2), truth)
    second = MemberHoldout("b", np.asarray(["s3", "s1", "s9"]), np.arange(6.0).reshape(3, 2), truth)
    shared = MemberHoldout.aligned([first, second])
    assert [member.studies.tolist() for member in shared] == [["s1", "s3"], ["s1", "s3"]]
    assert shared[0].predictions[0].tolist() == [2.0, 3.0]


def test_aligned_refuses_an_empty_intersection() -> None:
    """Disjoint holdouts have nothing to be scored on together, and the refusal names the narrowest.

    This is the shape a directory sweep lands in — a twelve-study `--smoke` record beside the real
    pool empties the intersection — and it used to surface as an IndexError from a float64 take.
    """
    truth = np.zeros((2, 2), dtype=np.int64)
    wide = MemberHoldout("wide", np.asarray(["s1", "s2"]), np.arange(4.0).reshape(2, 2), truth)
    smoke = MemberHoldout("smoke", np.asarray(["z1", "z2"]), np.arange(4.0).reshape(2, 2), truth)
    with pytest.raises(ValueError, match=r"share no study.*smoke \(2\)"):
        MemberHoldout.aligned([wide, smoke])


def test_loaded(tmp_path: Path) -> None:
    """A holdout record round-trips, with truth binarised at the project's midpoint."""
    path = tmp_path / "m.holdout.npz"
    np.savez(path, studies=np.asarray(["a"]), predictions=np.array([[0.9]]), truth=np.array([[0.8]]))
    member = MemberHoldout.loaded(path)
    assert member.name == "m"
    assert member.truth.tolist() == [[1]]


def test_line() -> None:
    """A printed row carries the rule that made the number."""
    assert "StackedLogit" in StackResult(rule="StackedLogit", fitted_on=1, scored_on=2, mean=0.87).line()


def test_shipped_weights() -> None:
    """One row per member, one column per finding, fitted on every study handed in."""
    per_member, truth = _pool(members=3, studies=300, findings=4)
    members = [
        MemberHoldout(f"m{index}", np.asarray([f"s{p}" for p in range(300)]), array, truth)
        for index, array in enumerate(per_member)
    ]
    assert StackFit().shipped_weights(members).shape == (3, 4)


def test_main() -> None:
    """The entrypoint is a callable the mirror can see; its behaviour is the CLI, exercised by hand."""
    assert callable(main)
