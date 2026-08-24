import json
from pathlib import Path

import numpy as np

from kneemri.evaluation.protocol_prior import TARGETS, Holdout, Ordering, ProtocolPrior


def _holdout(n: int = 40) -> Holdout:
    """A synthetic split where the presence pattern is the ONLY thing that carries the label, so a
    per-stratum prior has something real to find and the blend has somewhere to move."""
    presence = np.zeros((n, 6), dtype=np.int64)
    presence[: n // 2, :] = 1
    presence[n // 2 :, :3] = 1
    truth = np.zeros((n, len(TARGETS)), dtype=np.float64)
    truth[: n // 2, :] = 1.0
    predictions = np.tile(np.linspace(0.0, 1.0, n).reshape(-1, 1), (1, len(TARGETS)))
    return Holdout([f"s{i}" for i in range(n)], predictions, truth, presence)


def test_ranks():
    """Ordinal position rescaled to [0, 1] — the ends pinned, the order preserved."""
    scaled = Ordering.ranks(np.array([3.0, 1.0, 2.0]))

    assert scaled.tolist() == [1.0, 0.0, 0.5]


def test_auc():
    """A perfect separation reads 1.0, and truth is BINARISED first: a graded 0.9 is a positive, and
    summing the column instead of thresholding it is what once produced an AUC of 1.0466."""
    labels = np.array([0.0, 0.1, 0.9, 1.0])

    assert Ordering.auc(np.array([1.0, 2.0, 3.0, 4.0]), labels) == 1.0
    assert Ordering.auc(np.array([4.0, 3.0, 2.0, 1.0]), labels) == 0.0


def test_auc_returns_nan_on_a_constant_label():
    """A degenerate label has no AUC. Reporting 0.5 would flatter every mean built on top of it."""
    assert np.isnan(Ordering.auc(np.array([1.0, 2.0]), np.array([0.0, 0.0])))


def test_load(tmp_path: Path):
    """Two members are rank-averaged and joined to the mask BY UID, not by row position — the mask's
    index holds studies the holdout does not, and in a different order."""
    studies = np.array(["b", "a"], dtype="<U64")
    truth = np.array([[1.0] * len(TARGETS), [0.0] * len(TARGETS)], dtype=np.float32)
    for name, column in (("one", [0.2, 0.8]), ("two", [0.3, 0.9])):
        predictions = np.tile(np.array(column, dtype=np.float32).reshape(-1, 1), (1, len(TARGETS)))
        np.savez(tmp_path / f"{name}.holdout.npz", studies=studies, predictions=predictions, truth=truth)
    np.save(tmp_path / "mask_336px_20sl_0.1-0.9band.npy", np.array([[1, 1, 1, 0, 0, 0], [1] * 6, [0] * 6]))
    (tmp_path / "index_336px_20sl_0.1-0.9band.json").write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")

    holdout = Holdout.load(tmp_path, ("one", "two"))

    assert holdout.studies == ["b", "a"]
    assert holdout.presence[0].tolist() == [1] * 6, "row 0 is study b, which is index row 1"
    assert holdout.presence[1].tolist() == [1, 1, 1, 0, 0, 0]
    assert holdout.predictions.shape == (2, len(TARGETS))


def test_patterns():
    """The presence bits read as one stratum id, so two studies scanned alike share a stratum."""
    patterns = _holdout(4).patterns()

    assert patterns[0] == patterns[1], "both fully present"
    assert patterns[0] != patterns[2], "a three-slot study is a different stratum"


def test_fit():
    """The prevalence a study's OWN stratum shows once that study is removed from it."""
    strata = np.array([0, 0, 0, 1])
    estimate = ProtocolPrior(strata).fit(np.array([1.0, 1.0, 0.0, 1.0]))

    assert estimate[0] == 0.5, "the other two of its stratum are one positive, one negative"
    assert estimate[2] == 1.0, "a negative sees both of its neighbours positive"
    assert estimate[3] == 0.75, "a stratum of one has no LOO estimate and takes the global rate"


def test_report():
    """Every label gets one AUC per blend weight plus the prior standing alone, and where the pattern
    IS the label the prior alone separates perfectly."""
    table = _holdout().report()

    assert set(table) == set(TARGETS)
    assert all(len(row) == 7 for row in table.values()), "six blends and the prior alone"
    assert table["ACL"][-1] > 0.9, "the prior alone recovers a label carried entirely by the pattern"
