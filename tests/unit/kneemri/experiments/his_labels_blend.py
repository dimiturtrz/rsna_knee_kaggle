import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kneemri.experiments.his_labels_blend import FINDINGS, UID, Blend, Folds, main

COLUMNS = [f"f{index}" for index in range(FINDINGS)]
STUDIES = ["train_a", "train_b", "held_a"]
HELD = {"held_a"}
HIS = 0.2
TAUGHT = 1.0
WEIGHT = 0.3
BLENDED = (1 - WEIGHT) * HIS + WEIGHT * TAUGHT


def _his() -> pd.DataFrame:
    values = np.full((len(STUDIES), FINDINGS), HIS)
    return pd.DataFrame(values, columns=COLUMNS).assign(**{UID: STUDIES})[[UID, *COLUMNS]]


def _row(table: pd.DataFrame, uid: str) -> np.ndarray:
    return table[table[UID] == uid][COLUMNS].to_numpy(dtype=float)[0]


@pytest.fixture
def teacher(tmp_path: Path) -> Path:
    path = tmp_path / "teacher.npz"
    np.savez(
        path,
        scores=np.full((len(STUDIES), FINDINGS), TAUGHT),
        identifiers=np.array(STUDIES, dtype=object),
    )
    return path


def test_columns(teacher: Path):
    """His column COUNT is the only thing that makes the positional teacher mapping legal."""
    assert Blend(_his(), teacher, HELD, WEIGHT).columns() == COLUMNS

    with pytest.raises(ValueError, match="expected 12"):
        Blend(_his().drop(columns=["f0"]), teacher, HELD, WEIGHT).columns()


def test_table(teacher: Path):
    """Pin 2: the OOF truth his writer records (:545) must never carry a teacher value.

    If the held-out rows were blended, the arm would be scored on its own teacher's opinion and the
    solo would read high for a reason that has nothing to do with the trunk.
    """
    table, _ = Blend(_his(), teacher, HELD, WEIGHT).table()

    assert _row(table, "held_a") == pytest.approx(np.full(FINDINGS, HIS)), "a held-out row is his raw y"


def test_table_mixes_the_train_rows_at_exactly_the_weight(teacher: Path):
    """BCE is linear in its target, so the table IS the distillation term — at exactly w, not near it."""
    table, _ = Blend(_his(), teacher, HELD, WEIGHT).table()

    assert _row(table, "train_a") == pytest.approx(np.full(FINDINGS, BLENDED))
    assert list(table.columns) == [UID, *COLUMNS], "his layout, or his --labels reads the wrong column"


def test_table_marginal_prices_the_relabel_question(teacher: Path):
    """The marginal decided w = 0.3 over 0.7: a moved positive rate is a relabel, not a distillation."""
    _, marginal = Blend(_his(), teacher, HELD, WEIGHT).table()

    assert list(marginal.index) == COLUMNS
    assert marginal["his_mean"].to_numpy() == pytest.approx(np.full(FINDINGS, HIS))
    assert marginal["blend_mean"].to_numpy() == pytest.approx(np.full(FINDINGS, BLENDED))
    assert marginal["his_pos@.5"].to_numpy() == pytest.approx(np.zeros(FINDINGS)), "his threshold, his rate"


def test_table_refuses_a_teacher_missing_a_train_study(tmp_path: Path):
    """A silent drop would train on a shorter table and print a plausible number for the wrong cell."""
    path = tmp_path / "short.npz"
    np.savez(path, scores=np.full((1, FINDINGS), TAUGHT), identifiers=np.array(["train_a"], dtype=object))

    with pytest.raises(ValueError, match="no teacher row"):
        Blend(_his(), path, HELD, WEIGHT).table()


def test_members(tmp_path: Path):
    """The same predicate his script uses (:393-394) — a different one moves the held-out set."""
    path = tmp_path / "folds.json"
    path.write_text(json.dumps({"folds": {"a": 0, "b": 1, "c": 0}}))

    assert Folds(path).members(0) == {"a", "c"}
    assert Folds(path).members(1) == {"b"}


def test_main(tmp_path: Path, teacher: Path, monkeypatch: pytest.MonkeyPatch):
    """The arm's step 1: his layout, his column order, a new path named for the weight."""
    his = tmp_path / "his.csv"
    _his().to_csv(his, index=False)
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps({"folds": dict.fromkeys(HELD, 0)}))
    out = tmp_path / "nested" / "blend.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "his_labels_blend",
            "--his",
            str(his),
            "--teacher",
            str(teacher),
            "--fold-file",
            str(folds),
            "--weight",
            str(WEIGHT),
            "--out",
            str(out),
        ],
    )

    main()
    written = pd.read_csv(out)

    assert list(written.columns) == [UID, *COLUMNS]
    assert len(written) == len(STUDIES)
    assert _row(written, "train_a") == pytest.approx(np.full(FINDINGS, BLENDED))
    assert _row(written, "held_a") == pytest.approx(np.full(FINDINGS, HIS))
