"""The pf teacher: a per-finding rank fusion that keeps every CoAtNet study and joins by study."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.findings import STUDY, TARGETS
from kneemri.kaggle.pf_teacher import RAPTOR_SHARE, RAPTOR_SHARE_ELSEWHERE, PfTeacher


def _table(studies: list[str], column: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame(np.repeat(np.array(column)[:, None], len(TARGETS), axis=1), columns=list(TARGETS))
    frame.insert(0, STUDY, studies)
    return frame


def test_blended() -> None:
    """Opposite orders fuse at each finding's own share; a study the reader missed takes the CoAtNet rank."""
    raptor = _table(["a", "b", "c", "d"], [0.1, 0.2, 0.3, 0.4])
    reader = _table(["c", "b", "a"], [0.1, 0.2, 0.3])
    fused = PfTeacher().blended(raptor, reader).set_index(STUDY)

    assert fused.index.tolist() == ["a", "b", "c", "d"]
    share = RAPTOR_SHARE["Synovitis"]
    assert fused.loc["a", "Synovitis"] == pytest.approx(share * 0.25 + (1.0 - share) * 1.0)
    assert fused.loc["d", "Synovitis"] == pytest.approx(1.0)
    other = next(target for target in TARGETS if target not in RAPTOR_SHARE)
    assert fused.loc["c", other] == pytest.approx(RAPTOR_SHARE_ELSEWHERE * 0.75 + RAPTOR_SHARE_ELSEWHERE / 3.0)


def test_readings(tmp_path: Path) -> None:
    """Worker files concatenate to their target columns; a study in two files is refused."""
    first, second = tmp_path / "w0.csv", tmp_path / "w1.csv"
    _table(["a"], [0.1]).assign(extra=1).to_csv(first, index=False)
    _table(["b"], [0.2]).to_csv(second, index=False)
    assert PfTeacher.readings([first, second]).columns.tolist() == [STUDY, *TARGETS]

    _table(["a"], [0.3]).to_csv(second, index=False)
    with pytest.raises(ValueError, match="more than one worker"):
        PfTeacher.readings([first, second])
