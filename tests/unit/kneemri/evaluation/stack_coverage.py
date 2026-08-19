"""The depth-band profile, on a corpus laid out by hand so each band's answer is known."""

from pathlib import Path

import numpy as np

from kneemri.evaluation.stack_coverage import StackCoverage, StackCoverageReport
from kneemri.findings import TARGETS

STUDIES = 9


def _corpus(tmp_path: Path) -> tuple[Path, list[str]]:
    """Nine studies whose deepest series holds index+1 slices, so depth order is known."""
    root = tmp_path / "series"
    names = [f"s{index}" for index in range(STUDIES)]
    for index, name in enumerate(names):
        shallow = root / name / "a"
        shallow.mkdir(parents=True)
        for slice_index in range(index + 1):
            (shallow / f"{slice_index}.dcm").write_bytes(b"x")
    return root, names


def test_acquired(tmp_path: Path):
    """The DEEPEST series per study, counted as files, and studies with no directory left out."""
    root, names = _corpus(tmp_path)
    deeper = root / "s0" / "b"
    deeper.mkdir()
    for slice_index in range(20):
        (deeper / f"{slice_index}.dcm").write_bytes(b"x")

    depths = StackCoverage().acquired(root, [*names, "absent"])

    assert depths["s0"] == 20, "the deepest series wins, not the first"
    assert depths["s8"] == STUDIES
    assert "absent" not in depths, "a study with no directory is skipped, not counted as zero"


def test_bands(tmp_path: Path):
    """Equal-sized bands ordered shallowest first, holding row indices into the given order."""
    root, names = _corpus(tmp_path)

    bands = StackCoverage().bands(StackCoverage().acquired(root, names), names, groups=3)

    assert [len(band) for band in bands] == [3, 3, 3]
    assert bands[0] == [0, 1, 2], "the three shallowest studies"
    assert bands[-1] == [6, 7, 8]


def test_trend():
    """Per-band AUCs plus an interval, and a real effect separated from a flat one."""
    generator = np.random.default_rng(3)
    rows = 400
    # Positives ALTERNATE so both bands hold both classes. Splitting truth where the bands split left
    # every band single-class, so every AUC was nan and the assertion compared nan to zero.
    truth = np.zeros((rows, len(TARGETS)), dtype=np.float32)
    truth[::2, :] = 1.0
    predictions = generator.random((rows, len(TARGETS)))
    bands = [list(range(rows // 2)), list(range(rows // 2, rows))]
    # The deep band is made informative on the first finding only; every other finding stays a coin.
    deep = np.array(bands[1])
    predictions[deep, 0] = truth[deep, 0] * 0.5 + generator.random(len(deep)) * 0.1

    measured = StackCoverage().trend(predictions, truth, bands)

    per_band, low, high = measured[TARGETS[0]]
    assert len(per_band) == 2
    assert low > 0.0, "a band that really is better must clear zero"
    assert measured[TARGETS[1]][1] < 0.0 < measured[TARGETS[1]][2], "and a coin must not"


def test_lines():
    """Ordered by trend, starred only on intervals excluding zero, with the chance rate stated."""
    measured = {target: ([0.80, 0.80], -0.05, 0.05) for target in TARGETS}
    measured[TARGETS[0]] = ([0.70, 0.90], 0.10, 0.30)

    lines = StackCoverageReport(measured, [[0], [1]]).lines()
    text = "\n".join(lines)

    assert text.rstrip().endswith("chance alone predicts about 0.6")
    assert "1 of 12 clear zero" in text
    assert lines[-2].startswith(TARGETS[0]), "the largest positive trend sorts last"
    assert "*" in lines[-2]
