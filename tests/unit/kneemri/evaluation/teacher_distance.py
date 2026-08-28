"""Distance is read over the studies two members SHARE, in the row order each of them stored."""

from pathlib import Path

import numpy as np

from kneemri.evaluation.teacher_distance import TeacherDistance, TeacherReadings
from kneemri.teacher_file import TeacherFile

TARGETS = 12


def written(path: Path, studies: list[str], column: list[float]) -> Path:
    """A teacher file whose every target column carries the same per-study values."""
    scores = np.tile(np.array(column, dtype=np.float32)[:, None], (1, TARGETS))
    TeacherFile.write(path, studies, scores)
    return path


def test_read(tmp_path: Path) -> None:
    """A teacher file comes back under the names its writer stored it by, named for its file."""
    readings = TeacherReadings.read(written(tmp_path / "v4.npz", ["s1", "s0"], [1.0, 0.0]))
    assert readings.name == "v4"
    assert readings.identifiers == ("s1", "s0")
    assert readings.scores.shape == (2, TARGETS)


def test_aligned(tmp_path: Path) -> None:
    """Rows are REORDERED onto the asked-for studies, because a sharded pass need not store them in step.

    The equivalence class that matters: a member whose file is a permutation of another's must align to
    the same values, or the permutation itself reads as disagreement.
    """
    readings = TeacherReadings.read(written(tmp_path / "v4.npz", ["s1", "s0"], [1.0, 0.0]))
    assert readings.aligned(("s0", "s1"))[:, 0].tolist() == [0.0, 1.0]


def test_loaded(tmp_path: Path) -> None:
    """Every teacher file in the directory is read, and nothing else in it is."""
    written(tmp_path / "a.npz", ["s0"], [0.0])
    written(tmp_path / "b.npz", ["s0"], [1.0])
    (tmp_path / "notes.txt").write_text("not a teacher file")

    assert len(TeacherDistance.loaded(tmp_path).members) == 2


def test_shared(tmp_path: Path) -> None:
    """The shared set is the INTERSECTION, so a member capped for a smoke cannot silence the others."""
    written(tmp_path / "a.npz", ["s0", "s1", "s2"], [0.0, 1.0, 2.0])
    written(tmp_path / "b.npz", ["s1", "s2"], [1.0, 2.0])

    assert TeacherDistance.loaded(tmp_path).shared() == ("s1", "s2")


def test_pairs(tmp_path: Path) -> None:
    """Two members ordering the studies oppositely are the most distant pairing, and agree perfectly with
    a copy of themselves — the two boundaries of the ruler, read in one go."""
    written(tmp_path / "up.npz", ["s0", "s1", "s2"], [0.0, 1.0, 2.0])
    written(tmp_path / "same.npz", ["s0", "s1", "s2"], [0.1, 0.2, 0.3])
    written(tmp_path / "down.npz", ["s0", "s1", "s2"], [2.0, 1.0, 0.0])

    pairs = TeacherDistance.loaded(tmp_path).pairs()
    assert len(pairs) == 3
    assert pairs[0][0] < pairs[-1][0]
    assert {pairs[-1][1], pairs[-1][2]} == {"same", "up"}


def test_nearest(tmp_path: Path) -> None:
    """A member is judged by its CLOSEST neighbour: two twins and a stranger put the stranger first."""
    written(tmp_path / "twin_a.npz", ["s0", "s1", "s2"], [0.0, 1.0, 2.0])
    written(tmp_path / "twin_b.npz", ["s0", "s1", "s2"], [0.1, 0.2, 0.3])
    written(tmp_path / "stranger.npz", ["s0", "s1", "s2"], [2.0, 1.0, 0.0])

    assert TeacherDistance.loaded(tmp_path).nearest()[0][1] == "stranger"


def test_report(tmp_path: Path) -> None:
    """The report states the denominator it measured on, and carries a row for every pairing."""
    written(tmp_path / "a.npz", ["s0", "s1"], [0.0, 1.0])
    written(tmp_path / "b.npz", ["s0", "s1"], [1.0, 0.0])

    lines = TeacherDistance.loaded(tmp_path).report()
    assert "2 members over 2 shared studies" in lines
    assert sum(1 for line in lines if "  x  " in line) == 1
