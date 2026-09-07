"""The column-weight table and the block frame it votes over, on synthetic records.

Every record here is seeded and in memory. What these tests hold is the arithmetic the D63 reading
depends on — that the table is joined to members by NAME rather than by row order, that a uniform table
is the flat rule, and that the frame drops the gold rows and takes truth from the anchor exactly the way
`seat_swap` does. The reading itself (0.9260 uniform, 0.9320 weighted on the 896) is a statement about
real members and lives in the ledger, not here.
"""

from pathlib import Path

import numpy as np
import pytest

from core.findings import TARGETS
from kneemri.evaluation.column_weights import BlockFrame, ColumnWeights
from kneemri.inference.vote import LogitAverage

SEED = 2026
STUDIES = 40
MEMBERS = ("m0", "m1", "m2")
ANCHOR = "anchor"
GOLD_ROWS = 3


def _holdouts(tmp_path: Path) -> Path:
    """Three members and an anchor, all scored against one label table, all serving probabilities."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(SEED)
    graded = generator.choice([0.0, 0.28, 0.5, 0.68, 0.82, 1.0], size=(STUDIES, len(TARGETS)))
    graded[:GOLD_ROWS] = (generator.random((GOLD_ROWS, len(TARGETS))) > 0.5).astype(np.float64)
    truth = graded.astype(np.float32)
    signal = (truth > 0.5).astype(np.float64)
    studies = np.asarray([f"s{index:03d}" for index in range(STUDIES)])
    for name in (*MEMBERS, ANCHOR):
        served = signal * 2.0 - 1.0 + generator.normal(0.0, 0.6, signal.shape)
        np.savez(
            tmp_path / f"{name}.holdout.npz",
            studies=studies,
            predictions=1 / (1 + np.exp(-served)),
            truth=truth,
            epoch=np.asarray(7),
        )
    return tmp_path


def _table(tmp_path: Path, matrix: np.ndarray, members: tuple[str, ...] = MEMBERS) -> Path:
    """A fitted table written the way the fit writes it — member names in a header comment."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "weights.csv"
    path.write_text(f"# rows=members: {','.join(members)}\n")
    with path.open("a") as handle:
        np.savetxt(handle, matrix, delimiter=",")
    return path


def _uniform() -> np.ndarray:
    return np.full((len(MEMBERS), len(TARGETS)), 1 / len(MEMBERS))


def _frame(tmp_path: Path) -> BlockFrame:
    return BlockFrame.over(_holdouts(tmp_path), MEMBERS, ANCHOR)


def test_load(tmp_path: Path) -> None:
    """The rows come back with their names attached, and a table that miscounts them refuses."""
    weights = ColumnWeights.load(_table(tmp_path, _uniform()))

    assert weights.members == MEMBERS
    assert weights.matrix.shape == (len(MEMBERS), len(TARGETS))

    with pytest.raises(ValueError, match="names 4 members"):
        ColumnWeights.load(_table(tmp_path / "wrong", _uniform(), (*MEMBERS, "m3")))


def test_vote(tmp_path: Path) -> None:
    """The join is BY NAME: asking for the members in another order reorders the weights with them.

    The failure this guards has no symptom — a positional table still has the right shape, still sums to
    one down every column, and still moves the score by about what a real seat change moves.
    """
    tilted = _uniform()
    tilted[:, 0] = [1.0, 0.0, 0.0]
    weights = ColumnWeights.load(_table(tmp_path, tilted))

    assert weights.vote(MEMBERS).weights[:, 0] == pytest.approx([1.0, 0.0, 0.0])
    assert weights.vote(MEMBERS[::-1]).weights[:, 0] == pytest.approx([0.0, 0.0, 1.0])

    with pytest.raises(ValueError, match="no weights for"):
        weights.vote(("ghost",))


def test_shrunk(tmp_path: Path) -> None:
    """Half-way back to uniform is half the deviation, and shrinking all the way IS the uniform table."""
    tilted = _uniform()
    tilted[:, 0] = [1.0, 0.0, 0.0]
    weights = ColumnWeights.load(_table(tmp_path, tilted))

    half = weights.shrunk(0.5)

    assert half.members == weights.members, "shrinking a table does not reseat it"
    assert half.matrix[:, 0] == pytest.approx([2 / 3, 1 / 6, 1 / 6])
    assert weights.shrunk(0.0).matrix == pytest.approx(_uniform())


def test_flattened(tmp_path: Path) -> None:
    """A named column goes back to one vote each; the columns beside it are untouched."""
    tilted = _uniform()
    tilted[:, 0] = [1.0, 0.0, 0.0]
    tilted[:, 1] = [0.5, 0.25, 0.25]
    weights = ColumnWeights.load(_table(tmp_path, tilted))

    flat = weights.flattened((TARGETS[0],))

    assert flat.matrix[:, 0] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert flat.matrix[:, 1] == pytest.approx([0.5, 0.25, 0.25]), "only the named column is given back"

    with pytest.raises(ValueError, match="are not findings"):
        weights.flattened(("Sprain",))


def test_written(tmp_path: Path) -> None:
    """A shipped table round-trips: what `written` saves is what `load` reads, names included."""
    tilted = _uniform()
    tilted[:, 2] = [0.7, 0.2, 0.1]
    weights = ColumnWeights.load(_table(tmp_path, tilted)).shrunk(0.5)

    path = tmp_path / "shipped.csv"
    weights.written(path)
    read = ColumnWeights.load(path)

    assert read.members == weights.members
    assert read.matrix == pytest.approx(weights.matrix)


def test_over(tmp_path: Path) -> None:
    """The frame is the shared studies minus the gold rows, and it carries values, not ranks."""
    frame = _frame(tmp_path)

    assert len(frame.studies) == STUDIES - GOLD_ROWS, "the radiologist overrides are dropped"
    assert frame.truth.shape == (STUDIES - GOLD_ROWS, len(TARGETS))
    assert set(frame.predictions) == set(MEMBERS), "the anchor is a seat, never a block member"
    assert np.all((frame.predictions["m0"] > 0) & (frame.predictions["m0"] < 1)), "probabilities, unranked"


def test_macro(tmp_path: Path) -> None:
    """A member that serves the truth outscores one that serves noise, on the ledger's own ruler."""
    frame = _frame(tmp_path)
    generator = np.random.default_rng(SEED)

    informed = frame.macro((frame.truth > 0.5).astype(np.float64) * 0.9 + 0.05)
    blind = frame.macro(generator.random(frame.truth.shape))

    assert informed > blind


def test_kernel(tmp_path: Path) -> None:
    """The anchor carries a third: a block the anchor disagrees with is pulled toward the anchor."""
    frame = _frame(tmp_path)
    block = LogitAverage().combine([frame.predictions[name] for name in MEMBERS])

    assert frame.kernel(block) != frame.macro(block), "a third of the vote is not the block's"
    assert frame.kernel(block) == pytest.approx(frame.kernel(block * 0.5), abs=1e-12), "block ranks, not values"


def test_read(tmp_path: Path) -> None:
    """A uniform table IS the flat rule, so the equality check's delta is zero when nothing is tilted."""
    frame = _frame(tmp_path)
    uniform, weighted = frame.read(ColumnWeights.load(_table(tmp_path, _uniform())))

    assert weighted == pytest.approx(uniform, abs=1e-12)


SERVED = (
    "member,fitted_on,ACL,MCL,MedMen,LatMen,MedOA,LatOA,PFOA,Effusion,Synovitis,Bakers,Contusion,Fracture",
    "child_ff,parent_s2026,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6",
    "own_s2026,own_s2026,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3",
    "orphan_cf,uniform,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1",
)


def test_ported(tmp_path: Path):
    """A served-name fit reads under the names a kernel votes with, and under its ancestors' names.

    The parent keying is what can be CHECKED — only the fold-0 parents have holdouts — and the
    ancestor-less row has nothing to substitute, so it is dropped rather than carried under its own name.
    """
    path = tmp_path / "served.csv"
    path.write_text("# a comment\n" + "\n".join(SERVED) + "\n", encoding="utf-8")

    served = ColumnWeights.ported(path)
    assert served.members == ("child_ff", "own_s2026", "orphan_cf")
    assert served.matrix.shape == (3, len(TARGETS))
    assert served.matrix[0][0] == pytest.approx(0.6)

    parents = ColumnWeights.ported(path, by_parent=True)
    assert parents.members == ("parent_s2026", "own_s2026")
    assert parents.matrix[0][0] == pytest.approx(0.6), "the child's row, under the name that has a holdout"
