from pathlib import Path

from kneemri.experiments.progress import OVERDUE_FACTOR, ArmProgress
from kneemri.runs.run_metrics import EpochRow, RunMetrics

EPOCH_SECONDS = 400
STARTED = 1_700_000_000
PLANNED = 12


def _arm(tmp_path: Path, epochs: int, *, recipe: bool = True) -> ArmProgress:
    # A distinct file per recipe setting: two runs sharing one path append into the same record, so the
    # unrecorded-recipe case would read the recorded one's header and the assertion would pass on it.
    metrics = RunMetrics(tmp_path / f"band_dense{'' if recipe else '_norecipe'}_s2026.pt")
    if recipe:
        metrics.describe({"epochs": str(PLANNED), "slices": "20"})
    for epoch in range(epochs):
        metrics.record(
            EpochRow(
                epoch=epoch,
                loss=1.0,
                auc=0.80 + epoch / 1000,
                seconds=EPOCH_SECONDS,
                per_label={"ACL": 0.8},
                started=STARTED + epoch * EPOCH_SECONDS,
            )
        )
    return ArmProgress(metrics)


def test_planned(tmp_path: Path):
    """The denominator comes from the recorded recipe, and is 0 rather than invented when absent."""
    assert _arm(tmp_path, 3).planned == PLANNED
    assert _arm(tmp_path, 3, recipe=False).planned == 0, "no recipe means no denominator, not a full bar"


def test_producing(tmp_path: Path):
    """A run is producing until one epoch's start plus `OVERDUE_FACTOR` medians has passed."""
    arm = _arm(tmp_path, 4)
    last = STARTED + 3 * EPOCH_SECONDS
    assert arm.producing(last + EPOCH_SECONDS), "one median in, the next epoch is simply still running"
    assert not arm.producing(last + EPOCH_SECONDS * (OVERDUE_FACTOR + 1)), "nothing recorded, so it stopped"


def test_producing_without_a_stamp(tmp_path: Path):
    """A row written before `started` existed cannot claim the card — it is read as stopped."""
    metrics = RunMetrics(tmp_path / "old_s2026.pt")
    metrics.describe({"epochs": str(PLANNED)})
    metrics.record(EpochRow(epoch=0, loss=1.0, auc=0.8, seconds=EPOCH_SECONDS, per_label={"ACL": 0.8}))
    assert not ArmProgress(metrics).producing(STARTED), "an unstamped row says nothing about now"


def test_remaining(tmp_path: Path):
    """What is left is the unrun epochs at the median's cost, and exactly 0 once the run has stopped."""
    arm = _arm(tmp_path, 4)
    now = STARTED + 3 * EPOCH_SECONDS + 1
    assert arm.remaining(now) == (PLANNED - 4) * EPOCH_SECONDS
    assert arm.remaining(now + EPOCH_SECONDS * (OVERDUE_FACTOR + 1)) == 0.0, "a stopped run holds no slot"


def test_line(tmp_path: Path):
    """The line carries state, position and the peak — and no delta against any control."""
    line = _arm(tmp_path, 4).line(STARTED + 3 * EPOCH_SECONDS + 1)
    assert "RUNNING" in line
    assert "4/12 ep" in line
    assert "best ep 3 0.8030" in line
    assert "h left" in line
