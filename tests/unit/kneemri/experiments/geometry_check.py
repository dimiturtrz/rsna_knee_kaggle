"""The grouping, against the shape on disk that motivated it: one geometry, many heads."""

import json
from pathlib import Path

from kneemri.experiments.geometry_check import GeometryCheck, GeometryCluster, TrainedGeometry
from kneemri.runs.run_metrics import RunMetrics

EPOCH_SECONDS = 360
"""A twelve-epoch run at this rate costs 1.2 h, which is this project's measured mean per run."""


def write_run(directory: Path, name: str, *, crop_mm: str = "130.0", head: str = "dense", epochs: int = 12) -> Path:
    """A metrics file whose GEOMETRY is `crop_mm` and whose head is not part of the fingerprint.

    Written line by line rather than through `RunMetrics.record`, which reads the CUDA allocator — a unit
    test may touch nothing it did not create, and the device is not ours.
    """
    path = directory / f"{name}.metrics.jsonl"
    recipe = {"variant": "v3base", "crop_mm": crop_mm, "slices": "12", "image_size": "224", "pool": head}
    lines = [json.dumps({"recipe": recipe})]
    lines += [json.dumps({"epoch": epoch, "auc": 0.80, "seconds": EPOCH_SECONDS}) for epoch in range(epochs)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_of(tmp_path: Path) -> None:
    """A run reduces to what its trunk saw, and the head is not part of that."""
    same = [TrainedGeometry.of(RunMetrics(write_run(tmp_path, name, head=name))) for name in ("band_early", "band_mid")]
    assert same[0] is not None
    assert same[1] is not None
    assert same[0].fingerprint == same[1].fingerprint, "the head moved and the pixels did not"
    assert same[0].hours == 12 * EPOCH_SECONDS / 3600

    moved = TrainedGeometry.of(RunMetrics(write_run(tmp_path, "crop160", crop_mm="160.0")))
    assert moved is not None
    assert moved.fingerprint != same[0].fingerprint, "a crop change is a real retrain, not a warm-start candidate"


def test_of_returns_none_without_a_recipe(tmp_path: Path) -> None:
    """A pre-schema run must not become an empty fingerprint that gathers every other one into it."""
    path = tmp_path / "ancient.metrics.jsonl"
    path.write_text(json.dumps({"epoch": 0, "auc": 0.8, "seconds": 1}) + "\n", encoding="utf-8")
    assert TrainedGeometry.of(RunMetrics(path)) is None


def test_hours() -> None:
    """A cluster costs what its runs cost, summed from epochs rather than from a start-to-end span."""
    runs = tuple(TrainedGeometry("r%d" % index, ("g",), 1.5) for index in range(3))
    assert GeometryCluster(runs).hours == 4.5


def test_repeat_hours() -> None:
    """The FIRST run of a geometry is never a repeat — something had to fit it."""
    runs = tuple(TrainedGeometry("r%d" % index, ("g",), 1.5) for index in range(3))
    assert GeometryCluster(runs).repeat_hours == 3.0
    assert GeometryCluster(runs[:1]).repeat_hours == 0.0


def test_line() -> None:
    """The line carries the count, the total, the repeat share and the names — the evidence, not a verdict."""
    pair = (TrainedGeometry("band_early", ("g",), 1.5), TrainedGeometry("band_mid", ("g",), 1.5))
    line = GeometryCluster(pair).line()
    assert "2 runs" in line
    assert "3.0 h" in line
    assert "1.5 h repeat" in line
    assert "band_early, band_mid" in line


def test_clusters(tmp_path: Path) -> None:
    """Only fingerprints trained more than once are reported, costliest first."""
    for name in ("band_early", "band_mid", "band_deep"):
        write_run(tmp_path, name, head=name)
    write_run(tmp_path, "crop160", crop_mm="160.0")

    clusters = GeometryCheck().clusters(tmp_path)
    assert len(clusters) == 1, "the crop160 run trained its geometry once and is not a repeat"
    assert len(clusters[0].runs) == 3
    assert clusters[0].repeat_hours == 2 * 12 * EPOCH_SECONDS / 3600


def test_twins(tmp_path: Path) -> None:
    """The question before queuing an arm — answered without the asking run in its own answer."""
    write_run(tmp_path, "band_early", head="early")
    asked = RunMetrics(write_run(tmp_path, "band_mid", head="mid"))

    twins = GeometryCheck().twins(tmp_path, asked)
    assert [seen.name for seen in twins] == ["band_early"]

    novel = RunMetrics(write_run(tmp_path, "crop160", crop_mm="160.0"))
    assert GeometryCheck().twins(tmp_path, novel) == []
