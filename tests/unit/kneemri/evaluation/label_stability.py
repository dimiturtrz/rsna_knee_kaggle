"""Per-label stability across runs, on constructed profiles whose right answer is known by hand."""

import json
from pathlib import Path

from kneemri.evaluation.label_stability import LabelStability

STRONG = {"ACL": 0.90, "MCL": 0.75, "Fracture": 0.85}
"""One profile with a clear weakest label."""


def test_best(tmp_path: Path):
    """Per-label scores come from the BEST epoch, and a run that never scored one answers None."""
    scored = tmp_path / "a.metrics.jsonl"
    scored.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"epoch": 0, "auc": 0.80, "per_label": {"ACL": 0.1}},
                {"epoch": 1, "auc": 0.84, "per_label": {"ACL": 0.9}},
                {"epoch": 2, "auc": 0.82, "per_label": {"ACL": 0.5}},
            )
        ),
        encoding="utf-8",
    )
    unscored = tmp_path / "b.metrics.jsonl"
    unscored.write_text(json.dumps({"epoch": 0, "auc": None}), encoding="utf-8")

    assert LabelStability().best(scored) == {"ACL": 0.9}
    assert LabelStability().best(unscored) is None


def test_best_reads_only_the_last_run_in_a_re_run_file(tmp_path: Path):
    """A re-run appends, and only the newest run's weights survive — the older peak is not this run's.

    The equivalence class a whole-file maximum gets wrong, and it is on disk: `pair_memory_s2026` peaks
    at 0.8353 in its first run and 0.8155 in its second, so the higher profile belongs to a checkpoint
    that was overwritten. A recipe line is what separates the two runs, so the fixture carries one.
    """
    rerun = tmp_path / "c.metrics.jsonl"
    rerun.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"epoch": 0, "auc": 0.90, "per_label": {"ACL": 0.1}},
                {"recipe": {"fold": 0}},
                {"epoch": 0, "auc": 0.70, "per_label": {"ACL": 0.9}},
            )
        ),
        encoding="utf-8",
    )

    assert LabelStability().best(rerun) == {"ACL": 0.9}, "the LOWER-scoring second run is the one on disk"


def test_correlation():
    """Ranks, not values: a profile shifted by a constant agrees perfectly, a reversed one disagrees.

    The shift case is the one that matters. Two runs half a point apart everywhere order the labels
    identically, and it is the ORDER a per-label plan acts on.
    """
    stability = LabelStability()
    shifted = {label: score + 0.05 for label, score in STRONG.items()}
    reversed_profile = {"ACL": 0.75, "MCL": 0.90, "Fracture": 0.85}

    assert stability.correlation(STRONG, shifted) == 1.0
    assert stability.correlation(STRONG, reversed_profile) == -1.0


def test_worst():
    """Each run's weakest label, which is the column that says whether one label owns the bottom."""
    worst = LabelStability().worst({"a": STRONG, "b": {"ACL": 0.5, "MCL": 0.9, "Fracture": 0.8}})

    assert worst == {"a": "MCL", "b": "ACL"}


def test_report():
    """Both correlations, labelled so the within-fold one cannot be read as evidence about the model."""
    lines = LabelStability().report({"a": STRONG, "b": STRONG}, {"c": {"ACL": 0.88, "MCL": 0.70, "Fracture": 0.80}})

    assert "within one holdout (2 recipes)" in lines[0]
    assert "not evidence" in lines[0], "the within-fold number must carry its own caveat"
    assert "across holdouts" in lines[1]
    assert any("worst in c: MCL" in line for line in lines)
