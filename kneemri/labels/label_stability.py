"""Which weak labels are reproducible and which are the holdout's noise: `python -m kneemri.labels.label_stability`.

Every run writes a per-label AUC for every epoch, and nothing has ever read them ACROSS runs. That is
the only cheap way to separate the two things a low label score can mean.

THE QUESTION IT SETTLES. "Track per-label AUC and treat the worst label as the work item" was this
project's plan, and it was refuted — Fracture reads 0.784, 0.903 and 0.864 on three folds under an
IDENTICAL recipe, because an AUC from 62 positives has a standard error near 0.04 (`label_power`). But
the refutation was applied to per-label work in general, and it does not hold there: MCL is the worst
label in every run on every fold, which no amount of sampling error explains.

HOW TO READ THE TWO NUMBERS. Recipes sharing ONE fold also share that fold's sampling error, so their
per-label profiles agree almost by construction — a high within-fold correlation says nothing. Only the
ACROSS-fold correlation is evidence about the model, and it is the smaller of the two.
"""

import argparse
import itertools
import json
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

THROUGHPUT = "throughput"
"""Speed probes write metrics too, and they never score a label."""


class LabelStability:
    """Per-label scores from many runs, and how much their ORDER survives a change of holdout."""

    def best(self, metrics: Path) -> dict[str, float] | None:
        """The per-label scores from the run's best epoch, or None if it never scored one."""
        rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
        scored = [row for row in rows if row.get("auc") is not None and row.get("per_label")]
        return max(scored, key=lambda row: row["auc"])["per_label"] if scored else None

    def correlation(self, first: dict[str, float], second: dict[str, float]) -> float:
        """Spearman rank correlation between two runs' per-label profiles.

        Ranks rather than values because the question is which labels are WORST, not what they score:
        two runs half a point apart everywhere agree perfectly about the ordering, and it is the
        ordering that a per-label plan acts on.
        """
        labels = sorted(set(first) & set(second))
        ranked = (self._ranked([first[label] for label in labels]), self._ranked([second[label] for label in labels]))
        return float(np.corrcoef(*ranked)[0, 1])

    def _ranked(self, values: list[float]) -> NDArray[np.float64]:
        return np.argsort(np.argsort(values)).astype(np.float64)

    def worst(self, runs: dict[str, dict[str, float]]) -> dict[str, str]:
        """Each run's weakest label — the column that says whether one label owns the bottom."""
        return {name: min(scores, key=lambda label: scores[label]) for name, scores in runs.items()}

    def report(self, shared: dict[str, dict[str, float]], others: dict[str, dict[str, float]]) -> list[str]:
        """Within-holdout agreement, across-holdout agreement, and who is bottom in each run.

        `shared` are runs on one fold and `others` are runs on different ones; passing them apart is the
        whole measurement, since only the across pairs carry evidence about the model rather than about
        the studies that happened to land in one holdout.
        """
        within = [self.correlation(a, b) for a, b in itertools.combinations(shared.values(), 2)]
        across = [self.correlation(a, b) for a, b in itertools.product(shared.values(), others.values())]
        lines = [
            f"within one holdout ({len(shared)} recipes): median rank correlation {np.median(within):+.2f}, "
            f"n={len(within)} pairs — they share the fold's sampling error, so this is a floor, not evidence",
            f"across holdouts: median rank correlation {np.median(across):+.2f}, n={len(across)} pairs",
        ]
        lines.extend(f"  worst in {name}: {label}" for name, label in self.worst({**shared, **others}).items())
        return lines


def main() -> None:
    """Compare per-label profiles across every run that scored, split by which fold it held out."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--other-folds", nargs="+", default=["wholestack_fold1", "wholestack_fold2"])
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from core.paths import DataRoot  # noqa: PLC0415 - CLI-only, and this module is imported as a library

    stability = LabelStability()
    processed = DataRoot.from_config(arguments.paths).processed()
    runs: dict[str, dict[str, float]] = {}
    for metrics in sorted(processed.glob("*.metrics.jsonl")):
        name = metrics.name.removesuffix(".metrics.jsonl")
        scores = None if THROUGHPUT in name else stability.best(metrics)
        if scores:
            runs[name] = scores

    others = {name: scores for name, scores in runs.items() if name in set(arguments.other_folds)}
    shared = {name: scores for name, scores in runs.items() if name not in others}
    for line in stability.report(shared, others):
        logger.info(line)


if __name__ == "__main__":
    main()
