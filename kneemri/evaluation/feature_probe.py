"""How much of each trunk's answer is in its FROZEN features — the head removed. `knee-gd62`.

Every member score this project owns is a trunk and a head fitted together, so "DINOv2 is our best
encoder" has never been measured as a statement about DINOv2. It is a statement about one trunk under one
recipe, and `2026-08-30_the-non-dino-lane-is-quality-bound.md` reads the other families as worse without
being able to say whether the FEATURES are worse or the RECIPE was.

This separates them. A linear probe on features that are already cached, over the same fold-0 split every
member is ranked on, scored with the same `LabelScores`. No training run, no card: the encoders ran in
August and their outputs are on disk.

WHAT AN ANSWER MEANS. If the probe ordering matches the member ordering, the non-DINO trunks are genuinely
weaker representations and the lane stays closed. If a trunk probes WELL and serves BADLY, its features
carry signal our recipe is failing to extract, and that re-opens the trunk lane as a RECIPE question —
which is a different and much cheaper thing to fix than an architecture search.

A LINEAR PROBE IS A FLOOR, NEVER A CEILING. It bounds what a linear read of the pooled features gives, and
a trunk whose signal lives in a non-linear or un-pooled arrangement reads low here for a reason that is
about this instrument. That is why the comparison is between trunks under one probe, never between a probe
number and a member number.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float, Int
from numpy.typing import NDArray

from core import cli
from core.data.corpus.folds import ReportFolds
from core.denominator import Denominator
from core.findings import TARGETS
from kneemri.evaluation.scores import MIDPOINT, LabelScores

LOGGER = logging.getLogger(__name__)

TRUNKS: dict[str, str] = {
    "dinov2-small": "features_small_336px_12sl_r336_cls_mean_max",
    "dinov2-large": "features_large_336px_12sl_r252_mean_max",
    "vjepa2-large": "features_vjepa2l_336px_12sl_r256_mean_channels",
}
"""Trunk name to the cache stem its features were written under. A declaration, not a bucket: every
entry names an array that exists, and a trunk with no cache cannot be probed at all."""

RIDGE = 100.0
"""The probe penalty. Reported alongside a sweep rather than tuned, because a probe tuned per trunk stops
being one instrument applied three times."""

HOLDOUT_FOLD = 0
"""The fold every member in this project is ranked on. A probe on any other split would be a second ruler."""


@dataclass(frozen=True)
class ProbeResult:
    """One trunk's head-free reading: the per-finding areas and the mean over them."""

    trunk: str
    dimensions: int
    studies: int
    per_label: dict[str, float]

    @property
    def mean(self) -> float:
        """The mean over the twelve findings — the number comparable across trunks."""
        return float(np.mean(list(self.per_label.values())))

    def line(self) -> str:
        """One printable row, so a caller never formats a float and a name in two places."""
        return f"{self.trunk:<14} dim={self.dimensions:<5} n={self.studies:<5} mean={self.mean:.4f}"


class FeatureProbe:
    """A ridge read of pooled frozen features against the twelve findings, on the fold-0 split."""

    def __init__(self, penalty: float = RIDGE) -> None:
        self.penalty = penalty

    @staticmethod
    def pooled(
        array: Float[NDArray[np.floating], "studies slots slices dim"],
    ) -> Float[NDArray[np.float64], "studies dim"]:
        """Average a trunk's cache over slots and slices, which is the shape a head is handed.

        The pooling is the one thing this instrument imposes on a trunk. It is the same reduction every
        slot-pooling head starts from, so it is the fairest single choice; it is also why a trunk whose
        signal is positional reads low here.
        """
        return np.asarray(array, dtype=np.float64).mean(axis=(1, 2))

    def weights(
        self,
        features: Float[NDArray[np.float64], "studies dim"],
        truth: Int[NDArray[np.int64], "studies findings"],
    ) -> Float[NDArray[np.float64], "dim1 findings"]:
        """Closed-form ridge weights, with a bias column appended rather than an intercept fitted.

        Closed form rather than an iterative solver because the pooled width is at most 1024 and the
        normal equations are exact — an optimiser here would add a convergence question this measurement
        does not want to have to answer.
        """
        design = np.hstack([features, np.ones((len(features), 1))])
        gram = design.T @ design
        penalty = self.penalty * np.eye(len(gram))
        penalty[-1, -1] = 0.0
        solved = np.linalg.solve(gram + penalty, design.T @ truth.astype(np.float64))
        return np.asarray(solved, dtype=np.float64)

    @staticmethod
    def predicted(
        features: Float[NDArray[np.float64], "studies dim"],
        weights: Float[NDArray[np.float64], "dim1 findings"],
    ) -> Float[NDArray[np.float64], "studies findings"]:
        """Apply probe weights to held-out features."""
        return np.hstack([features, np.ones((len(features), 1))]) @ weights

    def scored(
        self,
        trunk: str,
        features: Float[NDArray[np.float64], "studies dim"],
        truth: Int[NDArray[np.int64], "studies findings"],
        holdout: Int[NDArray[np.int64], " held"],
        training: Int[NDArray[np.int64], " trained"],
    ) -> ProbeResult:
        """Fit on the training positions and score the holdout, one area per finding."""
        weights = self.weights(features[training], truth[training])
        predictions = self.predicted(features[holdout], weights)
        areas = {
            target: LabelScores.area_under_curve(predictions[:, column], truth[holdout, column])
            for column, target in enumerate(TARGETS)
        }
        return ProbeResult(trunk=trunk, dimensions=features.shape[1], studies=len(holdout), per_label=areas)


@dataclass(frozen=True)
class ProbeCorpus:
    """The inputs a probe is run over: a cache directory, a label table and the fold source."""

    processed: Path
    raw: Path

    def truth(self, identifiers: list[str]) -> Int[NDArray[np.int64], "studies findings"]:
        """Binarised report labels in corpus order, aligned to the feature index by study id."""
        table = pd.read_csv(self.processed / "labels" / "report_labels_v2.csv").set_index("StudyInstanceUID")
        aligned = table.reindex(identifiers)
        return (aligned[list(TARGETS)].to_numpy(dtype=np.float64) > MIDPOINT).astype(np.int64)

    def identifiers(self, stem: str) -> list[str]:
        """The study order a feature cache was written in."""
        index = self.processed / f"features_index_{stem.removeprefix('features_')}.json"
        return list(json.loads(index.read_text(encoding="utf-8")))

    def features(self, stem: str) -> Float[NDArray[np.floating], "studies slots slices dim"]:
        """Memory-map a feature cache rather than reading 500 MB per trunk."""
        return np.load(self.processed / f"{stem}.npy", mmap_mode="r")


def main() -> None:
    """Probe every cached trunk against the twelve findings on the fold-0 holdout."""
    command = cli.Command(__doc__, LOGGER)
    command.parser.add_argument("--penalty", type=float, nargs="+", default=[RIDGE], help="ridge penalties to report")
    command.parser.add_argument("--trunks", type=str, nargs="+", default=sorted(TRUNKS), help="trunk names to probe")
    arguments = command.parsed()
    root = command.rooted(arguments)
    corpus = ProbeCorpus(processed=root.processed(), raw=root.raw())
    folds = ReportFolds()
    results: list[ProbeResult] = []
    for name in arguments.trunks:
        stem = TRUNKS[name]
        identifiers = corpus.identifiers(stem)
        training, holdout = folds.holdout(folds.assign(str(corpus.raw / "train.csv"), identifiers), HOLDOUT_FOLD)
        features = FeatureProbe.pooled(corpus.features(stem))
        truth = corpus.truth(identifiers)
        for penalty in arguments.penalty:
            result = FeatureProbe(penalty).scored(name, features, truth, holdout, training)
            results.append(result)
            command.report([f"penalty={penalty:<8} {result.line()}"])
            command.report([f"    {target} {area:.4f}" for target, area in result.per_label.items()])
    Denominator(len(results), "no trunk had a cached feature array to probe").refuse_empty()


if __name__ == "__main__":
    main()
