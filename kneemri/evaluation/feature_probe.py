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
from kneemri.evaluation.slice_readout import READOUTS, SliceMean, SliceReadout

LOGGER = logging.getLogger(__name__)

SELECTION = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
"""The penalties a trunk may be read at. Spans five decades because the trunks do: the best penalty
measured here runs from 0.003 (radnet121) to 10.0 (v3large), and a grid that did not reach both ends
would reproduce the defect it exists to remove."""

INNER_SHARE = 0.5
"""Half the training positions choose the penalty, half fit under it."""

SELECTION_SEED = 0
"""Seeded so two readings of one trunk are comparable. Not a seed experiment — the ban is on averaging
over seeds to manufacture a result, not on making one measurement reproducible."""

TRUNKS: dict[str, str] = {
    "dinov2-small": "features_small_336px_12sl_r336_cls_mean_max",
    "dinov2-large": "features_large_336px_12sl_r252_mean_max",
    "vjepa2-large": "features_vjepa2l_336px_12sl_r256_mean_channels",
    # The controlled pair, added 2026-09-02 for `knee-440m`: `ortholarge` is `v3large` continued on
    # 1.25M knee images and identical to it in every other respect, so their difference here is the only
    # reading this project can take of in-domain SSL continuation without spending the SSL compute. They
    # share a read size with each other, not with the rows above — a probe compares trunks under one
    # instrument, and the instrument is the flag set, so only this pair's own delta is controlled.
    "v3large": "features_v3large_336px_12sl_r256_mean_max",
    "ortholarge": "features_ortholarge_336px_12sl_r256_mean_max",
    # The only VOLUMETRIC trunk in this table (`knee-6hfi`). OrthoDiffusion reads a whole (16, 256, 256)
    # stack through three plane-specific 3D U-Nets, so its row is NOT controlled against the ones above:
    # different read size, different pooling (`gap256` over a volume, not `mean_max` over slices), and a
    # denoising objective rather than a discriminative one. That is why the model's own docstring says a
    # ridge probe here is width-sensitive and a LOSS is confounded with capacity -- only a WIN is clean.
    # Read it against dinov2-large 0.7852 / v3large 0.7446 / ortholarge 0.7387 on that understanding.
    "orthodiff": "orthodiff_images_336px_12sl_r256_gap256_volume",
    # The readout sweep (`experiments/arms/readout_probe.sh`, knee-n647). The rows above hold BOTH
    # poolings at their diluting end -- `mean` over patches, then `max` over slices -- and the three
    # numbers they produced closed two members. These eight move the two axes independently over the
    # same trunks, same studies, same patch grid, so the gap can be attributed to an axis rather than
    # to the pair. `radnet121` joins here and nowhere above because it has no `mean_max` cache: it was
    # closed on a FIT, not a probe, and this is the first reading of it that spends no recipe.
    "dinov2-large@focal-max": "features_large_336px_12sl_r252_cls_mean_focal_max",
    "v3large@focal-max": "features_v3large_336px_12sl_r256_cls_mean_focal_max",
    "ortholarge@focal-max": "features_ortholarge_336px_12sl_r256_cls_mean_focal_max",
    "radnet121@focal-max": "features_radnet121_336px_12sl_r256_cls_mean_focal_max",
    "dinov2-large@focal-attend": "features_large_336px_12sl_r252_cls_mean_focal_attend",
    "v3large@focal-attend": "features_v3large_336px_12sl_r256_cls_mean_focal_attend",
    "ortholarge@focal-attend": "features_ortholarge_336px_12sl_r256_cls_mean_focal_attend",
    "radnet121@focal-attend": "features_radnet121_336px_12sl_r256_cls_mean_focal_attend",
}
"""Trunk name to the cache stem its features were written under. A declaration, not a bucket: every
entry names an array that exists, and a trunk with no cache cannot be probed at all."""

RIDGE = 100.0
"""The probe penalty. Reported alongside a sweep rather than tuned, because a probe tuned per trunk stops
being one instrument applied three times."""

HOLDOUT_FOLD = 0
"""The fold every member in this project is ranked on. A probe on any other split would be a second ruler."""

WITNESS_STUDIES = 8
"""How many studies' reduced features and logits ride along with shipping weights. A served probe rebuilds
the trunk from a mount and re-encodes pixels, so nothing but a stored input-output pair can prove the served
function is the fitted one — the same reason a member carries a witness."""


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

    @classmethod
    def chosen(
        cls,
        features: Float[NDArray[np.float64], "studies dim"],
        truth: Int[NDArray[np.int64], "studies findings"],
        training: Int[NDArray[np.int64], " trained"],
    ) -> float:
        """The penalty this trunk is entitled to, chosen inside the TRAINING positions only.

        A shared penalty is not a fair comparison between trunks. Trunk feature magnitudes differ by
        more than an order of magnitude here (radnet121 tops out at 0.17, v3large at 2.90), and a ridge
        term that is a mild regulariser for one is annihilation for the other. Read at 100 the two sit
        0.108 apart; read at the penalty each one wants they sit 0.017 apart the OTHER way. That is not
        a small correction to a ranking, it is the ranking.

        The choice never touches the caller's holdout, so the reported number stays a held-out number.
        """
        order = np.random.default_rng(SELECTION_SEED).permutation(len(training))
        cut = int(len(training) * INNER_SHARE)
        inner_fit, inner_score = training[order[:cut]], training[order[cut:]]
        areas = {penalty: cls(penalty).inner_area(features, truth, inner_fit, inner_score) for penalty in SELECTION}
        return max(areas, key=lambda penalty: areas[penalty])

    def inner_area(
        self,
        features: Float[NDArray[np.float64], "studies dim"],
        truth: Int[NDArray[np.int64], "studies findings"],
        inner_fit: Int[NDArray[np.int64], " fit"],
        inner_score: Int[NDArray[np.int64], " score"],
    ) -> float:
        """Mean area over the inner scoring positions, skipping labels they left with one class."""
        predictions = self.predicted(features[inner_score], self.weights(features[inner_fit], truth[inner_fit]))
        held = truth[inner_score]
        areas = [
            LabelScores.area_under_curve(predictions[:, column], held[:, column])
            for column in range(held.shape[1])
            if 0 < held[:, column].sum() < len(held)
        ]
        return float(np.mean(areas)) if areas else 0.0

    @staticmethod
    def pooled(
        array: Float[NDArray[np.floating], "studies slots slices dim"],
        readout: SliceReadout | None = None,
    ) -> Float[NDArray[np.float64], "studies width"]:
        """Reduce a trunk's cache to the shape a head is handed, under the given slice readout.

        The pooling is the one thing this instrument imposes on a trunk, which is why it is a strategy
        and why the default is stated rather than assumed: `SliceMean` is what every trunk number on
        the ledger was read with, so leaving it out reproduces those numbers exactly.
        """
        return (readout or SliceMean()).reduce(array)

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

    def holdout_record(
        self,
        features: Float[NDArray[np.float64], "studies dim"],
        truth: Int[NDArray[np.int64], "studies findings"],
        holdout: Int[NDArray[np.int64], " held"],
        training: Int[NDArray[np.int64], " trained"],
    ) -> Float[NDArray[np.float64], "held findings"]:
        """The held-out predictions themselves, in the shape a member's `.holdout.npz` carries.

        A probe number says whether a trunk CAN be read; it says nothing about whether the trunk adds
        anything the pool does not already have. Emitting the predictions lets the same stacking harness
        that judges members judge a probe alongside them, on the studies both were held out on — which
        is the only way to ask the diversity question without first spending a training run.
        """
        return self.predicted(features[holdout], self.weights(features[training], truth[training]))

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
    command.parser.add_argument(
        "--emit-holdout",
        type=Path,
        default=None,
        help="write each probe's held-out predictions as <trunk>_probe.holdout.npz, seatable beside members",
    )
    command.parser.add_argument(
        "--emit-weights",
        type=Path,
        default=None,
        help="write each probe's SHIPPING weights, fitted on every study, plus a witness sample to check a "
        "served probe against",
    )
    command.parser.add_argument(
        "--choose-penalty",
        action="store_true",
        help="give each trunk the penalty it wants, chosen on a nested split inside the training positions",
    )
    command.parser.add_argument(
        "--slice-readout",
        type=str,
        nargs="+",
        default=[SliceMean.name],
        choices=sorted(READOUTS),
        help="how the slice axis is reduced before the ridge",
    )
    arguments = command.parsed()
    root = command.rooted(arguments)
    corpus = ProbeCorpus(processed=root.processed(), raw=root.raw())
    folds = ReportFolds()
    results: list[ProbeResult] = []
    for name in arguments.trunks:
        stem = TRUNKS[name]
        identifiers = corpus.identifiers(stem)
        training, holdout = folds.holdout(folds.assign(str(corpus.raw / "train.csv"), identifiers), HOLDOUT_FOLD)
        truth = corpus.truth(identifiers)
        for readout in arguments.slice_readout:
            features = FeatureProbe.pooled(corpus.features(stem), READOUTS[readout])
            penalties = (
                [FeatureProbe.chosen(features, truth, training)] if arguments.choose_penalty else arguments.penalty
            )
            for penalty in penalties:
                result = FeatureProbe(penalty).scored(f"{name}@{readout}", features, truth, holdout, training)
                results.append(result)
                command.report([f"penalty={penalty:<8} {result.line()}"])
                command.report([f"    {target} {area:.4f}" for target, area in result.per_label.items()])
                if arguments.emit_holdout is not None:
                    arguments.emit_holdout.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        arguments.emit_holdout / f"{name}_{readout}_probe.holdout.npz",
                        studies=np.asarray(identifiers)[holdout],
                        predictions=FeatureProbe(penalty).holdout_record(features, truth, holdout, training),
                        truth=truth[holdout],
                    )
                if arguments.emit_weights is not None:
                    arguments.emit_weights.mkdir(parents=True, exist_ok=True)
                    everything = np.arange(len(identifiers))
                    shipped = FeatureProbe(penalty).weights(features, truth)
                    np.savez(
                        arguments.emit_weights / f"{name}_{readout}_probe.weights.npz",
                        weights=shipped,
                        penalty=np.float64(penalty),
                        readout=readout,
                        trunk=name,
                        stem=stem,
                        witness_studies=np.asarray(identifiers)[everything[:WITNESS_STUDIES]],
                        witness_features=features[everything[:WITNESS_STUDIES]],
                        witness_logits=FeatureProbe.predicted(features[everything[:WITNESS_STUDIES]], shipped),
                    )
    Denominator(len(results), "no trunk had a cached feature array to probe").refuse_empty()


if __name__ == "__main__":
    main()
