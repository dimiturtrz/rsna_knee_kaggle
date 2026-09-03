"""Does adapting on the TEST set's own features buy anything? `knee-tta1`.

Every lane this project has closed — seating, seat count, vote rule, calibration, foreign weight —
recombines models fitted on our 4407 studies. None of them touch the hidden test set. But this is a
CODE competition: the kernel is handed the test studies at inference, and internet-off does not stop
us fitting on them. That makes transductive adaptation the one axis the ledger has never measured
(grepped 2026-09-03: no pseudo-label, self-training or test-time-adaptation row exists).

WHAT IS AND IS NOT WORTH SIMULATING. A monotone per-label rescaling of the logits — calibration,
prevalence matching, distribution alignment — cannot change a per-label AUC at all, because AUC reads
only the ORDER. So the whole family of "align the outputs to the test set" ideas is arithmetically
dead before it is run, and this module does not implement it. Only something that changes the RANKING
can pay, which means refitting the head on test FEATURES under labels we do not have.

THE SUBSTITUTE FOR THE LABELS WE DO NOT HAVE. Fit on the training positions, predict the holdout, and
treat the most confident of those predictions as if they were truth. Refit including them. If the
features carry structure the training fit missed, the refit finds it; if the pseudo-labels merely echo
the first fit, the refit reproduces it and the delta is zero. That zero is a real answer and the
cheapest form of it available.

WHY THE HOLDOUT STANDS IN FOR THE TEST SET. The 909 fold-0 studies were withheld from the training
positions, so a head fitted on those positions has never seen them — the same relation the shipped
model has to the hidden test set. The simulation is therefore honest about the ADAPTATION even though
it cannot be honest about the distribution shift: our holdout is drawn from the same corpus as our
training data, and the real test set may not be. This bounds the gain from adaptation ALONE, with
shift set to zero. A published gain under shift does not transfer here, and a gain measured here is
the floor the kernel would see, not the ceiling.

WHAT THIS IS NOT. A probe head is not a member head. The ridge read here is weaker than any shipped
member, so the ABSOLUTE areas are not comparable to the ledger's member scores and must never be
entered beside them. Only the DELTA between the fitted and the adapted read is the measurement, and it
is a statement about the adaptation, not about the trunk.

THE MEASUREMENT, 2026-09-03. Eight cells, two trunks by four tail sizes, and every one is NEGATIVE:

    dinov2-large  0.05 -0.0036 | 0.1 -0.0026 | 0.25 -0.0020 | 0.4 -0.0029
    ortholarge    0.05 -0.0055 | 0.1 -0.0038 | 0.25 -0.0033 | 0.4 -0.0030

The SHAPE contradicts this module's own pre-registration below, which predicted a small tail would be
safe and a large one harmful. The smallest tail is the WORST on both trunks and the damage shrinks as
the tail grows — the opposite ordering. That is a signature of the LABELLING scheme rather than of
adaptation: `pseudo_labels` forces exactly `share` of studies positive per finding regardless of the
finding's true prevalence, so a small tail distorts the implied prior hardest.

WHY THIS IS NOT A CLOSURE, and the axis stays open. Two properties of the harness could produce this
result even if transductive adaptation were sound, and both are real rather than hypothetical:

  (1) THE HEAD IS LINEAR. Pseudo-labels read off a ridge fit lie along that fit's own decision
      direction by construction, so refitting a linear model on them cannot discover structure — it can
      only re-import the first fit's errors with extra weight. Linear-on-linear is close to circular,
      and a negative delta is what circularity looks like when the tails are noisy.
  (2) SHIFT IS ZERO BY CONSTRUCTION. The holdout is drawn from the same corpus as the training rows.
      Self-training pays UNDER distribution shift; this simulation has none, by the same design that
      makes it honest about the withheld relation.

So the eight cells bound naive self-training with a linear head under zero shift. They do not price
test-time adaptation with a member head under the real test set's shift, which is the case the kernel
would actually be in. Recorded as a measured negative on the cheapest version of the axis, NOT as a
closed lane, and no submission slot was spent on it.
"""

import logging
from dataclasses import dataclass

import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray

from core import cli
from core.data.corpus.folds import ReportFolds
from core.denominator import Denominator
from core.findings import TARGETS
from kneemri.evaluation.feature_probe import HOLDOUT_FOLD, TRUNKS, FeatureProbe, ProbeCorpus
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.slice_readout import READOUTS, SliceMean

LOGGER = logging.getLogger(__name__)

CONFIDENT_SHARE = 0.25
"""How much of each tail of the test predictions becomes a pseudo-label, per finding.

A quarter at each end labels half the studies and leaves the middle — where the head is least sure and
most wrong — out of the refit entirely. Stated rather than tuned: tuning this share on the same holdout
it is scored on would manufacture the gain the module exists to test for."""

SHARES = (0.05, 0.1, 0.25, 0.4)
"""The tail sizes reported. A sweep because the mechanism's own prediction is a SHAPE — a small tail
should be safe and weak, a large one should import the head's own errors — and one point cannot show
whether that shape is there."""


@dataclass(frozen=True)
class AdaptationResult:
    """One trunk's fitted read, its adapted read, and the difference between them."""

    trunk: str
    share: float
    studies: int
    fitted: dict[str, float]
    adapted: dict[str, float]

    @property
    def delta(self) -> float:
        """Mean area gained by adapting, over the twelve findings. The whole measurement."""
        return float(np.mean(list(self.adapted.values())) - np.mean(list(self.fitted.values())))

    def line(self) -> str:
        """One printable row, so a caller never formats a float and a name in two places."""
        fitted, adapted = np.mean(list(self.fitted.values())), np.mean(list(self.adapted.values()))
        return (
            f"{self.trunk:<26} share={self.share:<5} n={self.studies:<5} "
            f"fitted={fitted:.4f} adapted={adapted:.4f} delta={self.delta:+.4f}"
        )


class TestAdaptation:
    """Refit a probe head on the test set's own confident predictions, and score what it changed."""

    def __init__(self, penalty: float, share: float = CONFIDENT_SHARE) -> None:
        self.probe = FeatureProbe(penalty)
        self.share = share

    def pseudo_labels(
        self,
        predictions: Float[NDArray[np.float64], "held findings"],
    ) -> tuple[Int[NDArray[np.int64], "held findings"], Float[NDArray[np.float64], "held findings"]]:
        """Per finding, the confident tails of the test predictions as labels, and a 0/1 mask of which count.

        Confidence is read as RANK within the test set rather than as a distance from a threshold,
        because the probe's logits are unnormalised ridge outputs whose scale differs per finding — a
        fixed cutoff would label every study for one finding and none for another.
        """
        labels = np.zeros_like(predictions, dtype=np.int64)
        weights = np.zeros_like(predictions, dtype=np.float64)
        tail = max(1, int(len(predictions) * self.share))
        for column in range(predictions.shape[1]):
            order = np.argsort(predictions[:, column])
            labels[order[-tail:], column] = 1
            weights[order[:tail], column] = 1.0
            weights[order[-tail:], column] = 1.0
        return labels, weights

    def adapted(
        self,
        trunk: str,
        features: Float[NDArray[np.float64], "studies dim"],
        truth: Int[NDArray[np.int64], "studies findings"],
        holdout: Int[NDArray[np.int64], " held"],
        training: Int[NDArray[np.int64], " trained"],
    ) -> AdaptationResult:
        """Fit on training, pseudo-label the holdout, refit on both, and score both reads on the truth.

        The refit uses the pseudo-labelled holdout studies ALONGSIDE the real training rows rather than
        instead of them. Replacing the real labels would not be adaptation, it would be discarding the
        supervision we have for supervision we invented.
        """
        first = self.probe.weights(features[training], truth[training])
        predictions = self.probe.predicted(features[holdout], first)
        labels, weights = self.pseudo_labels(predictions)
        confident = weights.max(axis=1) > 0
        design = np.vstack([features[training], features[holdout][confident]])
        targets = np.vstack([truth[training], labels[confident]])
        second = self.probe.weights(design, targets)
        return AdaptationResult(
            trunk=trunk,
            share=self.share,
            studies=len(holdout),
            fitted=self.areas(predictions, truth[holdout]),
            adapted=self.areas(self.probe.predicted(features[holdout], second), truth[holdout]),
        )

    @staticmethod
    def areas(
        predictions: Float[NDArray[np.float64], "held findings"],
        held: Int[NDArray[np.int64], "held findings"],
    ) -> dict[str, float]:
        """One area per finding, in the fixed target order."""
        return {
            target: LabelScores.area_under_curve(predictions[:, column], held[:, column])
            for column, target in enumerate(TARGETS)
        }


def main() -> None:
    """Measure what refitting on the test set's own confident predictions changes, per trunk."""
    command = cli.Command(__doc__, LOGGER)
    command.parser.add_argument("--trunks", type=str, nargs="+", default=sorted(TRUNKS), help="trunk names to adapt")
    command.parser.add_argument("--penalty", type=float, default=None, help="ridge penalty; default is per-trunk")
    command.parser.add_argument("--shares", type=float, nargs="+", default=list(SHARES), help="confident tail sizes")
    arguments = command.parsed()
    root = command.rooted(arguments)
    corpus = ProbeCorpus(processed=root.processed(), raw=root.raw())
    folds = ReportFolds()
    results: list[AdaptationResult] = []
    for name in arguments.trunks:
        stem = TRUNKS[name]
        identifiers = corpus.identifiers(stem)
        training, holdout = folds.holdout(folds.assign(str(corpus.raw / "train.csv"), identifiers), HOLDOUT_FOLD)
        truth = corpus.truth(identifiers)
        features = FeatureProbe.pooled(corpus.features(stem), READOUTS[SliceMean.name])
        penalty = arguments.penalty or FeatureProbe.chosen(features, truth, training)
        for share in arguments.shares:
            result = TestAdaptation(penalty, share).adapted(name, features, truth, holdout, training)
            results.append(result)
            command.report([result.line()])
    Denominator(len(results), "no trunk had a cached feature array to adapt on").refuse_empty()


if __name__ == "__main__":
    main()
