"""Whether a label interaction GENERALISES — the verdict half of the residual-coupling gate.

`LabelCoupling` next door says whether structure is left in the residual. It cannot say whether that
structure is usable out of sample, and those are different questions: a correlation matrix is computed on
the same rows it describes. So the verdict is a held-out probe — fit a linear map from all twelve logits
onto one label's truth on half the holdout, score it on the other half against that label's own raw logit.

A linear label-interaction IS the refinement head `knee-w8h6` proposes, with its nonlinearity removed, and
it is the honest floor: if the linear form cannot beat the raw logit out of sample, a wider one is being
asked to find structure this measurement says is not there.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float
from numpy.typing import NDArray

from core.findings import TARGETS
from core.paths import DataRoot
from kneemri.evaluation.residual_coupling import LabelCoupling
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.runs.holdout_record import HoldoutRecord

LOGGER = logging.getLogger(__name__)

PROBABILITY_GUARD = 1e-6
"""Keeps the logit finite. Predictions are sigmoids and reach the ends on confident studies."""

RIDGE = 1.0
"""The probe's L2 penalty.

Not tuned, and deliberately not: the probe exists to answer whether ANY linear label interaction
generalises, so a penalty swept for the best answer would turn the gate into a search. 1.0 against twelve
standardised features and a few hundred rows is the neutral choice — enough to keep the normal equations
conditioned when two labels are near-collinear, far too little to explain a null on its own.
"""


class HeldOutProbe:
    """The refinement head with its nonlinearity removed, scored out of sample.

    THE SPLIT IS BY POSITION PARITY, which is deterministic and needs no seed. A unit test may touch no
    unseeded RNG (`hygiene.small`), and more to the point a gate whose answer moves with a seed is a gate
    that will be re-rolled until it agrees with whoever is reading it.
    """

    def __init__(
        self,
        predictions: Float[NDArray[np.float64], "studies labels"],
        truth: Float[NDArray[np.float32], "studies labels"],
        known: Bool[NDArray[np.bool_], "studies labels"],
        ridge: float = RIDGE,
    ) -> None:
        bounded = np.clip(predictions, PROBABILITY_GUARD, 1 - PROBABILITY_GUARD)
        self.features = np.log(bounded / (1 - bounded))
        self.predictions = predictions
        self.positive = (truth > MIDPOINT).astype(np.int64)
        self.known = known
        self.ridge = ridge
        self.fold = np.arange(len(predictions)) % 2 == 0

    def alone(self, column: int) -> tuple[float, float, int]:
        """The probe's NULL CONTROL: the same machinery given only the label's OWN logit.

        A ridge on one monotone feature is itself monotone, so this reproduces the raw AUC exactly. If it
        does not, the probe's loss on twelve features is arithmetic rather than mechanism — the same
        question `adapt_null` had to answer for `--encoder-weights` before four collapses could be read
        as evidence about an idea instead of about a load path.

        THE ONE CASE IT DOES NOT REPRODUCE is a column scoring BELOW chance on the fitting half: the
        weight is learned, so it comes out negative and the control reports 1 - raw rather than raw. That
        is the control working, not failing — it is reading a column whose logit is anti-correlated with
        its own label — and it cannot be mistaken for the twelve-feature reading, which never inverts.
        """
        return self.label(column, features=np.array([column]))

    def label(self, column: int, features: NDArray[np.int64] | None = None) -> tuple[float, float, int]:
        """One label's raw and refined holdout AUC, plus how many decided studies scored them.

        Both AUCs are taken on the SAME rows — the evaluation half's decided pairs for this label — so
        the difference is the refinement and nothing else. A matched control costs nothing here and is
        the whole reason this probe is worth more than the correlation matrix above it.
        """
        chosen = np.arange(self.features.shape[1]) if features is None else features
        fitting = self.fold & self.known[:, column]
        scoring = ~self.fold & self.known[:, column]
        weights = self.fit(self.features[np.ix_(fitting, chosen)], self.positive[fitting, column])
        refined = self.features[np.ix_(scoring, chosen)] @ weights
        truth = self.positive[scoring, column]
        raw = LabelScores.area_under_curve(self.predictions[scoring, column], truth)
        return raw, LabelScores.area_under_curve(refined, truth), int(scoring.sum())

    def fit(self, features: NDArray[np.float64], truth: NDArray[np.int64]) -> NDArray[np.float64]:
        """Ridge by the normal equations, on centred features with an intercept folded into the mean.

        A closed form rather than an iterative fit because twelve features and a few hundred rows make
        the solve exact and instant, and because an optimiser would add a learning rate and a step count
        to a measurement whose entire value is that it has no knobs to blame a null on.
        """
        centre = features.mean(axis=0)
        centred = features - centre
        gram = centred.T @ centred + self.ridge * np.eye(features.shape[1])
        return np.linalg.solve(gram, centred.T @ (truth - truth.mean()))

    @classmethod
    def report(cls, record: Path, table: Path, ridge: float = RIDGE) -> dict[str, object]:
        """The gate, as one JSON-able reading: the coupling matrix's extremes and the held-out probe.

        The probe owns it because the probe is what returns the VERDICT — the coupling matrix beside it
        is context for reading that verdict, and on its own has never been the thing in question.
        """
        reading = HoldoutRecord(record).read()
        if reading is None:
            message = f"{record} carries no holdout predictions; this analysis has nothing to read"
            raise FileNotFoundError(message)
        known = LabelCoupling.decided(table, reading.studies)
        predictions = reading.predictions.astype(np.float64)
        coupling = LabelCoupling(predictions, reading.truth, known)
        probe = cls(predictions, reading.truth, known, ridge=ridge)
        per_label = {target: probe.label(column) for column, target in enumerate(TARGETS)}
        deltas = [refined - raw for raw, refined, _ in per_label.values() if not np.isnan(raw + refined)]
        controls = [
            probe.alone(column)[1] - probe.alone(column)[0]
            for column in range(len(TARGETS))
            if not np.isnan(sum(probe.alone(column)[:2]))
        ]
        return {
            "record": record.name,
            "epoch": int(reading.epoch),
            "decided_pairs": int(known.sum()),
            "total_pairs": int(known.size),
            "top_truth_coupling": [
                (r.first, r.second, round(r.truth, 4), r.studies)
                for r in sorted(coupling.pairs(), key=lambda r: -abs(r.truth))[:5]
            ],
            "top_residual_coupling": [
                (r.first, r.second, round(r.residual, 4), r.studies) for r in coupling.pairs()[:5]
            ],
            "probe": {
                target: {"raw": round(raw, 4), "refined": round(refined, 4), "studies": studies}
                for target, (raw, refined, studies) in per_label.items()
            },
            "ridge": ridge,
            "null_control_max_abs": round(float(np.max(np.abs(controls))), 6),
            "mean_probe_delta": round(float(np.mean(deltas)), 4),
            "labels_improved": int(sum(delta > 0 for delta in deltas)),
            "labels_scored": len(deltas),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="a run's .holdout.npz, or any of its own names")
    parser.add_argument("--labels", type=Path, default=None, help="the label table carrying the verdicts")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    parser.add_argument(
        "--ridge",
        type=float,
        nargs="+",
        default=[RIDGE],
        help="probe penalties to report. Several exist so a null can be shown not to be the penalty's",
    )
    arguments = parser.parse_args()
    processed = DataRoot.from_config(arguments.paths).processed()
    table = arguments.labels or processed / "labels" / "report_labels_v2.csv"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for ridge in arguments.ridge:
        LOGGER.info(json.dumps(HeldOutProbe.report(arguments.record, table, ridge), indent=2))


if __name__ == "__main__":
    main()
