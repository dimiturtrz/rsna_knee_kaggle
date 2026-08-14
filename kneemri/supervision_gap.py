"""Is a finding held back by its LABELS or by the MODEL? `python -m kneemri.supervision_gap`.

Every local number in this project scores agreement with LLM-read report labels, while the competition
scores agreement with radiologist annotations — and where the report says YES the radiologist agrees
only 65.3% of the time. A third of the positives we train on are not positives.

That fact alone says the labels are noisy. It does NOT say what to do about it, because the answer is
different for different findings, and this module is the measurement that separates them. Two numbers
per finding against the SAME radiologist truth, on the studies train.csv annotates:

    REFERENCE — the report label itself, used as a predictor. How much signal the supervision carries.
    ACHIEVED  — what a model fitted on those labels actually reaches.

"CEILING" IS THE WRONG WORD FOR THE REFERENCE and this module deliberately avoids it: a model fitted on
noisy labels CAN beat them by looking at the image, and measured on the published members it does so on
half the findings. Effusion's report label predicts the radiologist at 0.8137 and the model reaches
0.9453 — it is correcting the reader. The reference is a comparison, never a bound.

WHAT THE SPLIT DECIDES. Where the model already beats the report, the reader is the weak part and a
better label reader is the lever. Where the report describes a finding well and the model cannot
reproduce it — ACL at 0.9841 against 0.8640, Lateral Meniscus at 0.8429 against 0.6497 — the signal is
in the label and the model is failing to extract it from the pixels; no label work can help those, and
showing the model more of the image might.

IT NEEDS PREDICTIONS THAT DID NOT TRAIN ON THE ANNOTATIONS. Ours are fitted with the annotated studies
upweighted, so scoring them here would be scoring training data. The published members are fitted on
`report_labels_v2.csv` with no annotation weighting and carry out-of-fold predictions for every study,
so the radiologist labels are genuinely unseen by them. See `kneemri.oof_oracle`.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float

from kneemri.model import TARGETS
from kneemri.scores import LabelScores

logger = logging.getLogger(__name__)

PRESENT = 0.5
"""Where a graded value becomes a positive, matching `LabelScores.per_label`'s own binarisation."""


class SupervisionGap:
    """The labels' own predictive power beside the model's, per finding."""

    def __init__(self, scores: LabelScores | None = None) -> None:
        self.scores = scores or LabelScores()

    def annotated(self, train_table: Path) -> tuple[list[str], Float[np.ndarray, "studies targets"]]:
        """The studies a radiologist annotated on EVERY finding, and those annotations.

        All-or-nothing rather than per-cell: train.csv annotates a study on all twelve or none, so a
        partial row would be a study whose columns mean different things, and there are none to
        scavenge. Returned in table order so a caller can align predictions to it.
        """
        table = pd.read_csv(train_table).set_index("StudyInstanceUID")
        columns = [target for target in TARGETS if target in table.columns]
        complete = table[columns].dropna(how="any")
        return list(complete.index), complete.to_numpy(dtype=np.float32)

    def against(
        self, values: Float[np.ndarray, "studies targets"], truth: Float[np.ndarray, "studies targets"]
    ) -> dict[str, float]:
        """One AUC per finding, of anything shaped like predictions against the annotations.

        The same method serves both columns of the report — the labels used as a predictor and the
        model's probabilities — because they are the same measurement and giving them separate methods
        would let the two drift into meaning different things.
        """
        return {
            target: self.scores.area_under_curve(
                values[:, column].astype(np.float64), (truth[:, column] > PRESENT).astype(np.int64)
            )
            for column, target in enumerate(TARGETS[: values.shape[1]])
        }

    def report(self, reference: dict[str, float], achieved: dict[str, float]) -> list[str]:
        """Every finding ordered by what holds it back, with the dividing line drawn explicitly.

        Sorted by the gap rather than by either number, because the QUESTION is which side is weaker —
        an ordering by score would put the findings we do well beside the findings whose labels are
        easy, and those are not the same group.
        """
        shared = [target for target in reference if target in achieved]
        gaps = sorted(shared, key=lambda target: achieved[target] - reference[target], reverse=True)
        lines = [
            f"aggregate: labels {np.nanmean([reference[t] for t in shared]):.4f}   "
            f"model {np.nanmean([achieved[t] for t in shared]):.4f}",
            f"{'finding':20s} {'labels':>8s} {'model':>8s} {'model-labels':>13s}",
        ]
        divided = False
        for target in gaps:
            delta = achieved[target] - reference[target]
            if delta < 0 and not divided:
                lines.append("  --- above: the model beats its own labels, so the READER is the weak part ---")
                divided = True
            lines.append(f"{target:20s} {reference[target]:8.4f} {achieved[target]:8.4f} {delta:+13.4f}")
        return lines


def main() -> None:
    """Split the findings by whether their labels or their model is the weaker half."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="the report-derived table to price")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from core.paths import DataRoot  # noqa: PLC0415 - CLI-only
    from kneemri.ensemble import Ensemble  # noqa: PLC0415 - CLI-only
    from kneemri.oof_oracle import OutOfFoldOracle  # noqa: PLC0415 - CLI-only

    root = DataRoot.from_config(arguments.paths)
    gap = SupervisionGap()
    identifiers, truth = gap.annotated(root.raw() / "train.csv")

    names, per_seed = OutOfFoldOracle().corpus(root.processed() / "published")
    combined = Ensemble.combine(per_seed)
    position = {name: index for index, name in enumerate(names)}
    keep = [index for index, name in enumerate(identifiers) if name in position]
    rows = np.array([position[identifiers[index]] for index in keep])
    logger.info("%d annotated studies, %d of them with a held-out prediction", len(identifiers), len(keep))

    # THE REPORT TABLE READ RAW. `Targets.load` overwrites the derived label with the annotation
    # wherever a study has one, which is right for training and would make this measurement compare the
    # annotations against themselves — a perfect score, and a meaningless one.
    table = pd.read_csv(arguments.labels).set_index("StudyInstanceUID").reindex([identifiers[i] for i in keep])
    columns = [target for target in TARGETS if target in table.columns]
    values = np.asarray(np.nan_to_num(table[columns].to_numpy(dtype=np.float32), nan=PRESENT), dtype=np.float32)

    annotations = truth[keep]
    for line in gap.report(gap.against(values, annotations), gap.against(combined[rows], annotations)):
        logger.info(line)


if __name__ == "__main__":
    main()
