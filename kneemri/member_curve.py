"""What does each additional ensemble member actually buy?

`python -m kneemri.member_curve --checkpoints a.pt b.pt c.pt --labels labels.csv`

This question was answered once already and the answer was void. The members were trained at different
model seeds, the fold split was derived from that seed, and so each member was scored on a holdout that
was training data for the others — which reported an ensemble of five at 0.8807 when its own members
logged 0.83. The split is now a property of the corpus (`--split-seed`), so members that differ only in
initialisation share one holdout and the comparison means something.

The curve is the deciding number for two open questions at once. It says whether our fifth member earns
its forward passes at all, and the Efficiency Prize track turns entirely on that: teams scoring exactly
our 0.888 sit at efficiency ranks 29-50, while the best score on that board sits at 638, so a cheap
submission at nearly our accuracy is worth far more there than a costly one.

Every subset of each size is scored, not one arbitrary subset, because with three or five members the
spread between the best and worst subset of a given size is the honest error bar on the mean.
"""

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray
from transformers import AutoModel

from core.paths import DataRoot
from kneemri.ensemble import Ensemble
from kneemri.folds import ReportFolds
from kneemri.holdout_predictions import GROUP, HoldoutPredictions
from kneemri.scores import LabelScores
from kneemri.slot_reader import IMAGE_SIZE
from kneemri.slots import SLOTS
from kneemri.study_cache import StudyCache
from kneemri.targets import Targets

DINOV2 = "facebook/dinov2-small"
DIM = 384

logger = logging.getLogger(__name__)


class MemberCurve:
    """Mean score against ensemble size, over every subset of that size."""

    def __init__(self, scores: LabelScores) -> None:
        self.scores = scores

    def sizes(self, per_member: list[NDArray[np.float64]], truth: Float[np.ndarray, "b t"]) -> dict[int, list[float]]:
        """Every subset's score, grouped by how many members it holds.

        All subsets rather than a sample: with five members that is 31 combinations and a few seconds of
        numpy, and the spread within a size is the only error bar available without retraining.
        """
        by_size: dict[int, list[float]] = {}
        for size in range(1, len(per_member) + 1):
            by_size[size] = [
                self.scores.mean(Ensemble.combine([per_member[index] for index in subset]), truth)
                for subset in itertools.combinations(range(len(per_member)), size)
            ]
        return by_size

    def report(self, by_size: dict[int, list[float]]) -> list[str]:
        """One line per ensemble size: its mean, its range, and what the member added."""
        lines = []
        previous = None
        for size, values in sorted(by_size.items()):
            mean = float(np.mean(values))
            gain = "" if previous is None else f"   {mean - previous:+.4f} over {size - 1}"
            lines.append(f"{size} member(s): mean {mean:.4f}   [{min(values):.4f}, {max(values):.4f}]{gain}")
            previous = mean
        return lines


def main() -> None:
    """Score every subset of the given checkpoints on their shared holdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="members sharing ONE split")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--slices", type=int, default=12)
    parser.add_argument("--overlap", action="store_true", help="every window start, not just disjoint ones")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = DataRoot.from_config(arguments.paths)
    corpus, identifiers = StudyCache(root.processed(), slices=arguments.slices).load()
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    _, holdout = folds.holdout(folds.assign(str(root.raw() / "train.csv"), identifiers), arguments.fold)
    truth = Targets.load(arguments.labels, root.raw() / "train.csv", identifiers).scores[holdout]

    predictions = HoldoutPredictions(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    starts = list(range(arguments.slices - GROUP + 1)) if arguments.overlap else [i * GROUP for i in range(4)]
    images, mask = corpus.images[holdout], corpus.mask[holdout]
    scores = LabelScores()
    per_member = []
    for checkpoint in arguments.checkpoints:
        member = predictions.load(checkpoint, AutoModel.from_pretrained(DINOV2), DIM, len(SLOTS))
        scored = predictions.predict(member, images, mask, starts, IMAGE_SIZE)
        per_member.append(scored)
        logger.info("%s alone: %.4f", checkpoint.name, scores.mean(scored, truth))

    curve = MemberCurve(scores)
    for line in curve.report(curve.sizes(per_member, truth)):
        logger.info(line)


if __name__ == "__main__":
    main()
