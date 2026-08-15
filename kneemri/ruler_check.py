"""Does the choice of evaluation ruler change WHICH recipe wins?

Every score this project reports is agreement with LLM-read report labels, because `train.csv` carries
the twelve radiologist annotations for exactly 58 studies — 1.32% of the label cells, all-or-nothing.
The competition scores the annotations. So the local number is a proxy, and the question that decides
how much to trust it is not "how big is the gap" (a proxy can be biased and still rank recipes
correctly) but **"does a lower-noise ruler put the recipes in a different order?"**

If the order holds, the proxy is a fine instrument for choosing what to submit and the gap is a
constant we can ignore. If the order flips, then some of what this project has "measured" is a property
of the label reader rather than of the model, and the local ledger cannot be used to pick a recipe.

The lower-noise ruler is the same table with the reader's own UNKNOWN calls **dropped** — not scored as
negatives. That distinction is the whole reason `LabelScores.per_label` takes a selection: a pair the
reader could not read sits at the 0.5 default, which binarises to zero, so scoring a filtered table
without dropping those pairs would assert that every unknown finding was absent and produce a ruler
that is more biased than the one it replaces, not less.

`python -m kneemri.ruler_check --checkpoints a.pt b.pt --labels full.csv --confident confident.csv`
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Bool, Float
from transformers import AutoModel

from core.paths import DataRoot
from kneemri.backbone import ENCODERS, SMALL
from kneemri.cache_flags import CacheFlags
from kneemri.folds import ReportFolds
from kneemri.holdout_predictions import GROUP, HoldoutPredictions, Window
from kneemri.scores import LabelScores
from kneemri.slot_reader import IMAGE_SIZE
from kneemri.slots import SLOTS
from kneemri.targets import Targets

logger = logging.getLogger(__name__)


class RulerCheck:
    """Scores each checkpoint under two rulers and reports whether they agree on the ordering."""

    def __init__(self, scores: LabelScores | None = None) -> None:
        self.scores = scores or LabelScores()

    def under(
        self,
        predictions: dict[str, Float[np.ndarray, "b o"]],
        truth: Float[np.ndarray, "b o"],
        where: Bool[np.ndarray, "b o"] | None = None,
    ) -> dict[str, float]:
        """Each checkpoint's mean AUC under one ruler."""
        return {
            name: float(np.nanmean(list(self.scores.per_label(scored, truth, where).values())))
            for name, scored in predictions.items()
        }

    def report(self, full: dict[str, float], confident: dict[str, float], kept: float) -> list[str]:
        """The two rulers side by side, ordered by the full-table score, and whether they disagree.

        The ordering is what matters, so it is stated as a verdict rather than left to be eyeballed
        out of a table.
        """
        lines = [f"confident ruler keeps {kept:.1%} of the (study, finding) pairs", ""]
        lines.append(f"{'checkpoint':34s} {'full':>8s} {'confident':>10s} {'delta':>8s}")
        for name, value in sorted(full.items(), key=lambda item: -item[1]):
            lines.append(f"{name:34s} {value:8.4f} {confident[name]:10.4f} {confident[name] - value:+8.4f}")

        by_full = [name for name, _ in sorted(full.items(), key=lambda item: -item[1])]
        by_confident = [name for name, _ in sorted(confident.items(), key=lambda item: -item[1])]
        lines.append("")
        if by_full == by_confident:
            lines.append("RULERS AGREE on the ordering — the proxy can be trusted to CHOOSE a recipe")
        else:
            lines.append("RULERS DISAGREE — the local ledger is ranking label noise, not recipes")
            lines.append(f"  full:      {' > '.join(by_full)}")
            lines.append(f"  confident: {' > '.join(by_confident)}")
        return lines


def main() -> None:
    """Score every checkpoint on its shared holdout under both rulers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="members sharing ONE split")
    parser.add_argument("--labels", type=Path, required=True, help="the full report-derived table")
    parser.add_argument("--confident", type=Path, required=True, help="the same table with UNKNOWN calls struck")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=2026)
    CacheFlags.add_to(parser)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = DataRoot.from_config(arguments.paths)
    corpus, identifiers = CacheFlags.cache(arguments, root.processed()).load()
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    _, holdout = folds.holdout(folds.assign(str(root.raw() / "train.csv"), identifiers), arguments.fold)

    train_table = root.raw() / "train.csv"
    truth = Targets.load(arguments.labels, train_table, identifiers).scores[holdout]
    # Weight zero is exactly "the reader did not read this", which is what has to be dropped rather
    # than scored — see the module docstring for why scoring it as a negative is worse than not
    # filtering at all.
    kept = Targets.load(arguments.confident, train_table, identifiers).weights[holdout] > 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = HoldoutPredictions(device)
    starts = list(range(arguments.slices - GROUP + 1))
    images, mask = corpus.images[holdout], corpus.mask[holdout]

    scored: dict[str, Float[np.ndarray, "b o"]] = {}
    for checkpoint in arguments.checkpoints:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        encoder = ENCODERS[str(saved.get("config", {}).get("variant", SMALL))]
        model = predictions.load(checkpoint, AutoModel.from_pretrained(encoder.repository), encoder.dim, len(SLOTS))
        scored[checkpoint.stem] = predictions.predict(
            model, images, mask, Window(starts=starts, width=GROUP, size=IMAGE_SIZE)
        )
        logger.info("scored %s", checkpoint.stem)

    check = RulerCheck()
    for line in check.report(check.under(scored, truth), check.under(scored, truth, kept), float(kept.mean())):
        logger.info("%s", line)


if __name__ == "__main__":
    main()
