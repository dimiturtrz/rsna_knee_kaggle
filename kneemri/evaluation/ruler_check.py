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

`python -m kneemri.evaluation.ruler_check --checkpoints a.pt b.pt --labels full.csv --confident confident.csv`
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from jaxtyping import Bool, Float

from core.paths import DataRoot
from kneemri.corpus.folds import ReportFolds
from kneemri.device import VisibleCards
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.member_readings import MemberReadings
from kneemri.inference.members import Member
from kneemri.inference.prediction_cache import PredictionCache
from kneemri.labels.targets import Targets

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
    parser.add_argument("--overlap", action="store_true", help="every window start, not just disjoint ones")
    parser.add_argument("--refresh", action="store_true", help="recompute rather than reuse stored predictions")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = DataRoot.from_config(arguments.paths)
    # EACH MEMBER'S OWN READING, never one cache and one size for all of them. This CLI loaded whatever
    # cache its flags named, rebuilt every checkpoint at the canonical six slots, passed IMAGE_SIZE as
    # the read size, windowed by GROUP and predicted in the fp16 default — so a whole-stack member was
    # scored three slices at a time, a 448 px member could not be scored at all and a bf16 member was
    # ranked as a function it was never fitted as. `member_curve` carried the identical defect and was
    # fixed; this is the other half of the same question, and it is the tool that decides whether the
    # local ledger can rank recipes AT ALL — a wrong reading here does not lose a member, it invalidates
    # the verdict about the ruler.
    readings = MemberReadings([Member.from_checkpoint(path) for path in arguments.checkpoints], root.processed())

    # The split is taken from the FIRST cache and reused, so every member is scored on the same studies
    # in the same order; any other cache that disagrees about that order is refused rather than silently
    # misaligned against the truth vector.
    corpus, identifiers = readings.reference()
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    _, holdout = folds.holdout(folds.assign(str(root.raw() / "train.csv"), identifiers), arguments.fold)

    train_table = root.raw() / "train.csv"
    truth = Targets.load(arguments.labels, train_table, identifiers).scores[holdout]
    # Weight zero is exactly "the reader did not read this", which is what has to be dropped rather
    # than scored — see the module docstring for why scoring it as a negative is worse than not
    # filtering at all.
    kept = Targets.load(arguments.confident, train_table, identifiers).weights[holdout] > 0

    device = VisibleCards().device()
    predictions = HoldoutPredictions(device)

    # The SAME store `member_curve` writes, keyed on the studies scored, so asking the ruler question
    # about members already on the curve costs no encoder at all.
    store = PredictionCache(
        root.processed() / "holdout_predictions",
        [identifiers[position] for position in holdout],
        predictions.views,
        refresh=arguments.refresh,
    )

    scored: dict[str, Float[np.ndarray, "b o"]] = {}
    for member, slab, held in readings.slabs(holdout, identifiers, corpus):
        window = member.window(held.slices, overlap=arguments.overlap)
        scored[member.file.stem] = predictions.cached(member, slab, store, window)
        logger.info("scored %s", member.file.stem)

    check = RulerCheck()
    for line in check.report(check.under(scored, truth), check.under(scored, truth, kept), float(kept.mean())):
        logger.info("%s", line)


if __name__ == "__main__":
    main()
