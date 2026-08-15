"""How tightly a per-label delta can be read when both arms saw the SAME studies.

`python -m kneemri.paired_floor`

knee-3vh states that every per-label claim this project has quoted was read bare, and that calibrating
them against each label's own unpaired error bar is too conservative to read a knob by — two arms scored
on the same 881 studies share their sampling error almost entirely, so the DIFFERENCE has a far tighter
interval than either AUC does. It puts the tightening at "about eleven times", which nobody measured.

This measures it, on the published out-of-fold predictions: four seeds over five folds whose id-sets
partition all 4,407 studies, so every study carries several predictions from models that did not train
on it. No GPU, no labels of ours, no training.

THE PAIR USED IS A NULL PAIR, and that is the point rather than a limitation. Two seeds of one recipe
differ by nothing that should show up per label, so the interval measured across them is the
instrument's own width — which is exactly what a floor is. This is a measurement OF the ruler using
seed twins as a null, not a seed experiment: nothing here trains anything or proposes to.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from core.paths import DataRoot
from kneemri.label_power import LabelPower
from kneemri.model import TARGETS
from kneemri.oof_oracle import OutOfFoldOracle
from kneemri.scores import MIDPOINT, RESAMPLES, LabelScores
from kneemri.targets import Targets

logger = logging.getLogger(__name__)

CONFIDENCE = 1.96
"""Standard normal at 95%, matching what `run_comparison` already multiplies the unpaired error by."""

FOLD_STUDIES = 881
"""Our own ruler's size, so the corpus-scale number can be quoted at the scale arms are actually read."""

SUBSET_SEED = 2026
"""Fixed, so the fold-scale row is the one anyone else recomputes."""


class PairedFloor:
    """The width of a per-label delta's interval, paired against unpaired, on the same predictions."""

    def __init__(self, power: LabelPower, scores: LabelScores | None = None) -> None:
        # Injected rather than constructed here: the unpaired bound is `LabelPower`'s formula and there
        # must be exactly one copy of it, or the comparison is between two spellings of the same thing.
        self.power = power
        self.scores = scores or LabelScores()

    def paired(
        self,
        first: NDArray[np.float64],
        second: NDArray[np.float64],
        targets: NDArray[np.float32],
        resamples: int = RESAMPLES,
    ) -> dict[str, float]:
        """Each label's paired 95% interval WIDTH, by bootstrap over the studies both arms scored."""
        interval = self.scores.paired_interval(first, second, targets, resamples)
        return {target: high - low for target, (low, high) in interval.items()}

    def unpaired(self, targets: NDArray[np.float32]) -> dict[str, float]:
        """Each label's unpaired 95% width — the bound `run_comparison` prints today.

        Computed from the positive counts actually present in these studies rather than from a corpus
        table, so the two sides of the comparison are measured on identical data.
        """
        studies = targets.shape[0]
        counted = [int(value) for value in (targets > MIDPOINT).sum(axis=0)]
        return {
            target: 2 * CONFIDENCE * self.power.standard_error(positives, studies - positives)
            for target, positives in zip(TARGETS, counted, strict=True)
        }

    def tightening(self, paired: dict[str, float], unpaired: dict[str, float]) -> dict[str, float]:
        """How many times tighter the paired interval is, per label. The bead's claim is about eleven."""
        return {target: unpaired[target] / paired[target] if paired.get(target) else float("nan") for target in TARGETS}

    def mean_width(self, widths: dict[str, float]) -> float:
        """The mean width over labels that are defined at all."""
        return float(np.nanmean(list(widths.values())))

    def subset(self, studies: int, total: int) -> NDArray[np.int64]:
        """A seeded subset of study rows, so the floor can be quoted at our own ruler's size.

        Without replacement: this asks what a SMALLER holdout would show, and drawing with replacement
        would answer a different question by putting the same study in twice.
        """
        generator = np.random.default_rng(SUBSET_SEED)
        return generator.choice(total, size=min(studies, total), replace=False)

    def report(
        self,
        first: NDArray[np.float64],
        second: NDArray[np.float64],
        targets: NDArray[np.float32],
        resamples: int = RESAMPLES,
    ) -> dict[str, float]:
        """Log both widths and their ratio per label, and hand back the paired widths."""
        paired = self.paired(first, second, targets, resamples)
        unpaired = self.unpaired(targets)
        tighter = self.tightening(paired, unpaired)
        logger.info("%-18s %9s %9s %8s", "label", "paired", "unpaired", "tighter")
        for target in TARGETS:
            logger.info("%-18s %9.4f %9.4f %7.1fx", target, paired[target], unpaired[target], tighter[target])
        finite = [value for value in tighter.values() if np.isfinite(value)]
        logger.info(
            "%-18s %9.4f %9.4f %7.1fx",
            "MEAN",
            self.mean_width(paired),
            self.mean_width(unpaired),
            float(np.mean(finite)),
        )
        return paired


def main() -> None:
    """Measure the paired floor on the published out-of-fold predictions, at two holdout sizes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = DataRoot.from_config(arguments.paths)
    identifiers, per_seed = OutOfFoldOracle().corpus(root.processed() / "published")
    truth = Targets.load(arguments.labels, root.raw() / "train.csv", identifiers).scores
    floor = PairedFloor(LabelPower(arguments.labels))

    # A NULL PAIR: two seeds of one recipe, so the true per-label difference is zero and the interval
    # measured is the instrument's own width rather than a mixture of width and effect.
    first, second = per_seed[0], per_seed[1]

    logger.info("corpus: %d studies, %d resamples", len(identifiers), arguments.resamples)
    floor.report(first, second, truth, arguments.resamples)

    drawn = floor.subset(FOLD_STUDIES, len(identifiers))
    logger.info("")
    logger.info("our ruler's size: %d studies, %d resamples", len(drawn), arguments.resamples)
    floor.report(first[drawn], second[drawn], truth[drawn], arguments.resamples)


if __name__ == "__main__":
    main()
