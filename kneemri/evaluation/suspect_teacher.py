"""Write a CORRECTED score for the cells an independent signal says the reader got wrong.

`errors_dense` pruned them and lost 0.0158 against `band_dense`. What a prune throws away is the whole
cell: a hard-but-correct study stops contributing any gradient at all, and `errors_dense` measured those
to be worth three to four times a random cell. Re-labelling keeps the cell in the loss and changes only
what it points at, which is the treatment the label-noise literature actually uses.

THE MIXTURE IS THE FIELD'S, NOT OURS. Every method that corrects rather than discards writes
`y_hat = alpha * posterior + (1 - alpha) * original`: Reed 2014 with alpha fixed, M-correction and
DivideMix sampling it per cell from a beta mixture, SELC accumulating it over time, ALR with a momentum
term. BoMD (ICCV 2023) is the one that ablates the weight with the DETECTOR held fixed — 0.2 / 0.6 / 1.0
on one detected set, and 0.6 wins with BOTH extremes losing. No paper we found holds the detector fixed
and ablates alpha = 1 against alpha < 1, so pure posterior is rare rather than refuted; the mechanism
against it is that a wrong correction at full weight injects a mislabel that now looks clean. 0.6 is
therefore the only value in the literature with an ablation under it, and it is what `ALPHA` carries.

IT IS DELIBERATELY NOT MEAN-PRESERVING, and that is the difference from `unread_teacher`. That module
spreads an unread block around the constant it already sits on, because the constant is a measured level
and only ORDER was missing. Here the claim is that the level itself is wrong on these cells — 108
confident false positives against 32 misses over the 58 annotated studies — so a correction that cannot
move the mean cannot express the defect. The per-finding mean shift is REPORTED for exactly that reason:
a prevalence shift is the one thing a twelve-head trunk transmits between heads, so it has to be read
rather than assumed small.

THE TWO TABLES ARE PUT IN ONE SPACE BEFORE THEY ARE MIXED. The label table is written in the reader's
five levels (0.08 / 0.28 / 0.68 / 0.82 / 0.94, or their measured rates under `--calibrated-targets`); the
published pool emits a probability fitted under its own objective. Adding those directly is the mistake
`UnreadTeacher.band` documents from the other side. `expressed` maps each pool score through its own
corpus-wide rank onto the label column's corpus-wide distribution, so a corrected value is always a value
the reader could have written, and the mixture is an interpolation inside one vocabulary rather than
between two.

WHERE THE POSTERIOR COMES FROM. The published pool's twenty checkpoints are out-of-fold for the whole
corpus (`PublishedRuler`), so no cell is corrected by a model that trained on it. Using one of our own
members would be self-distillation through a shared trunk, and it would also be the same trunk the
detector's out-of-fold predictions came from.

WHICH CELLS. The RANKING selector, not cleanlab's mask: on the 108 confident false positives that are
77% of our gold census the mask reads lift 1.63x and the plain quality ranking reads 4.05x at budget 25.
See `kneemri.evaluation.label_errors.DetectedCells.ranked` and knee-jom26.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.data.labels.targets import SUSPECT_STUDY, SUSPECT_TARGET, UNREAD_VALUE
from core.findings import TARGETS
from kneemri.evaluation.published_ruler import PublishedRuler
from kneemri.inference.vote import RankAverage

logger = logging.getLogger(__name__)

STUDY = "StudyInstanceUID"

ALPHA = 0.6
"""How much of the corrected target is the posterior. BoMD's ablated middle; 0.2 and 1.0 both lost."""

RANKABLE = 2
"""The fewest values that carry an order; below it a rank says nothing about where a cell sits."""


@dataclass(frozen=True)
class SuspectTeacher:
    """Corrected scores for the flagged cells, in the label table's own vocabulary."""

    labels: Path
    """The report-derived table the corrections are mixed against and expressed in."""

    published: Path
    """The directory the twenty published checkpoints were unzipped into."""

    flags: Path
    """A (study, target) table naming the cells to correct — `label_errors --budget`'s output."""

    alpha: float = ALPHA
    """The posterior's share of the corrected target."""

    def expressed(self, scores: NDArray[np.float64], levels: NDArray[np.float64]) -> NDArray[np.float64]:
        """Pool scores mapped through their own rank onto the label column's distribution.

        The quantile is taken at the NEAREST observed value rather than interpolated between two, so
        what comes back is one of the reader's own levels. A linear quantile invents values in the gaps
        the reader never writes — 0.155 and 0.33 between 0.08, 0.28 and 0.68 — and then the mixture is
        blending a verdict with something that is not one. Off-vocabulary values are still what a
        corrected cell ENDS at, deliberately: that is the mixture's doing and it is one decision, not
        two. Ties take the average rank, and the rank is read at the midpoint of its bin so the extremes
        are not forced onto the two end levels by an off-by-one.
        """
        if len(scores) < RANKABLE:
            middle = float(np.quantile(levels, 0.5, method="nearest"))
            return np.full(len(scores), middle, dtype=np.float64)
        ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
        return np.asarray(np.quantile(levels, (ranks - 0.5) / len(scores), method="nearest"), dtype=np.float64)

    def table(self) -> pd.DataFrame:
        """One row per flagged cell: the study, the finding, and the corrected score to train it toward."""
        derived = pd.read_csv(self.labels).set_index(STUDY)
        flagged = pd.read_csv(self.flags)
        names, per_seed = PublishedRuler(self.published).per_seed()
        logger.info("%d studies in the published pool, %d flagged cells", len(names), len(flagged))
        position = {name: index for index, name in enumerate(names)}
        corpus = [name for name in names if name in derived.index]
        pooled = RankAverage().combine(per_seed)[[position[name] for name in corpus]].astype(np.float64)
        labels = derived.loc[corpus]
        rows = [
            frame
            for column, target in enumerate(TARGETS)
            if (frame := self._corrected(labels, pooled[:, column], target, flagged)) is not None
        ]
        return (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame({SUSPECT_STUDY: [], SUSPECT_TARGET: [], UNREAD_VALUE: []})
        )

    def _corrected(
        self,
        labels: pd.DataFrame,
        pool: Float[NDArray[np.float64], " corpus"],
        target: str,
        flagged: pd.DataFrame,
    ) -> pd.DataFrame | None:
        """One finding's flagged cells mixed toward the pool's expressed posterior, or None if none is carried.

        `labels` and `pool` are the corpus the pool and the table share, in the pool's order, so a
        flagged study is carried exactly when it is one of their rows.
        """
        asked = flagged.loc[flagged[SUSPECT_TARGET] == target, SUSPECT_STUDY]
        at = {name: index for index, name in enumerate(labels.index)}
        carried = [name for name in asked if name in at]
        if not carried:
            return None
        levels = labels[target].to_numpy(dtype=np.float64)
        posterior = self.expressed(pool, levels)[[at[name] for name in carried]]
        original = labels.loc[carried, target].to_numpy(dtype=np.float64)
        corrected = self.alpha * posterior + (1.0 - self.alpha) * original
        logger.info(
            "%-18s %5d corrected, mean %.4f -> %.4f (%+.4f), corpus mean %.4f",
            target,
            len(carried),
            float(original.mean()),
            float(corrected.mean()),
            float(corrected.mean() - original.mean()),
            float(labels[target].mean()),
        )
        return pd.DataFrame({SUSPECT_STUDY: carried, SUSPECT_TARGET: target, UNREAD_VALUE: corrected})


def main() -> None:
    """Write the corrected scores `--unread-targets` trains toward on the flagged cells."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, required=True, help="the report-derived table to correct")
    command.parser.add_argument("--flags", type=Path, required=True, help="the (study, target) cells to correct")
    command.parser.add_argument("--alpha", type=float, default=ALPHA, help="the posterior's share of the target")
    command.parser.add_argument("--out", type=Path, required=True, help="where the (study, target, value) table goes")
    arguments = command.parsed()

    root = command.rooted(arguments)
    teacher = SuspectTeacher(arguments.labels, root.processed() / "published", arguments.flags, arguments.alpha)
    table = teacher.table()
    table.to_csv(arguments.out, index=False)
    logger.info("%d corrected cells written to %s at alpha %.2f", len(table), arguments.out, arguments.alpha)


if __name__ == "__main__":
    main()
