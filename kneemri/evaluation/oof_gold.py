"""The honest 58-study gold read, stitched from a family's five folds.

    python -m kneemri.evaluation.oof_gold --twins 'oof_fold[0-4]_s2026.holdout.npz'
    python -m kneemri.evaluation.oof_gold --twins 'oof_reader_fold[0-4]_s2026.holdout.npz' \
        --against 'oof_fold[0-4]_s2026.holdout.npz'

WHY IT EXISTS. Until 2026-09-10 every gold number this project quoted was a SINGLE-FOLD read: a member
holds out fold 0, thirteen of the fifty-eight radiologist-annotated studies land in that holdout, and the
other forty-five sat in its training table at their report-derived values. `--hold-annotations` does not
change that — it skips only the block that OVERWRITES those cells with the radiologist value at
`ANNOTATED_WEIGHT` (`options.py:314` -> `targets.py:453`), so the STUDY is trained on either way and only
the VALUE is withheld. So a single-fold gold macro is 78% in-sample, and the honest 22% of it is thirteen
studies carrying an unpaired standard deviation of 0.034 (D296 §3).

WHAT THE STITCH BUYS. The five fold holdouts PARTITION the corpus — 909 + 875 + 884 + 849 + 890 = 4407,
disjoint, union the whole thing — and the gold studies fall across them 13 / 8 / 16 / 11 / 10 = 58. So a
recipe run on all five folds yields one prediction per study from a member that never saw it, and the
gold read over that stitch is fifty-eight HONEST studies rather than thirteen. The price is five arms
instead of one; there is no cheaper route, and no amount of record-mining substitutes for it, because a
holdout record contains only the studies its member held OUT.

WHICH ESTIMATOR DECIDES, AND WHICH ONE THE BOARD USES — THEY ARE NOT THE SAME ONE. `gold_estimators.py`
carries the family and the argument for it: the registered internal ruler is the per-finding
RANK-NORMALISED pooled cell AUC at paired sd 0.0096, the raw pooled reading is disqualified for being
movable 0.0288 by a recalibration that changes no ranking, and Hanley weighting is rejected for breaking
the pairing.

D298 then read the competition's own evaluation page — text, verbatim, `Final Score = (1/12) Σ AUC_i`,
"the macro-averaged AUC ROC" — so the BOARD'S statistic is the flat macro over twelve at equal weight.
Every other estimator here is an INTERNAL INSTRUMENT: sharper on our 58 studies and a different quantity
from the one the competition scores. So the ledger's board-comparable row stays `macro` or it stops being
comparable, and `rankpooled` decides only internal questions — which label table supervises better, which
of two arms to keep — where resolution is what is scarce. That also makes pair-count weighting point AWAY
from the board rather than merely being less sharp: Fracture at three positives is worth exactly what
Effusion is worth to the scorer, and any pair-count weight up-weights the populated columns the board
weights identically.

The same reading also promotes the raw-pooling disqualification from a semantic argument to the scorer's
definition: AUC is rank-only per column and the columns never meet, so the entire between-finding
calibration axis is worth exactly zero on the board.

Every line printed here names its estimator, because the same records read 0.8927 flat-macro and 0.8645
rank-norm-pooled — 0.028 apart on ONE system, wider than any decision band they would be quoted into
(knee-f9aao).

WHAT IT REFUSES. A stitch that does not cover the corpus. Folds that overlap are refused upstream by
`OutOfFoldTeacher.readings`; a stitch that covers too FEW studies is refused here, because a four-fold
glob silently drops a fifth of the gold set and still returns a plausible macro.

WHAT IT CANNOT RECOVER. The fold-to-fold gold spread, 0.0550 on one recipe and one seed. D297 tried
every estimator on it and only the disqualified one halves it — for the same reason it is disqualified.
That spread is the RULER, not the macro: ten gold studies cannot read a model under any weighting, and
the answer to a thin gold read is more studies, never a cleverer statistic.

WHAT IT CANNOT SETTLE. Whether the gold annotations are right. They are the only truth in this project
not produced by a reader, which is why they are the ruler — not evidence that they are without error.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Int
from numpy.typing import NDArray

from core import cli
from core.bootstrap import StudyBootstrap
from core.findings import TARGETS
from kneemri.evaluation.gold_estimators import ESTIMATORS, Cells, GoldEstimator
from kneemri.evaluation.gold_rows import PRESENT, GoldRows
from kneemri.evaluation.oof_teacher import OutOfFoldTeacher

logger = logging.getLogger(__name__)

CORPUS_STUDIES = 4407
"""Every study the five folds partition. A stitch below this is missing a fold, not merely small."""

GOLD_STUDIES = 58
"""The studies a radiologist read on every finding. Quoted so a silent alignment loss cannot pass."""

TWO_SIDED = 1.96
"""Normal two-sided 95% multiplier, used only to turn a measured sd into a readable half-width."""


@dataclass(frozen=True)
class Stitch:
    """One family's out-of-fold predictions, aligned onto the gold studies."""

    family: str
    cells: Cells
    covered: int

    def positives(self) -> dict[str, int]:
        """Positives per finding inside the gold set — the term that makes a flat macro fragile."""
        return {target: int(self.cells.labels[:, index].sum()) for index, target in enumerate(TARGETS)}


class GoldRuler:
    """A family's honest gold reading, and how tightly that reading can be taken.

    It owns the resample stream and the alignment onto the gold studies; what a reading MEANS belongs to
    the estimator it is taken with, which is why every number below is handed to one rather than computed
    here.
    """

    def __init__(self, bootstrap: StudyBootstrap | None = None) -> None:
        self.bootstrap = bootstrap or StudyBootstrap()
        self.gold = GoldRows()

    def stitch(self, family: str, parts: tuple[Path, ...], train_table: Path) -> Stitch:
        """The family's out-of-fold predictions for exactly the gold studies, in the gold table's order.

        Refuses a stitch that does not cover the corpus: a four-fold glob drops a fifth of the gold set
        and still returns a plausible macro, which is the failure this check exists for.
        """
        names, scores = OutOfFoldTeacher(parts=parts).readings()
        if len(names) < CORPUS_STUDIES:
            message = (
                f"{family} stitches {len(names)} studies of {CORPUS_STUDIES} — a fold is missing, and a "
                "gold read over a partial stitch silently drops gold studies"
            )
            raise ValueError(message)
        identifiers, annotations = self.gold.annotated(train_table)
        placed = {name: row for row, name in enumerate(names)}
        found = [(placed[identifier], gold) for gold, identifier in enumerate(identifiers) if identifier in placed]
        rows = [row for row, _ in found]
        golds = [gold for _, gold in found]
        return Stitch(
            family=family,
            cells=Cells.of(scores[rows].astype(np.float64), (annotations[golds] >= PRESENT).astype(np.int64)),
            covered=len(rows),
        )

    def draws(self, stitch: Stitch) -> list[Int[NDArray[np.int64], " draw"]]:
        """This stitch's study resamples, drawn once so two estimators read the same ones."""
        return list(self.bootstrap.draws(stitch.covered))

    def spread(self, stitch: Stitch, estimator: GoldEstimator) -> tuple[float, float]:
        """One estimator's reading of this family, and the sd of that reading over study resamples."""
        return estimator.spread(stitch.cells, self.draws(stitch))

    def paired(self, stitch: Stitch, against: Stitch, estimator: GoldEstimator) -> tuple[float, float]:
        """The difference between two families under one estimator, and the sd of the DIFFERENCE."""
        return estimator.difference(stitch.cells, against.cells, self.draws(stitch))

    def report(self, stitch: Stitch, against: Stitch | None = None) -> list[str]:
        """Every estimator's level and resolution, and the paired difference when a control is given."""
        lines = [
            f"{stitch.family}: {stitch.covered} of {GOLD_STUDIES} gold studies, out of fold by construction",
            "",
        ]
        for estimator in ESTIMATORS:
            level, deviation = self.spread(stitch, estimator)
            lines.append(f"  {estimator.name:<10} {level:.4f}   sd {deviation:.4f}   +/-{TWO_SIDED * deviation:.4f}")
        lines += ["", "positives inside the gold set, per finding:"]
        lines += [f"  {target:<18} {count}" for target, count in stitch.positives().items()]
        if against is None:
            return lines
        lines += ["", f"PAIRED against {against.family} — the difference and ITS sd, which is the ruler:"]
        for estimator in ESTIMATORS:
            difference, deviation = self.paired(stitch, against, estimator)
            lines.append(
                f"  {estimator.name:<10} {difference:+.4f}   sd {deviation:.4f}   "
                f"resolves {TWO_SIDED * deviation:.4f} at two sd"
            )
        return lines


def main() -> None:  # devtools-ignore: test-mirror
    """Read one family's honest gold level, and optionally its paired difference from a control."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--models", type=Path, default=None, help="where the fold twins live")
    command.parser.add_argument("--twins", required=True, help="glob for the family's five fold records")
    command.parser.add_argument("--against", default=None, help="glob for a control family's five records")
    arguments = command.parsed()
    root = command.rooted(arguments)
    models = arguments.models or root.models()
    ruler = GoldRuler()

    def family(pattern: str) -> Stitch:
        parts = tuple(sorted(models.glob(pattern)))
        if not parts:
            message = f"no fold records under {models} matching {pattern}"
            raise FileNotFoundError(message)
        return ruler.stitch(pattern, parts, root.raw() / "train.csv")

    control = family(arguments.against) if arguments.against else None
    command.report(ruler.report(family(arguments.twins), control))


if __name__ == "__main__":
    main()
