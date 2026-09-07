"""The scored epoch a run's KEPT PREDICTIONS reconstruct, for the runs that predate the recorder.

Seventeen runs on disk kept their per-study predictions and twenty-four wrote a metrics file, and the
two sets are not the same set. `wsfold0_e6` is in the gap — and it is the control knee-bot's whole
argument rests on, so `run_comparison` refused the one comparison the ledger most needs with
"control does not exist" while the arrays that answer it sat beside the checkpoint.

RECONSTRUCTED, NOT ESTIMATED. The holdout record holds the predictions, the truth they were scored
against and the epoch they came from, so the per-label AUCs come out of the same arrays the ledger's
numbers were read off. What it cannot recover is the loss and the wall-clock, which were never in the
array; those come back as NaN and 0 rather than as plausible values.

NOTHING IS WRITTEN. A reconstructed row is returned in memory and never appended to a metrics file:
its 0-second epoch would then be a real row on disk, and `RunMetrics.epoch_seconds` would read a run's
cost as zero — a fabricated number outliving the comparison that needed it.
"""

from dataclasses import dataclass

import numpy as np

from kneemri.evaluation.scores import LabelScores
from kneemri.runs.holdout_record import HoldoutReading, HoldoutRecord
from kneemri.runs.run_metrics import EpochRow, RunMetrics

UNRECOVERABLE_SECONDS = 0
"""A reconstructed epoch has no wall-clock: the array never held one. Zero rather than a guess, and it
never reaches disk."""


@dataclass(frozen=True)
class ScoredEpoch:
    """A run's best epoch and where the numbers came from, because the two are not equally complete."""

    row: EpochRow
    reconstructed: bool

    def named(self) -> str:
        """The sentence a report prints so a reconstructed control is never read as a recorded one."""
        if not self.reconstructed:
            return "scores as recorded by the run"
        return "scores RECONSTRUCTED from the run's kept predictions (it predates the metrics recorder)"


class HoldoutMetrics:
    """A run's scored epoch, from its metrics file when it wrote one and from its predictions when not."""

    def __init__(self, scores: LabelScores | None = None, excluded: frozenset[str] | None = None) -> None:
        self.scores = scores or LabelScores()
        self.excluded = excluded or frozenset()

    def kept(self, reading: HoldoutReading) -> HoldoutReading:
        """The same reading without the studies this instrument was told to drop.

        WHY ANY STUDY IS EVER DROPPED. A foreign arm is not always scored on the set it was selected
        on. The dread trainer validates on the gold-58 and keeps its best epoch by that number, so a
        gold study inside OUR held-out fold is in-sample for it in the one way that matters — it chose
        the weights. Comparing on those rows credits the arm with a selection it made, and the effect
        is directional, never noise. Excluding them by NAME rather than by count, because which
        studies overlap is a property of the fold file and moves when the fold does.
        """
        if not self.excluded:
            return reading
        kept = [position for position, study in enumerate(reading.studies) if study not in self.excluded]
        rows = np.array(kept, dtype=np.int64)
        return HoldoutReading(
            studies=[reading.studies[position] for position in kept],
            predictions=reading.predictions[rows],
            truth=reading.truth[rows],
            epoch=reading.epoch,
            source=reading.source,
        )

    def epoch(self, reading: HoldoutReading) -> EpochRow:
        """The per-label AUCs a kept holdout implies, scored exactly as the training loop scored them.

        Through `LabelScores.per_label` rather than a second implementation of the same loop: the
        binarisation at the midpoint is the ruler's own decision, and a reconstruction that spelled it
        out again would be free to drift from the runs it is compared against.
        """
        reading = self.kept(reading)
        per_label = self.scores.per_label(reading.predictions, reading.truth)
        return EpochRow(
            epoch=reading.epoch,
            loss=float("nan"),
            auc=float(np.nanmean(list(per_label.values()))),
            seconds=UNRECOVERABLE_SECONDS,
            per_label=per_label,
            # NaN rather than a guess, under the same discipline as `loss`: the kept arrays hold the
            # predictions and the epoch, and none of these was ever in them. A reconstructed ECE could
            # in principle be computed from the predictions — but the recorder reads it on the SAME
            # binarised truth, and computing it here would be free to drift from that; absent is honest.
            ece=float("nan"),
            loss_q25=float("nan"),
            loss_q50=float("nan"),
            loss_q75=float("nan"),
            loss_q90=float("nan"),
            loss_q95=float("nan"),
        )

    def of(self, metrics: RunMetrics) -> ScoredEpoch | None:
        """This run's best epoch, preferring what it recorded, or None when neither source can say.

        THE RECORDED ROW WINS when there is one. It carries the loss and the wall-clock, and a run that
        wrote its own scores has already answered the question — reconstructing over the top would
        replace a measurement with a recomputation of the same measurement, and any disagreement
        between them would then be invisible rather than a bug worth seeing.

        UNLESS STUDIES ARE EXCLUDED, and then it cannot win: the recorded row is an AGGREGATE over the
        whole holdout, so there is no arithmetic that removes a study from it. Reconstruction is the
        only way to answer the question that was asked, and the answer says so — `reconstructed` is
        True and `named()` prints it — because a filtered number that claimed to be the recorded one
        would be the more dangerous of the two mistakes.

        A run that held NOTHING back is refused outright, whatever files sit beside it. `RunMetrics.best`
        already refuses one on its recipe, and reconstruction must obey the same rule or it becomes the
        way around it: a final fit's `--fold -1` means there is no held-out study to score, so any
        number computed here would be a score on training data wearing a holdout's name.
        """
        recorded = metrics.best()
        if recorded is not None and not self.excluded:
            return ScoredEpoch(row=recorded, reconstructed=False)
        recipe = metrics.recipe()
        if recipe is not None and int(recipe.get("fold", 0)) < 0:
            return None
        reading = HoldoutRecord(metrics.path).read()
        return ScoredEpoch(row=self.epoch(reading), reconstructed=True) if reading is not None else None
