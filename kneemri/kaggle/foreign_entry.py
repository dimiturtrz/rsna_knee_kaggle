"""Serve mattiaangeli's five released folds on a Kaggle split, as one more foreign table.

`RaptorEntry` is the same shape for a different foreign system, and the two are deliberately not
merged: that member carries its own five-slot volume and its own 42-window evaluation, this one
carries a whole lifted PROGRAM of someone else's — their decode, their model classes, their serving
loop, cut out of their kernel by `foreign_folds`. What the two share is `RaptorSeat.combined`, the
rank-space join every foreign seat this project has shipped goes through, and `Submission.check`, the
refusal a second producer of submissions is held to without a second copy of it.

THE FIVE FOLDS ARE RANK-MEANED, WHICH IS THEIR OWN SERVING FORM AND NOT OURS TO CHANGE. Offline we
price the family fold-matched — every study scored by the ONE fold that never saw it — because on our
909 studies four of five folds trained on each one, and a rank-mean there reads four in-sample
opinions per study. On the hidden test split none of the five trained on anything, so the contamination
that forces the fold-matched reading offline does not exist here and their rank-mean is simply the
better estimator. The consequence is that the offline number is a FLOOR on this: 0.8812 fold-matched
is one fold's opinion, and what ships is five.

WHY THEIR SOURCE IS MOUNTED RATHER THAN RETYPED. Their file holds three programs sharing a namespace,
and two of the three `SLICE_BAND`/`CROP_MM` values in it are wrong for these weights. A port that
resolved names by exec-then-look-up would feed their model an input it has never seen and the low
score would read as "their model is not better than ours". `foreign_folds.SERVING` names the line
range of the one program that serves the released checkpoints.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float
from numpy.typing import NDArray

from core.findings import STUDY, TARGETS
from kneemri.inference.foreign_folds import ForeignFolds
from kneemri.inference.vote import Vote
from kneemri.kaggle.submission import Split, Submission

logger = logging.getLogger(__name__)

WORKING = Path("/kaggle/working")

CHECKPOINTS = "knee-mri-fold-weights"
"""The public CC0-1.0 dataset their released folds already live in — mounted, never redistributed."""

SOURCE = "bend_the_knee_to_dinov3_ensembled.py"
"""Their ENSEMBLED kernel, carried in our kit so the lift has a file to cut from with no internet.

Their other released notebook shares most of these symbol names and defines none of the four the lift
needs, so naming it costs a run that dies on `defines none of ['SIZE', 'N_SLICE', 'INTENSITY', 'LABELS']`
— which is the good failure. `foreign_folds.SERVING` is a line range INTO this file, and a line range
into the wrong file of a pair that reads alike is the failure that would not have announced itself.
"""


@dataclass(frozen=True)
class ForeignEntry:
    """One submission table from their five folds: where their program is, and where their weights are."""

    competition: Path
    kernel: Path
    checkpoints: Path
    series_root: Path
    working: Path = WORKING
    split: Split = field(default_factory=Split)

    @staticmethod
    def ranked(
        predictions: Float[NDArray[np.float32], "folds studies labels"],
    ) -> Float[NDArray[np.float64], "studies labels"]:
        """Their five folds as one opinion, averaged in RANK space rather than probability space.

        Rank rather than probability for the reason `RaptorSeat` gives about two systems: five heads
        trained on five different subsets are five different calibrations, and the metric reads order.
        Averaging their probabilities would let whichever fold happens to be most confident decide.
        """
        return np.mean([Vote.ranks(fold.astype(np.float64)) for fold in predictions], axis=0)

    def table(
        self, identifiers: list[str], probabilities: Float[NDArray[np.float64], "studies labels"]
    ) -> pd.DataFrame:
        """The wide submission frame, held to the same refusal every other producer is held to."""
        frame = pd.DataFrame(probabilities, columns=list(TARGETS))
        frame.insert(0, STUDY, identifiers)
        Submission.check(frame, identifiers)
        return frame

    def run(self) -> pd.DataFrame:  # devtools-ignore: test-mirror
        """Lift their program, load their five folds, score the split, write the CSV."""
        test = pd.read_csv(self.competition / f"{self.split.name}.csv")
        identifiers = [str(study) for study in test[STUDY].tolist()]
        series = pd.read_csv(self.competition / f"{self.split.name}_series.csv")
        folds = ForeignFolds(self.kernel, self.checkpoints, self.series_root)
        module = folds.module()
        if list(module.LABELS) != list(TARGETS):
            message = f"their label order is not ours: {list(module.LABELS)}"
            raise ValueError(message)
        folds.nets(module)
        predictions = folds.serve(module, identifiers, folds.records(series, identifiers))
        frame = self.table(identifiers, self.ranked(predictions))
        frame.to_csv(self.working / "submission.csv", index=False)
        logger.info("wrote %d rows from %d folds", len(frame), len(predictions))
        return frame
