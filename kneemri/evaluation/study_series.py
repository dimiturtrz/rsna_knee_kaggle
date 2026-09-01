"""The competition's per-series acquisition table, reduced to one number per study.

`train_series.csv` is the whole of the acquisition metadata the competition ships: one row per series,
carrying `Fluid_Sensitive`, `Fat_Suppression` and `Anatomical_Plane`. There is no scanner field, no
laterality field and no per-study demographic, so this is the entire vocabulary any population cut of
our results can be phrased in.

Two of its three columns turn out to say nothing. `Anatomical_Plane` is constant — all 4,407 training
studies carry Axial, Coronal and Sagittal alike — and `Fat_Suppression` equals `Fluid_Sensitive` on
every one of the 24,371 rows, which `redundant` exists to keep asserting rather than remembering. What
is left is how many series a study has, and how many of them are fluid-sensitive.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Int
from numpy.typing import NDArray


class StudySeries:
    """The competition's per-series acquisition table, reduced to one number per study."""

    def __init__(self, series: Path) -> None:
        self.table = pd.read_csv(series)

    def counts(self, studies: list[str]) -> Int[NDArray[np.int64], " studies"]:
        """How many series each study has, in the given order."""
        sizes = self.table.groupby("StudyInstanceUID").size()
        return np.asarray(sizes.reindex(studies).to_numpy(), dtype=np.int64)

    def fluid(self, studies: list[str]) -> Int[NDArray[np.int64], " studies"]:
        """How many of each study's series are fluid-sensitive, in the given order."""
        sums = self.table.groupby("StudyInstanceUID")["Fluid_Sensitive"].sum()
        return np.asarray(sums.reindex(studies).to_numpy(), dtype=np.int64)

    def redundant(self) -> bool:
        """Whether `Fat_Suppression` says anything `Fluid_Sensitive` does not, over every row."""
        return bool((self.table["Fluid_Sensitive"] == self.table["Fat_Suppression"]).all())
