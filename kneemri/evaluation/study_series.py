"""The competition's per-series acquisition table, reduced to one number per study.

`train_series.csv` is the whole of the acquisition metadata the competition SHIPS: one row per series,
carrying `Fluid_Sensitive`, `Fat_Suppression` and `Anatomical_Plane`. It has no scanner field, no
laterality field and no per-study demographic.

THAT IS A FACT ABOUT THE SHIPPED TABLE AND NOT ABOUT THE DATA, and this docstring used to state it as
the latter — "so this is the entire vocabulary any population cut of our results can be phrased in".
The DICOM headers underneath carry `Manufacturer`, `ManufacturerModelName` and `MagneticFieldStrength`,
and the corpus is four vendors deep (Siemens 1,949, Philips 1,301, GE 905, Canon 226) across two field
strengths. Our reader dropped both tags until knee-m4jkz; `canonical_metadata.csv` now carries them and
`kneemri.evaluation.vendor_ruler` cuts on them. `slots.py` had already established the general rule for
fat suppression: where the headers and the shipped columns both answer, the headers are the better
witness.

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

from core.findings import STUDY


class StudySeries:
    """The competition's per-series acquisition table, reduced to one number per study."""

    def __init__(self, series: Path) -> None:
        self.table = pd.read_csv(series)

    def counts(self, studies: list[str]) -> Int[NDArray[np.int64], " studies"]:
        """How many series each study has, in the given order."""
        sizes = self.table.groupby(STUDY).size()
        return np.asarray(sizes.reindex(studies).to_numpy(), dtype=np.int64)

    def fluid(self, studies: list[str]) -> Int[NDArray[np.int64], " studies"]:
        """How many of each study's series are fluid-sensitive, in the given order."""
        sums = self.table.groupby(STUDY)["Fluid_Sensitive"].sum()
        return np.asarray(sums.reindex(studies).to_numpy(), dtype=np.int64)

    def redundant(self) -> bool:
        """Whether `Fat_Suppression` says anything `Fluid_Sensitive` does not, over every row."""
        return bool((self.table["Fluid_Sensitive"] == self.table["Fat_Suppression"]).all())
