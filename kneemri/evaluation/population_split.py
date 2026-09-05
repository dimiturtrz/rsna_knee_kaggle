"""Whether the pool's residual concentrates in a KIND of study: `python -m kneemri.evaluation.population_split`.

`residual_split` cuts the same errors by finding and by how many members agreed. It cannot say whether
the studies we get wrong are a recognisable population — and if they are, that is a DATA lane (sample
more of them, weight them, render them differently) rather than the model lane every other instrument
here reports on.

WHAT THE COMPETITION SHIPS to cut on. `train_series.csv` carries one row per series with
`Fluid_Sensitive`, `Fat_Suppression` and `Anatomical_Plane`, and that is the whole of the SHIPPED
acquisition metadata — no scanner column, no laterality column, no per-study demographic. Of what is there:

 * ANATOMICAL PLANE is not a population axis at all. All 4,407 training studies carry all three of
   Axial, Coronal and Sagittal, so a cut on plane coverage has one stratum.

 * FLUID_SENSITIVE AND FAT_SUPPRESSION ARE THE SAME COLUMN. They agree on 24,371 of 24,371 rows. Two
   of the three fields are one field, which is why this module cuts on the count of fluid-sensitive
   series and never on fat suppression separately.

 * SERIES COUNT is the only real axis IN THE SHIPPED TABLE, 3 to 14 with a median of 5. It is how
   much of the knee a member was shown, so if the residual is thin-acquisition-heavy the fix is an
   input fix.

AND IT IS NOT AN INPUT AXIS ONLY. A longer protocol is also a sicker knee — 2.8 positive findings per
study at four series against 4.0 at six — so raw errors per study rise with series count without any
member having been helped or hurt by the extra sequences. That is why every band reports errors per
POSITIVE beside errors per study, and why only the former may be read.

AND THE SHIPPED TABLE IS NOT THE WHOLE VOCABULARY. The DICOM headers carry the manufacturer, the model
and the field strength, which our reader dropped until knee-m4jkz — four vendors and two field strengths
over 4,407 studies. Those cuts are a DIFFERENT question and live in `kneemri.evaluation.vendor_ruler`:
this module asks where the pool's residual concentrates, that one asks whether the member ORDERING is a
property of our scanner mix. A stratum that reorders the seats is not the same finding as a stratum that
carries more errors, and reading one as the other is how a seat search gets fitted to a nuisance axis.

WHAT IT CANNOT SEE. The same two limits `residual_split` carries, since it borrows that decomposition
whole: the operating point is prevalence-matched so false negatives and false positives are equal by
construction, and members of one family are not an honest sample of all models. Added to those: the
public test metadata is a three-study stub, so nothing measured here can be checked against the studies
we are actually scored on.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Int
from numpy.typing import NDArray

from core import cli
from core.denominator import Denominator
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.residual_split import ResidualSplit
from kneemri.evaluation.study_series import StudySeries
from kneemri.evaluation.uncertainty_split import UncertaintySplit

logger = logging.getLogger(__name__)

SERIES_BANDS: tuple[tuple[int, int, str], ...] = (
    (0, 4, "4 or fewer series"),
    (5, 5, "5 series"),
    (6, 6, "6 series"),
    (7, 99, "7 or more series"),
)
"""Series-count strata, cut at the median rather than at equal widths: 5 is 2,299 of 4,407 studies, so
a band per count would report ten strata of which two carry the corpus."""

FLUID_BANDS: tuple[tuple[int, int, str], ...] = (
    (0, 2, "2 fluid-sensitive"),
    (3, 3, "3 fluid-sensitive"),
    (4, 99, "4 or more fluid-sensitive"),
)
"""Fluid-sensitive series strata. Same shape of choice, over a narrower range (2 to 7)."""


@dataclass(frozen=True)
class PopulationRow:
    """One stratum's residual: how many studies fell in it and how badly they went."""

    stratum: str
    studies: int
    positives: int
    errors: int
    unanimous: int

    @property
    def per_study(self) -> float:
        """Wrong cells per study. CONFOUNDED across strata, and reported only beside `per_positive`."""
        return self.errors / self.studies if self.studies else 0.0

    @property
    def per_positive(self) -> float:
        """Wrong cells per POSITIVE finding — the number that is actually comparable across strata.

        A study with a longer protocol also has more findings (2.8 positives at four series against 4.0
        at six), so the per-study column rises with series count for a reason that is about how sick the
        knee is rather than about how much of it was imaged. Dividing by the positives it had to get
        right is what separates those two, and it is the column a reading may be taken from.
        """
        return self.errors / self.positives if self.positives else 0.0

    @property
    def unanimous_share(self) -> float:
        """The share of this stratum's errors that every member also makes."""
        return self.unanimous / self.errors if self.errors else 0.0

    def line(self) -> str:
        """This stratum's row."""
        return (
            f"  {self.stratum:<26} studies {self.studies:<5} positives/study "
            f"{self.positives / self.studies if self.studies else 0.0:.2f}  errors {self.errors:<5} "
            f"per study {self.per_study:.3f}  per positive {self.per_positive:.3f}  "
            f"unanimous {self.unanimous_share:.1%}"
        )


class PopulationSplit:
    """The pool's per-study error load, banded by an acquisition property of the study."""

    def __init__(self, split: ResidualSplit, studies: list[str]) -> None:
        self.split = split
        self.studies = studies

    def per_study_errors(self) -> Int[NDArray[np.int64], " studies"]:
        """How many findings the ensemble gets wrong on each study."""
        return self.split.ensemble().sum(axis=1).astype(np.int64)

    def per_study_unanimous(self) -> Int[NDArray[np.int64], " studies"]:
        """How many of each study's wrong findings EVERY member also gets wrong."""
        return (self.split.agreement() == self.split.members()).sum(axis=1).astype(np.int64)

    def banded(
        self,
        values: Int[NDArray[np.int64], " studies"],
        bands: tuple[tuple[int, int, str], ...],
    ) -> list[PopulationRow]:
        """One row per band, over the studies whose value falls inside it."""
        errors = self.per_study_errors()
        unanimous = self.per_study_unanimous()
        positives = self.split.truth().sum(axis=1)
        rows: list[PopulationRow] = []
        for low, high, name in bands:
            inside = (values >= low) & (values <= high)
            rows.append(
                PopulationRow(
                    stratum=name,
                    studies=int(inside.sum()),
                    positives=int(positives[inside].sum()),
                    errors=int(errors[inside].sum()),
                    unanimous=int(unanimous[inside].sum()),
                )
            )
        return rows

    def report(self, series: StudySeries) -> list[str]:
        """Both acquisition cuts, each under its own heading."""
        lines = [
            f"{self.split.members()} members over {len(self.studies)} held-out studies; "
            f"Fat_Suppression duplicates Fluid_Sensitive: {series.redundant()}",
            "by series count",
            *[row.line() for row in self.banded(series.counts(self.studies), SERIES_BANDS)],
            "by fluid-sensitive series count",
            *[row.line() for row in self.banded(series.fluid(self.studies), FLUID_BANDS)],
        ]
        return lines


def main() -> None:
    """Cut these members' residual by the acquisition properties of the studies it falls on."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--records",
        type=Path,
        nargs="+",
        required=True,
        help="the checkpoints to decompose, sharing ONE split; every one must carry a holdout record",
    )
    arguments = command.parsed()
    root = command.rooted(arguments)
    records = list(arguments.records)
    split = ResidualSplit(UncertaintySplit.over(records).pool)
    studies = RecordMix(records).shared()
    Denominator(len(studies), "the members share no held-out study to cut").refuse_empty()
    series = StudySeries(root.raw() / "train_series.csv")
    command.report(PopulationSplit(split, studies).report(series))


if __name__ == "__main__":
    main()
