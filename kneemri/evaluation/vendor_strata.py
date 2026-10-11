"""The acquisition columns of `canonical_metadata.csv`, cut into named subsets of a study list.

Separate from `vendor_ruler` because the two answer different questions and only one of them is about
members: this file turns a metadata table into strata, and knows nothing about scores, orderings or
null bands. A second reader of the same columns — a per-vendor residual cut, a vendor covariate at
training time — takes this and not the ruler.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Int
from numpy.typing import NDArray

from core.findings import STUDY

IDENTIFIER = STUDY
SCANNER = "scanner"
FIELD_STRENGTH = "field_strength_t"

HIGH_FIELD_T = 2.0
"""Where the field-strength axis is cut, between the corpus' two occupied values (1.5 T and 3.0 T).

A cut rather than an equality because the census also holds 24 studies at 1.16 T and one at 1.0 T, and
an equality test would drop them into an absent bucket that means something else.
"""


@dataclass(frozen=True)
class Stratum:
    """A named subset of the shared holdout, as positions into the pool's arrays."""

    name: str
    studies: Int[NDArray[np.int64], " studies"]

    @property
    def size(self) -> int:
        """How many studies fell in it — the denominator every number read over it uses."""
        return int(self.studies.size)


class VendorStrata:
    """The acquisition columns of `canonical_metadata.csv`, cut into strata over a list of studies."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table

    @classmethod
    def read(cls, metadata: Path) -> "VendorStrata":
        """The canonical metadata table, indexed by study identifier.

        Read from the table rather than from the DICOMs because the reader already wrote it there and a
        second header pass over 4,407 studies would be minutes of IO for the same two columns. Both
        columns were empty in every build before 2026-09-05 (knee-m4jkz); a table from before that fix
        yields no stratum at all, which `strata` reports as a size rather than as an absence of shift.
        """
        return cls(pd.read_csv(metadata).set_index(IDENTIFIER))

    def scanners(self, studies: list[str]) -> "pd.Series[str]":
        """The manufacturer column, in the order the pool holds its studies, absent where unjoined."""
        return self.table[SCANNER].reindex(studies)

    def fields(self, studies: list[str]) -> "pd.Series[float]":
        """The field-strength column, in the same order, as floats so the threshold can be applied."""
        return self.table[FIELD_STRENGTH].reindex(studies).astype(float)

    def strata(self, studies: list[str]) -> list[Stratum]:
        """Every vendor and both field-strength bands, as positions into `studies`.

        Vendor and field strength are returned as ONE list rather than two reports because they are not
        independent cuts of the corpus — 3 T is concentrated in the newer scanners — and a reader
        comparing them side by side is reading the same shift twice. Whichever moves is the axis to take
        to a card; the report does not pick between them.
        """
        scanners = self.scanners(studies)
        fields = self.fields(studies)
        found = [
            Stratum(str(vendor), np.flatnonzero((scanners == vendor).to_numpy()))
            for vendor in sorted({str(value) for value in scanners.dropna().tolist()})
        ]
        low = np.flatnonzero((fields < HIGH_FIELD_T).to_numpy())
        high = np.flatnonzero((fields >= HIGH_FIELD_T).to_numpy())
        return [*found, Stratum("under 2 T", low), Stratum("2 T and above", high)]
