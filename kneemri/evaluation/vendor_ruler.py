"""Whether member RANK survives a change of scanner: `python -m kneemri.evaluation.vendor_ruler`.

The 909-study holdout is a RANDOM split over 4,407 studies drawn from four vendors, so train and
holdout share the scanner mix by construction and no number this project has quoted was measured
against acquisition shift. The Kaggle test set need not share that mix. This module asks the one
question the records already on disk can answer: if the mix moved, would the seat order move with it?

IT READS RANK AND NOT LEVEL, deliberately. A vendor stratum is a few hundred studies where the pooled
floor of 0.0062 was measured on 909, so its absolute AUC carries an interval too wide to state a member's
quality with. The ORDERING is a paired comparison over the same members on the same labels, and it is
the quantity a seat search actually consumes: greedy selection spent one ordering, and if that ordering
is a property of our scanner mix then every seated ensemble was fitted to it.

THE CONTROL IS THE WHOLE READING. Any subset of a few hundred studies reorders members somewhat, purely
by sampling — so a vendor rho below 1.0 is not evidence of anything on its own. Each vendor is therefore
scored against the SAME statistic computed on random subsets of exactly its own size, drawn from the
same shared studies. The finding is the GAP between the two, and a vendor whose rho sits inside its own
random band has been shown to carry no ordering signal rather than left unmeasured. Reporting the vendor
number alone is how this axis would produce a confident result about sampling noise.

WHAT IT CANNOT SEE. Every member in the pool was TRAINED on all four vendors, so this measures whether
a vendor prices members differently, not what a member trained on three vendors does on the fourth. That
is a card question and this is not it. `InstitutionName`, `StationName` and `DeviceSerialNumber` are
absent from the corpus, so there is no site axis below the vendor one — a vendor stratum pools every
site that bought that manufacturer.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Int
from numpy.typing import NDArray

from core import cli
from core.data.corpus.build_cache import METADATA
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.vendor_strata import Stratum, VendorStrata

logger = logging.getLogger(__name__)

MIN_STRATUM = 120
"""The smallest stratum whose member ordering is reported at all.

A rank correlation over a handful of studies is dominated by which studies were drawn, and the control
below would widen to cover any value — so the stratum would read as null whatever it holds, which is a
refusal wearing a measurement's clothes. Stating the size and declining is the honest form.
"""

DRAWS = 200
"""Random subsets per stratum, for the null band. Two hundred puts the 5th and 95th percentiles on
about ten draws each, which is enough to place a single observed rho against them and cheap enough that
the whole report stays a few seconds of arithmetic."""

BAND = 5.0
"""The percentile the null band is drawn at, each side. A vendor rho below the 5th percentile of its own
random band is the finding; anything inside the band is this axis reading as silent."""

SEATS = 8
"""How many members the seat-overlap line compares. The pool's own count curve is flat from six to
twenty-four, and eight is where `record_mix` has historically read its peak — so it is the number a
seat search would have spent, which is what makes an overlap under it worth printing.

THE COLUMN IT FEEDS IS NOT READABLE YET (knee filed): `line` prints the overlap with no null band
beside it, and subset membership reshuffles by sampling alone at these sizes — Philips keeps 2 of 8
while its rho sits comfortably inside its own band."""


class VendorRuler:
    """Every member scored inside each stratum, and whether the ordering is the pool's own."""

    def __init__(self, pool: Pool, strata: list[Stratum], scores: LabelScores, generator: np.random.Generator) -> None:
        self.pool = pool
        self.strata = strata
        self.scores = scores
        self.generator = generator

    def scored(self, studies: Int[NDArray[np.int64], " studies"]) -> list[float]:
        """Each member's mean twelve-column score over exactly these studies.

        `Pool.rows` restricts the members and the truth together, which is the whole reason it is used
        here: slicing predictions with one index array and truth with another scores every member
        against another stratum's labels and nothing raises.
        """
        rows = self.pool.rows(studies)
        return [self.scores.mean(member, rows.truth) for member in rows.predictions]

    @staticmethod
    def agreement(here: list[float], there: list[float]) -> float:
        """Spearman rank correlation between two orderings of the same members.

        Pearson over ranks, with ties averaged: two members scoring identically inside a stratum must
        not have their order decided by their position in the pool, which a plain argsort would do
        silently and which is exactly the disagreement this module claims to be measuring.
        """
        ranked = [pd.Series(values).rank().to_numpy() for values in (here, there)]
        if any(float(np.std(values)) == 0.0 for values in ranked):
            return float("nan")
        return float(np.corrcoef(ranked[0], ranked[1])[0, 1])

    def null_band(self, size: int, whole: list[float]) -> tuple[float, float]:
        """What `agreement` reads on random subsets of this size — the band a vendor must fall below.

        The subset is drawn from the shared studies without replacement and compared to the POOLED
        ordering, which is the same comparison the vendor gets. So the two numbers differ in one thing
        only: whether the subset was chosen by a manufacturer or by a coin.
        """
        population = len(self.pool.truth)
        drawn = [
            self.agreement(self.scored(self.generator.choice(population, size=size, replace=False)), whole)
            for _ in range(DRAWS)
        ]
        kept = [value for value in drawn if not np.isnan(value)]
        return (
            (float(np.percentile(kept, BAND)), float(np.percentile(kept, 100 - BAND)))
            if kept
            else (
                float("nan"),
                float("nan"),
            )
        )

    def seats(self, scored: list[float]) -> set[str]:
        """The names of the `SEATS` best members under one ordering — what a seat search would take."""
        order = sorted(range(len(scored)), key=lambda index: scored[index], reverse=True)
        return {self.pool.names[index] for index in order[:SEATS]}

    def line(self, stratum: Stratum, whole: list[float], pooled_seats: set[str]) -> str:
        """One stratum's row: its size, its agreement with the pooled ordering, and its null band."""
        if stratum.size < MIN_STRATUM:
            return f"  {stratum.name:<16} {stratum.size:>4} studies — under the {MIN_STRATUM} floor, not read"
        scored = self.scored(stratum.studies)
        rho = self.agreement(scored, whole)
        low, high = self.null_band(stratum.size, whole)
        verdict = "BELOW ITS NULL BAND" if rho < low else "inside its null band"
        kept = len(pooled_seats & self.seats(scored))
        return (
            f"  {stratum.name:<16} {stratum.size:>4} studies ({stratum.size / len(self.pool.truth):.1%})  "
            f"rho {rho:.3f}  null {low:.3f}-{high:.3f}  {verdict}  "
            f"top-{SEATS} kept {kept}/{SEATS}"
        )

    def report(self) -> list[str]:
        """The strata against the pooled ordering, each beside the band its own size earns."""
        whole = self.scored(np.arange(len(self.pool.truth), dtype=np.int64))
        pooled_seats = self.seats(whole)
        return [
            f"{len(self.pool.names)} members over {len(self.pool.truth)} shared studies, "
            f"vote {self.pool.rule()}; pooled best {max(whole):.4f}",
            "rho is Spearman against the POOLED member ordering; the null band is the same statistic on "
            f"{DRAWS} random subsets of the same size",
            *(self.line(stratum, whole, pooled_seats) for stratum in self.strata),
        ]


def main() -> None:
    """Score every member with a holdout record inside each vendor and field-strength stratum."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--records", type=Path, nargs="*", default=None, help="checkpoints to read; every record by default"
    )
    command.parser.add_argument("--seed", type=int, default=2026, help="the draw the null band is taken under")
    arguments = command.parsed()

    root = command.rooted(arguments)
    records = arguments.records if arguments.records else sorted(root.models().glob("*.pt"))
    mixed = RecordMix(records)
    for dropped in mixed.narrow():
        logger.info("%s", dropped)
    pool = mixed.per_member()
    strata = VendorStrata.read(root.processed() / "metadata" / METADATA).strata(mixed.shared())
    ruler = VendorRuler(pool, strata, LabelScores(), np.random.default_rng(arguments.seed))
    command.report(ruler.report())


if __name__ == "__main__":
    main()
