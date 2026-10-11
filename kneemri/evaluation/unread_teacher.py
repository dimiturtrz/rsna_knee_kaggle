"""Write a score for the cells the report never read, from a pool that never read them either.

27.3% of this project's supervision is a CONSTANT: 14,445 of 52,872 (study, finding) pairs carry the
UNK verdict, which `report_reader` assigns when the report does not name the structure at all —
Synovitis on 84.2% of studies, Fracture 56.4%, Baker's 45.9%. Those cells train toward one number for
every study in the corpus, so they carry no ORDER, and order is the whole of what the metric reads.

`supervision_gap` measured the consequence on the 58 annotated studies: labels - model = -0.0470
[-0.0717, -0.0218], with Synovitis 0.687 against 0.794 and Fracture 0.870 against 0.942. A model
fitted on those constants out-reads them. This module turns that into supervision.

WHERE THE SCORE COMES FROM, and why it is not ours. The published pool's twenty checkpoints partition
the corpus into five folds over four seeds, so every study has a prediction from a model that did not
train on it (`PublishedRuler`). Using OUR member instead would be self-distillation through a shared
trunk on cells our member was itself fitted on — a training-dynamics experiment wearing a label
experiment's clothes. The published arrays are out-of-fold for the whole corpus, cost no card hour,
and are the same arrays `supervision_gap`'s -0.0470 was measured with.

WHAT IT WRITES, and the two things it deliberately does not do.

It replaces the SCORE and never the weight. An UNK cell keeps whatever `--weight-floor` gave it: this
changes what the cell pulls toward, not how hard, so the experiment stays one variable wide.

It is MEAN-PRESERVING within each finding's UNK block. The rank is taken inside that block, so it is
uniform by construction, and the band is centred on the constant those cells already carry. The
marginal a head sees is therefore unchanged and only the ORDER is added — which matters because the
constant is not arbitrary: it is the level the reader emits for silence, and its measured positive
rate is what `--calibrated-targets` fits. Moving the mean would be a second change, and a prevalence
shift is the one thing a twelve-head trunk transmits between heads.

THE BAND IS DERIVED, NOT CHOSEN. Its half-width is the smaller distance from the UNK level to its
neighbouring levels in `MEASURED_AGREEMENT`, so an override never asserts more about a cell than the
reader's own adjacent verdict would have. That table pairs the level the reader emits with the rate
the radiologist confirms, and a label table is written in one space or the other, so the axis is
chosen by which one the centre sits on rather than assumed: uncalibrated, 0.28 against 0.08 and 0.68
gives a half-width of 0.20 and a band of [0.08, 0.48]; under `--calibrated-targets`, 0.136 against
0.041 and 0.378 gives 0.095 and [0.041, 0.231]. The conservative side binds on purpose: the pool's
discriminative power on UNK cells cannot be measured corpus-wide (only 58 studies carry an
annotation), so the size of the claim is capped by something that was measured.

WHAT THIS IS NOT. It is not a reading improvement — `report_reader` already showed the text is not
there to read: on Synovitis UNK cells the report names the structure in 1.8% of 1,424, and on Effusion
0 of 84. It is semi-supervised targeting of cells whose supervision was never a measurement, and it
should be read against the pairs the lexicon left blank. See knee-s6s9r.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from core import cli
from core.data.labels.targets import MEASURED_AGREEMENT, SUSPECT_STUDY, SUSPECT_TARGET, UNREAD_VALUE
from core.findings import STUDY, TARGETS
from kneemri.evaluation.published_ruler import PublishedRuler
from kneemri.inference.vote import RankAverage

logger = logging.getLogger(__name__)

UNREAD_VERDICT = "UNK"
"""The verdict `report_reader` writes when the report does not name the structure at all."""

VERDICT_SUFFIX = "__verdict"

RANKABLE = 2
"""The fewest values that carry an order; below it a rank says nothing and the centre is the honest answer."""


@dataclass(frozen=True)
class UnreadTeacher:
    """Scores for the unread cells, from the published out-of-fold pool, centred where they already sit."""

    labels: Path
    """The report-derived table whose UNK verdicts say which cells are unread."""

    published: Path
    """The directory the twenty published checkpoints were unzipped into."""

    def band(self, centre: float) -> float:
        """Half-width of the override band: the nearer gap from the unread level to a neighbouring one.

        Read off `MEASURED_AGREEMENT` rather than stated, because that table is what a calibrated run
        rewrites the levels to and a hard-coded width would silently describe the uncalibrated scale
        after a switch flip. The NEARER neighbour and not the mean of the two, so the claim is capped
        by the tighter side on both sides of the centre.

        THE AXIS IS CHOSEN, NOT ASSUMED, and that is the whole reason this is a method. Each pair in
        `MEASURED_AGREEMENT` is (level the reader emits, rate the radiologist confirms), and a label
        table is written in ONE of those two spaces: 0.28 uncalibrated, 0.136 under
        `--calibrated-targets`. Reading the rate axis for an uncalibrated centre measures 0.28 against
        neighbours it has no scale in common with — it returns 0.098 where the raw neighbours 0.08 and
        0.68 say 0.20, so the band comes out half-width with a rationale that does not describe it.
        The unread centre IS one of the five levels by construction, so the axis it sits nearest to is
        the space the table is written in, and the gaps are measured from that LEVEL rather than from
        the centre itself. The centre arrives as the mean of thousands of copies of one constant, so it
        is 0.28000000000000003 and not 0.28 — measured from there, the level's own gap is 1e-17 instead
        of zero, it survives any "not the same point" filter, and the band comes back 0.0000 with every
        override collapsing onto the constant it was meant to spread. Snapping to the level first makes
        the arithmetic exact rather than tolerant.
        """
        axes = ([level for level, _ in MEASURED_AGREEMENT], [rate for _, rate in MEASURED_AGREEMENT])
        axis = min(axes, key=lambda values: min(abs(value - centre) for value in values))
        nearest = min(axis, key=lambda value: abs(value - centre))
        gaps = [abs(value - nearest) for value in axis if value != nearest]
        return min(gaps) if gaps else 0.0

    def ordered(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        """The values as ranks in [0, 1], uniform by construction, so the mean of any band is its centre.

        Ties take the average rank, which is what keeps two identical predictions from being separated
        by an accident of sort order — the pool's arrays hold repeated values on the low tail.
        """
        if len(values) < RANKABLE:
            return np.full(len(values), 0.5, dtype=np.float64)
        ranks = pd.Series(values).rank(method="average").to_numpy(dtype=np.float64)
        return ((ranks - 1.0) / (len(values) - 1.0)).astype(np.float64)

    def table(self) -> pd.DataFrame:
        """One row per unread cell: the study, the finding, and the score to train it toward."""
        derived = pd.read_csv(self.labels).set_index(STUDY)
        names, per_seed = PublishedRuler(self.published).per_seed()
        pooled = RankAverage().combine(per_seed)
        logger.info("%d studies in the published pool, %d in the label table", len(names), len(derived))

        rows: list[pd.DataFrame] = []
        for column, target in enumerate(TARGETS):
            verdict = f"{target}{VERDICT_SUFFIX}"
            if verdict not in derived.columns:
                continue
            unread = derived.index[derived[verdict] == UNREAD_VERDICT]
            carried = [name for name in names if name in set(unread)]
            if not carried:
                continue
            position = {name: index for index, name in enumerate(names)}
            read = pooled[[position[name] for name in carried], column]
            centre = float(derived.loc[carried, target].mean())
            half = self.band(centre)
            rows.append(
                pd.DataFrame(
                    {
                        SUSPECT_STUDY: carried,
                        SUSPECT_TARGET: target,
                        UNREAD_VALUE: centre + (2.0 * self.ordered(read) - 1.0) * half,
                    }
                )
            )
            logger.info("%-18s %6d unread cells, centre %.4f, band +/-%.4f", target, len(carried), centre, half)
        return (
            pd.concat(rows, ignore_index=True)
            if rows
            else pd.DataFrame({SUSPECT_STUDY: [], SUSPECT_TARGET: [], UNREAD_VALUE: []})
        )


def main() -> None:
    """Write the unread-cell scores `--unread-targets` trains toward."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, required=True, help="the report-derived table to read UNK from")
    command.parser.add_argument("--out", type=Path, required=True, help="where the (study, target, value) table goes")
    arguments = command.parsed()

    root = command.rooted(arguments)
    table = UnreadTeacher(arguments.labels, root.processed() / "published").table()
    table.to_csv(arguments.out, index=False)
    logger.info("%d unread cells written to %s", len(table), arguments.out)


if __name__ == "__main__":
    main()
