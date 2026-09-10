"""Re-spread the reader's middle rung from the model's own out-of-fold order.

    python -m core.data.labels.band_resolution --table report_labels_v2.csv --scale raw \
        --stitch 'oof_fold[0-4]_s2026.holdout.npz' --share 0.5 --out report_labels_v6band50.csv

`--scale` MUST match the table, and nothing downstream will tell you if it does not — a mismatch selects
zero cells and reports a clean no-op. `report_labels_v2.csv` is `raw` and `report_labels_v3.csv` is
`calibrated`; the arms train on v2 (`experiments/lib.sh:72`), so `raw` is the one that reaches anything.

WHAT THE MIDDLE RUNG IS AND WHY IT IS THE LARGEST LABEL-SIDE LEVER LEFT. `Supervision.MEASURED_AGREEMENT`
emits five levels; the third, calibrated to 0.3776, is the one that asserts almost nothing — the
radiologist agrees with 37.8% of those calls, so the cell trains at a value a coin would beat by a
whisker. It is not a corner case: D301's census over all 4406 rows of `report_labels_v3.csv` puts 6461
cells on that rung, **12.22% of the table**, and puts them there almost evenly — nine of the twelve
columns hold between 415 and 568 band cells against a uniform 538, seven of them within 6% of uniform.
Effusion is 3x uniform and Synovitis and Fracture a quarter of it, and that is the whole of the
non-uniformity.

WHY THAT SHAPE IS THE ARGUMENT. D298 read the competition's own evaluation page: the board scores
`(1/12) Σ AUC_i`, twelve columns at equal weight. A label-side treatment whose mass concentrates in one
or two columns ceilings at `2/12 × gain` however well it works. This one does not concentrate: its
per-column weight is nearly flat under a scorer whose weights are exactly flat, which is the best shape
a label lever can have here. D300's earlier reading that 67% of the evidence was Effusion was a PAIR
count, not a cell count — a column near 50/50 dominates `n+ × n-` no matter how few cells it holds.

WHY THE MODEL MAY ORDER THESE CELLS AT ALL. D300 measured it on the 98 gold cells that sit on the rung:
within-band AUC 0.7490, bootstrap sd 0.0500 over its own cells, 95% [0.6467, 0.8407], and — the number
that matters — **z +4.14 against the null that belongs to the statistic**, which is 0.5409 rather than
0.5 because pooling twelve strata at different positive rates inflates it by four points on its own.
Taking the model's top 37 of those 98 returns 21 true positives where chance returns 14: **0.568
precision against a 0.378 base rate, a 1.50x lift.** So the pixels carry information the rung threw
away, and this module is what puts it back.

WHAT IT DOES. Within each finding, the cells on the rung are ranked by that finding's out-of-fold score
— a prediction from a fold that never saw the study — and their targets are re-spread SYMMETRICALLY about
the rung, by up to the NEARER neighbour's distance either way (see `reach`, which is where the price of
that symmetry is argued). `--share` is how far: at 0.0 the table is unchanged, at 1.0 a band cell reaches
the nearer neighbour exactly. It can never pass one, and the band's MEAN does not move. The ordering is
per column, so no cell is ever compared to a cell of another finding, which is the same discipline
`gold_estimators` enforces on the reading side and for the same reason: between-finding calibration is
worth exactly zero to a scorer that reads rank within a column.

WHAT IT IS, SAID PLAINLY. This is self-training. The order comes from a model fitted on this corpus's
own report labels, so a systematic reader error the model learned is an error this module will write
back into the table with more confidence than it had. Three things bound that and none of them removes
it: the model never saw the study whose cell it orders, the re-spread cannot cross a rung the reader did
not already bracket, and `--share` prices the whole treatment on one dial. It stays a supervision
refinement rather than a pseudo-label because the reader's bracket is kept.

WHAT IT CANNOT SETTLE. Whether the model orders the rung in the ELEVEN columns that are not Effusion.
D300's gold sample holds 2-24 within-band pairs a column outside it, which resolves nothing; the corpus
census says the band is worth addressing in eleven columns and cannot say the model can address it in
eleven. The gold ruler at n=58 with fold-to-fold sd 0.0550 also cannot see a regression smaller than
about a point and a half, so a null here is a null AT THIS RESOLUTION and never a refutation.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.data.labels.targets import MEASURED_AGREEMENT
from core.findings import TARGETS

logger = logging.getLogger(__name__)

MIDDLE = 2
"""Which of the reader's five rungs is the one that asserts almost nothing."""

RAW = 0
CALIBRATED = 1
"""Which column of `MEASURED_AGREEMENT` a table is written in — the level the reader emits, or the rate
it was measured at. `report_labels_v2.csv` carries the first and `report_labels_v3.csv` the second, and
they are the SAME cells at different values, so the rungs a re-spread brackets against must be read off
the same scale the table is stored in or the treatment reaches nothing and reports a clean no-op."""

SCALES: dict[str, int] = {"raw": RAW, "calibrated": CALIBRATED}
"""Declaration-only dispatch: the name an arm says, and the column of the level table it selects."""

TOLERANCE = 1e-4
"""How close a stored value must sit to the rung to be one of its cells.

The tables are written through a CSV, so a rung arrives as text and returns as a float that is equal to
the constant only up to the round trip. An exact comparison silently reads zero band cells and reports a
clean no-op, which is the failure this constant exists to make impossible."""

MIDPOINT = 0.5
"""Where a percentile with no spread sits, so a single band cell in a column is left exactly on the rung."""


class BandResolution:
    """One label table, with the reader's middle rung re-spread from a model's out-of-fold order."""

    def __init__(self, table: pd.DataFrame, share: float, scale: int = RAW) -> None:
        self.table = table
        self.share = share
        self.band = MEASURED_AGREEMENT[MIDDLE][scale]
        self.lower = MEASURED_AGREEMENT[MIDDLE - 1][scale]
        self.upper = MEASURED_AGREEMENT[MIDDLE + 1][scale]

    def banded(self, target: str) -> "pd.Series[bool]":
        """The rows whose value for this finding sits on the middle rung — the only rows this may write."""
        return (self.table[target] - self.band).abs() < TOLERANCE

    def percentiles(self, ordering: Float[NDArray[np.float64], " studies"]) -> Float[NDArray[np.float64], " studies"]:
        """The order mapped onto [0, 1], computed WITHIN the band cells of one column and nowhere else.

        A percentile over the whole column would carry the reader's own rung structure into the spread —
        the band cells would land wherever the rungs above and below left room, which is a fact about the
        reader rather than about the pixels. Ranking only the cells being rewritten is what makes the
        spread a reading of the model's opinion of THOSE cells.
        """
        if len(ordering) < 2:  # noqa: PLR2004
            return np.full(len(ordering), MIDPOINT)
        rank = np.argsort(np.argsort(ordering)).astype(np.float64)
        return rank / (len(ordering) - 1)

    def reach(self) -> float:
        """How far either way a cell may move at share 1.0 — the NEARER neighbour's distance, both sides.

        THE BRACKET IS SYMMETRIC ON PURPOSE, AND IT COSTS SOMETHING TO MAKE IT SO. The reader's own two
        neighbours are not equidistant on either scale — raw is 0.40 below the rung and 0.14 above,
        calibrated 0.242 and 0.310 — so a spread that simply reached both would move the band's MEAN as
        well as its order. On the raw scale that is 0.68 -> 0.615, a systematic downward pull of 0.065
        confounded with the ordering under test: an arm that gained could not say whether the pixels
        ordered the rung or whether 0.68 was merely too high, and D300's own calibration says 0.68 IS
        too high, which is exactly what makes the two indistinguishable. Reaching only the nearer
        neighbour's distance on both sides leaves the band's mean where the reader put it and moves ONE
        thing. The price is the wider side's unused room, which is the cheaper of the two mistakes.
        """
        return min(self.band - self.lower, self.upper - self.band)

    def spread(self, percentile: Float[NDArray[np.float64], " studies"]) -> Float[NDArray[np.float64], " studies"]:
        """Where a band cell's target moves, given its rank among the band cells of its own column."""
        return self.band + self.share * self.reach() * (2.0 * percentile - 1.0)

    def resolved(self, scores: Float[NDArray[np.float64], "studies targets"]) -> pd.DataFrame:
        """A copy of the table with every finding's band cells re-spread by that finding's own scores.

        The score array is indexed by the table's own rows, so alignment is the caller's job and a
        mis-join cannot be repaired here — it would show up as a spread that reads like noise and still
        writes a plausible table.
        """
        written = self.table.copy()
        for index, target in enumerate(TARGETS):
            band = self.banded(target)
            if not band.any():
                continue
            written.loc[band, target] = self.spread(self.percentiles(scores[band.to_numpy(), index]))
        return written

    def census(self, scores: Float[NDArray[np.float64], "studies targets"]) -> list[str]:
        """How many cells move, per column, and how far — the count IS the reading of the treatment."""
        written = self.resolved(scores)
        cells = len(self.table) * len(TARGETS)
        moved = sum(int(self.banded(target).sum()) for target in TARGETS)
        lines = [
            f"band {self.band:.4f} re-spread onto [{self.lower:.4f}, {self.upper:.4f}] at share {self.share:.2f}",
            f"{moved} of {cells} cells on the rung ({100.0 * moved / cells:.2f}% of the table)",
            "",
        ]
        for target in TARGETS:
            band = self.banded(target)
            after = written.loc[band, target]
            lines.append(
                f"  {target:<18} {int(band.sum()):>5} cells  {100.0 * band.mean():>5.2f}% of column  "
                f"-> mean {after.mean():.4f}  range [{after.min():.4f}, {after.max():.4f}]"
            )
        return lines


def main() -> None:  # devtools-ignore: test-mirror
    """Write a label table whose middle rung carries the model's out-of-fold order within each finding."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--table", type=Path, required=True, help="the label table to re-spread")
    command.parser.add_argument("--models", type=Path, default=None, help="where the fold records live")
    command.parser.add_argument("--stitch", required=True, help="glob for the five out-of-fold records")
    command.parser.add_argument("--share", type=float, default=0.5, help="how far toward a neighbouring rung")
    command.parser.add_argument("--scale", default="raw", choices=sorted(SCALES), help="the table's own scale")
    command.parser.add_argument("--out", type=Path, required=True, help="where the re-spread table is written")
    arguments = command.parsed()
    root = command.rooted(arguments)
    models = arguments.models or root.models()

    parts = tuple(sorted(models.glob(arguments.stitch)))
    if not parts:
        message = f"no fold records under {models} matching {arguments.stitch}"
        raise FileNotFoundError(message)
    held = [np.load(part, allow_pickle=False) for part in parts]
    names = [str(name) for record in held for name in record["studies"]]
    predictions = np.concatenate([record["predictions"] for record in held], axis=0)
    if len(set(names)) < len(names):
        message = f"{len(names) - len(set(names))} studies appear in two folds — the stitch is not a partition"
        raise ValueError(message)

    table = pd.read_csv(arguments.table)
    placed = {name: row for row, name in enumerate(names)}
    rows = table["StudyInstanceUID"].map(placed)
    if rows.isna().any():
        message = f"{int(rows.isna().sum())} of {len(table)} table rows have no out-of-fold prediction"
        raise ValueError(message)
    scores = predictions[rows.to_numpy(dtype=np.int64)].astype(np.float64)

    resolution = BandResolution(table, arguments.share, SCALES[arguments.scale])
    command.report(resolution.census(scores))
    out: Path = arguments.out
    out.parent.mkdir(parents=True, exist_ok=True)
    resolution.resolved(scores).to_csv(out, index=False)
    command.report([f"{len(table)} rows -> {out}"])


if __name__ == "__main__":
    main()
