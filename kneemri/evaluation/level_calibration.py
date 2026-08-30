"""The report reader's confidence levels, re-mapped onto the rate at which they are actually positive.

WHAT IS BROKEN, measured on the 58 radiologist-annotated studies (696 cells, of which 684 are read;
see `GoldRows`). The reader's ORDERING is good — mean per-finding rank AUC 0.866, ACL 0.987 — and its
SCALE is not. Pooled over all twelve findings its five levels carry gold-positive rates of 0.041 /
0.136 / 0.378 / 0.688 / 0.958. The two ends are nearly right, which is why nothing downstream noticed.
The 0.68 bucket is 98 cells, 14% of everything, 62% of them actually negative, trained today as a 0.68
positive.

WHY POOLED AND NOT PER-FINDING. Fifty-eight studies give a finding at most 58 cells spread over five
levels, so a per-finding map would fit noise; pooled it is five parameters on 684 cells. The ordering
being already good is exactly what licenses that: a per-finding correction would be correcting the part
that works.

WHY MONOTONE, even though nothing currently pools. The reader emits a confidence, so a higher level
must not mean a lower rate, and the estimator has to say that rather than a threshold hiding it. The
only inversion ever observed here was an ARTEFACT: a twelve-cell 0.50 bucket reading 0.417 against the
98-cell 0.68's 0.378, where 0.50 is `GoldRows`' unread fill and not a level at all (see `vocabulary`).
With the fill excluded the observed series is already non-decreasing and this step is the identity —
which is the outcome to want from it, not a reason to drop it.

WHAT THIS IS NOT. It does not improve the reader and it cannot: the ordering is what a better reader
would move. It restates the same ordering at the values the annotations support.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.gold_rows import PRESENT, GoldRows

logger = logging.getLogger(__name__)

DECIMALS = 6
"""Where a float level is rounded before it is treated as a category.

The reader writes its levels as floats, so two cells of the same level can differ in the last bit and
become two buckets of one cell each. Six decimals is far below the spacing of any level the reader
emits and far above float32's noise.
"""


@dataclass(frozen=True)
class LevelRates:
    """Each level the reader emits, how often it occurred, and how often it was right.

    `raw` is the observed rate and `fitted` is that rate made monotone. Both are kept because the
    difference between them is the whole evidence for the pooling: a caller writing up the map has to
    be able to say which cells moved and by how much.
    """

    levels: Float[np.ndarray, " levels"]
    counts: Float[np.ndarray, " levels"]
    raw: Float[np.ndarray, " levels"]
    fitted: Float[np.ndarray, " levels"]

    def line(self) -> str:
        """One row per level, which is the table an interpretation quotes."""
        rows = [
            f"  {level:.2f}  n={int(count):4d}  raw {observed:.3f}  fitted {value:.3f}"
            for level, count, observed, value in zip(self.levels, self.counts, self.raw, self.fitted, strict=True)
        ]
        return "\n".join(rows)


class LevelCalibration:
    """Fits the reader's levels onto their empirical gold rate, and applies the fit to a table."""

    def vocabulary(self, table: Float[np.ndarray, "studies targets"]) -> Float[np.ndarray, " levels"]:
        """The levels the reader actually emits, taken from the whole table rather than a subset.

        THE FIT NEEDS THIS AND CANNOT DERIVE IT. `GoldRows.reported` lands a cell the reader could not
        read at PRESENT, so over the 58 gold rows an unread cell is indistinguishable from a confident
        0.5 — and the first fit here pooled twelve such cells as if they were a level, moving the
        98-cell bucket by 0.004 on evidence about missingness. Over all 4,406 rows the reader's
        vocabulary is five levels and 0.5 is not among them, which is the observation that catches it.
        """
        flat = np.round(np.asarray(table, dtype=np.float64).ravel(), DECIMALS)
        return np.unique(flat[~np.isnan(flat)])

    def observed(
        self,
        reader: Float[np.ndarray, "studies targets"],
        gold: Float[np.ndarray, "studies targets"],
        vocabulary: Float[np.ndarray, " levels"],
    ) -> LevelRates:
        """The rate at which each reader level is a gold positive, pooled over findings.

        Both arrays are flattened together: the claim being estimated is about the reader's SCALE, which
        is one scale shared by all twelve columns, not twelve scales that happen to use the same numbers.

        Cells outside `vocabulary` are DROPPED rather than bucketed. They are cells the reader did not
        read, and a rate computed over them is a fact about which reports omit a finding, not about how
        confident the reader is when it speaks.
        """
        flat = np.round(np.asarray(reader, dtype=np.float64).ravel(), DECIMALS)
        positive = (np.asarray(gold, dtype=np.float64).ravel() > PRESENT).astype(np.float64)

        levels = np.asarray([level for level in np.unique(flat) if level in set(vocabulary.tolist())])
        read = np.isin(flat, levels)
        flat, positive = flat[read], positive[read]
        counts = np.array([float((flat == level).sum()) for level in levels], dtype=np.float64)
        raw = np.array([float(positive[flat == level].mean()) for level in levels], dtype=np.float64)
        return LevelRates(levels=levels, counts=counts, raw=raw, fitted=self.monotone(raw, counts))

    def monotone(
        self, rates: Float[np.ndarray, " levels"], weights: Float[np.ndarray, " levels"]
    ) -> Float[np.ndarray, " levels"]:
        """Pool-adjacent-violators: the closest non-decreasing series to `rates` under `weights`.

        A violated pair is replaced by their weighted mean, which is why the twelve-cell bucket moves
        almost all the way to its 98-cell neighbour rather than dragging it: the estimator spends its
        correction where the evidence is thin.
        """
        values = [float(rate) for rate in rates]
        mass = [float(weight) for weight in weights]
        width = [1] * len(values)
        index = 0
        while index < len(values) - 1:
            if values[index] <= values[index + 1]:
                index += 1
                continue
            total = mass[index] + mass[index + 1]
            values[index] = (values[index] * mass[index] + values[index + 1] * mass[index + 1]) / total
            mass[index] = total
            width[index] += width[index + 1]
            del values[index + 1], mass[index + 1], width[index + 1]
            index = max(index - 1, 0)

        spread = np.repeat(np.array(values, dtype=np.float64), np.array(width, dtype=np.int64))
        return np.asarray(spread, dtype=np.float64)

    def applied(
        self, values: Float[np.ndarray, "studies targets"], rates: LevelRates
    ) -> Float[np.ndarray, "studies targets"]:
        """`values` with every level replaced by its fitted rate, unknown levels left alone.

        A level the fit never saw is left untouched rather than interpolated: the whole point of the map
        is that it is measured, and a value invented for an unseen level would be the reader's own scale
        wearing the fit's clothes.
        """
        mapped = np.asarray(values, dtype=np.float64).copy()
        rounded = np.round(mapped, DECIMALS)
        for level, value in zip(rates.levels, rates.fitted, strict=True):
            mapped[rounded == level] = value
        return np.asarray(mapped, dtype=np.float32)


def main() -> None:
    """Fit the map on the annotated rows and write the recalibrated training table.

    The fit is taken from `--labels` read RAW over the gold studies and then applied to the WHOLE table,
    those studies included: a study's report reading is recalibrated whether or not a radiologist also
    read its images, because the recalibration is a statement about the reader, not about the study.
    `Targets.load` still overwrites the annotated rows with their annotations at training time.
    """
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, required=True, help="the report table to recalibrate")
    command.parser.add_argument("--out", type=Path, required=True, help="where to write the recalibrated table")
    arguments = command.parsed()

    root = command.rooted(arguments)
    identifiers, gold = GoldRows().annotated(root.raw() / "train.csv")
    reader = GoldRows().reported(arguments.labels, identifiers)

    table = pd.read_csv(arguments.labels)
    columns = [target for target in TARGETS if target in table.columns]
    values = table[columns].to_numpy(dtype=np.float32)

    calibration = LevelCalibration()
    rates = calibration.observed(reader, gold, calibration.vocabulary(values))
    logger.info(f"fitted over {int(rates.counts.sum())} read cells of {gold.size}\n{rates.line()}")

    table[columns] = calibration.applied(values, rates)
    table.to_csv(arguments.out, index=False)
    logger.info(f"wrote {len(table)} rows over {len(columns)} findings to {arguments.out}")


if __name__ == "__main__":
    main()
