"""Is one calibration constant right for all twelve findings? `python -m core.data.labels.level_agreement`.

`Supervision.applied` replaces each reader level with the rate it was measured to agree at, and those
rates (`MEASURED_AGREEMENT`) are ONE NUMBER PER LEVEL, pooled across every finding. `calib_span` spent an
arm on that remap and came back −0.0020 with the per-label pattern INVERTED against its own prediction:
Effusion, named as the label with most to gain, was the worst row on the board at −0.0212.

This module measures the same object the constants estimate, WITHOUT pooling: for each finding and each
level, how often the radiologist agreed. It needs no report text, so unlike `severity_boundary` it reads
the non-English 59.9% too, and its Effusion row rests on 27 cells rather than that module's 12.

A LEVEL THAT CROSSES 0.5 IS NOT THE SAME MISTAKE AS ONE THAT DOES NOT. The remap writes a target VALUE,
and the metric ranks, so a constant that lands on the wrong SIDE of the finding's own rate inverts that
finding's low-grade positives, while one that merely overshoots on the correct side only softens them.
`inverted` names the findings where the pooled constant does the first thing.
"""

import logging
from pathlib import Path

import pandas as pd

from core import cli
from core.data.labels.targets import ASSERTS_PRESENCE, MEASURED_AGREEMENT
from core.findings import STUDY, TARGETS

logger = logging.getLogger(__name__)

LABELS = "report_labels_v2.csv"
TOLERANCE = 1e-6
"""Levels are written as decimals by the reader and read back as floats; this compares them as levels."""

THIN = 5
"""Below this many annotated cells a rate is not reported as one. Two cells agreeing is not a rate, and a
pooled constant is not refuted by a number whose own interval spans it."""


class LevelAgreement:
    """Per finding and per reader level, how often the radiologist agreed with the reader's assertion."""

    def __init__(self, measured: tuple[tuple[float, float], ...] = MEASURED_AGREEMENT) -> None:
        self.measured = measured
        self.levels = tuple(level for level, _ in measured)
        self.pooled = dict(measured)

    def rates(self, reader: pd.DataFrame, annotated: pd.DataFrame) -> pd.DataFrame:
        """One row per (finding, level): the annotated positive rate there, and how many cells it rests on.

        Joined on the study identifier rather than positionally, because the reader table covers the whole
        corpus and the annotations cover 58 of it — a positional read would silently score the wrong rows.
        """
        shared = reader.index.intersection(annotated.index)
        rows = []
        for finding in TARGETS:
            if finding not in reader.columns:
                continue
            value = reader.loc[shared, finding]
            gold = annotated.loc[shared, finding]
            for level in self.levels:
                cells = (value - level).abs() < TOLERANCE
                cells &= gold.notna()
                rate = float(gold[cells].mean()) if cells.any() else float("nan")
                rows.append((finding, level, rate, int(cells.sum())))
        return pd.DataFrame(rows, columns=["finding", "level", "rate", "studies"])

    def spread(self, rates: pd.DataFrame) -> list[str]:
        """Each level's per-finding rates beside the one constant standing in for all of them."""
        lines = []
        for level in self.levels:
            rows = rates[(rates["level"] == level) & (rates["studies"] >= THIN)]
            if rows.empty:
                continue
            lines.append(
                f"  level {level:.2f}  pooled constant {self.pooled[level]:.3f}   "
                f"per-finding {rows['rate'].min():.3f} to {rows['rate'].max():.3f} "
                f"over {len(rows)} findings with {THIN}+ annotated cells"
            )
        return lines

    def inverted(self, rates: pd.DataFrame) -> list[str]:
        """Findings whose own rate sits on the OTHER side of 0.5 from the constant that replaces it.

        This is the row that changes a ranking rather than a magnitude, and it is the shape `calib_span`
        measured as its largest negative.
        """
        thick = rates[(rates["studies"] >= THIN) & rates["rate"].notna()]
        found = []
        for _, row in thick.iterrows():
            constant = self.pooled[row["level"]]
            if (row["rate"] > ASSERTS_PRESENCE) != (constant > ASSERTS_PRESENCE):
                found.append(
                    f"  {row['finding']:18s} level {row['level']:.2f}  own rate {row['rate']:.3f}  "
                    f"constant {constant:.3f}  ({int(row['studies'])} annotated cells)"
                )
        return found


def main() -> None:
    """Measure each finding's own agreement rate at each reader level, against the pooled constant."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, help="the reader table; defaults to report_labels_v2.csv")
    arguments = command.parsed()

    root = command.rooted(arguments)
    reader = pd.read_csv(arguments.labels or root.processed() / "labels" / LABELS).set_index(STUDY)
    annotated = pd.read_csv(root.raw() / "train.csv").set_index(STUDY).dropna(subset=TARGETS, how="all")

    agreement = LevelAgreement()
    rates = agreement.rates(reader, annotated)
    command.report([f"{len(reader.index.intersection(annotated.index))} annotated studies in the reader table"])
    command.report(["per-finding spread against the pooled constant:", *agreement.spread(rates)])
    # The rows that invert rather than soften — a ranking metric feels these and not the others.
    command.report(["findings the pooled constant puts on the WRONG SIDE of 0.5:", *agreement.inverted(rates)])


if __name__ == "__main__":
    main()
