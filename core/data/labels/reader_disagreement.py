"""Where independent readers of the same report disagree: `python -m core.data.labels.reader_disagreement`.

WHAT THIS IS FOR. The hosts labelled the gold 58 and the hidden test FROM THE IMAGES; every table we
train on is read from report TEXT. `kneemri.evaluation.table_separation` then showed that recomposing
the blend is not a lever — every recipe lives inside the gold ruler's +/-0.008. So the remaining move on
the label axis is not a better reader, it is how much the trainer BELIEVES each cell.

`confident_labels` already drops the cells a reader marked UNK, but a reader can only flag what it KNOWS
it could not read. The damaging class is confident-and-wrong, and the only witness we own for it is other
readers of the same sentence: where four independent readers split, the TEXT is ambiguous, whatever any
one of them reports about its own confidence.

THIS MODULE MEASURES, IT DOES NOT CORRECT. It emits a per-finding disagreement rate, because that rate is
the gate on the whole lane (knee-1ww2): the mechanism story is that hedging concentrates on the small,
focal findings the model already loses (knee-lxi1). If disagreement is FLAT across the twelve findings,
that story is wrong and no arm is licensed by it.

WHAT THE NUMBER IS NOT. Readers do not share a scale — ours emits five levels, an LLM emits a
probability — so an absolute difference mixes disagreement with resolution. The per-finding figures are
therefore comparable to EACH OTHER (every finding is read by the same set of readers on the same scale
mix) and not to any external threshold. It is a ranking instrument, and it is used as one.
"""

import logging
from pathlib import Path

import pandas as pd

from core import cli
from core.data.labels.table_blend import LabelBlend
from core.findings import STUDY

logger = logging.getLogger(__name__)

MIN_READERS = 2
"""A cell needs two readings before it can disagree; one reader's opinion is not a split."""

SPLIT = 0.34
"""How far apart two readings must sit before the cell counts as SPLIT. Our own scale steps by 0.25, so
this is 'more than one level apart' — a threshold on the coarsest reader in the set, which is the only
reader whose steps are meaningful. Reported alongside the scale-free mean, never instead of it."""


class ReaderDisagreement:
    """Per-finding spread across independent readers of the same reports, on the base table's rows."""

    def __init__(self, blend: LabelBlend) -> None:
        self.blend = blend

    def spread(self, finding: str) -> pd.Series:
        """Each study's reader spread on one finding: max reading minus min, over the readers present.

        Studies read by fewer than `MIN_READERS` are dropped rather than scored zero — a cell only one
        reader reached is unmeasured here, and calling that agreement would put the corpus's coverage
        holes into the column that is supposed to carry ambiguity.
        """
        readings = pd.concat(self.blend.readings(finding), axis=1)
        covered = readings[readings.notna().sum(axis=1) >= MIN_READERS]
        return covered.max(axis=1) - covered.min(axis=1)

    def profile(self) -> pd.DataFrame:
        """One row per finding: how many studies were readable, and how far the readers sat apart."""
        rows = [
            {
                "finding": finding,
                "studies": int(spread.size),
                "mean_spread": float(spread.mean()),
                "split_rate": float((spread > SPLIT).mean()),
            }
            for finding, spread in ((finding, self.spread(finding)) for finding in self.blend.findings())
        ]
        return pd.DataFrame(rows).sort_values("mean_spread", ascending=False).reset_index(drop=True)

    def cells(self) -> pd.DataFrame:
        """Every study's spread on every finding, wide, for a downstream weighting to consume.

        Carries the base table's study column so it joins to a label table by name and never by position —
        the defect `SupervisionGap.reported` shipped, where a table missing one finding scored every later
        one under its neighbour's name.
        """
        table = pd.DataFrame(index=self.blend.base.index)
        for finding in self.blend.findings():
            table[finding] = self.spread(finding)
        return table.rename_axis(STUDY).reset_index()

    def report(self) -> list[str]:
        """The profile as lines, most-disputed finding first, with the gate stated next to the numbers."""
        profile = self.profile()
        lines = [*self.blend.coverage(), f"split threshold {SPLIT}, {MIN_READERS}+ readers per cell"]
        lines += [
            f"{row.finding:<24} {row.studies:5d} studies  mean {row.mean_spread:.4f}  split {row.split_rate:.3f}"
            for row in profile.itertuples()
        ]
        span = float(profile["mean_spread"].max() - profile["mean_spread"].min())
        lines.append(f"spread across findings {span:.4f} — flat here refutes the concentration story")
        return lines


def main() -> None:
    """Measure where independent readers of the same reports disagree, per finding."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--base", type=Path, required=True, help="the table whose rows lead")
    command.parser.add_argument("--foreign", type=Path, nargs="+", required=True, help="other readers")
    command.parser.add_argument("--out", type=Path, help="optional per-cell spread table")
    arguments = command.parsed()

    blend = LabelBlend(LabelBlend.read(arguments.base), [LabelBlend.read(path) for path in arguments.foreign])
    disagreement = ReaderDisagreement(blend)
    command.report(disagreement.report())

    destination = arguments.out
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        disagreement.cells().to_csv(destination, index=False)
        command.report([f"per-cell spread -> {destination}"])


if __name__ == "__main__":
    main()
