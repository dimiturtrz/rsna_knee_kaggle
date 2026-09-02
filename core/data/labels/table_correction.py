"""Correct a derived label table where something outside it knows better: `python -m core.data.labels.table_correction`.

`table_blend`'s own conclusion is that a better label table has to come from a better READER, not a new
weighting of the readers already here. This module is the cheapest form of a better reader: the table
stays, and a rule that knows something the reader was not told rewrites the cells it names.

TWO RULES, EACH ARGUED SEPARATELY, AND BOTH ARE ORDER-PRESERVING WHERE THEY TOUCH NOTHING.

`fracture_contusion` — report-derived Fracture misses bone-contusion impaction injury. Of the four
annotated studies where the reader said UNK and the radiologist said fracture, all four reports describe
a bone bruise and none uses a fracture word: Contusion is the report's name for what the radiologist
scored. Lifting Fracture by a quarter of Contusion and rescaling back onto the column's own range reads
+0.0632 on the gold Fracture column and exactly +0.0000 on the other eleven, permutation p = 0.0022.

`rubric_subthreshold` — the host PUBLISHED the rules the competition labels were graded under, and our
reader was never given them. A report saying "intrasubstance degeneration without definite tear" earns a
grade above absent from a transcriber and is graded 0 by the host. `rubric_gap` already decides this
lexically, per cell, from the published criteria — no ground truth and no language model are needed for
the decision, which is why this rule can be applied at all. It is applied ONLY under the rules the
annotations BACK, computed here rather than hardcoded, so a rule the gold set refuses (MCL, Effusion,
the three OA compartments) can never enter a training table by being written down once.

NEITHER RULE HARD-FLIPS A CELL. `overcall_audit`'s own audit puts the clean extractor failure rate at
15%, so a rule that is right most of the time is applied as a pull toward the column's negative floor
with that residual left standing, never as a zero. A hard flip asserts the rule is never wrong, and this
one is wrong about one cell in seven.

THE ORACLE THIS MODULE IS PROVED BY: `--rule fracture_contusion` over `blend_hybrid.csv` reproduces
`blend_hybrid_fx.csv` — that table was written by hand before this module existed and its gold numbers
are already recorded. Measured: zero difference on all twelve target columns. NOT byte-identical, and
the reason is worth stating rather than hiding — the table also carries the reader's `__conf` columns,
and a pandas read/write round-trip re-formats their floats. The oracle is over the labels, which is what
a training run reads; the confidence columns are carried, not corrected.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from core import cli
from core.data.labels.rubric_gap import RUBRIC, RubricGap
from core.findings import STUDY, TARGETS

logger = logging.getLogger(__name__)

DECIMALS = 4
"""The written precision of every label table in this project; the rescale is exact above it."""


@dataclass(frozen=True)
class LabelSources:
    """Everything a correction may read: the table it rewrites, the reports behind it, the annotations."""

    table: pd.DataFrame
    reports: pd.DataFrame
    truth: pd.DataFrame


class Correction(Protocol):
    """A named rewrite of one label table into another with the same rows, columns and scale."""

    def applied(self, sources: LabelSources) -> pd.DataFrame:
        """The corrected table."""


@dataclass(frozen=True)
class PartnerLift:
    """`column := rescale(column + weight * partner)` — a finding the reader named under another name."""

    column: str
    partner: str
    weight: float

    def rescaled(self, values: pd.Series, low: float, high: float) -> pd.Series:
        """`values` mapped linearly onto `[low, high]`, which is order-preserving and so AUC-identical."""
        floor, ceiling = float(values.min()), float(values.max())
        return (values - floor) / (ceiling - floor) * (high - low) + low

    def applied(self, sources: LabelSources) -> pd.DataFrame:
        """The table with `column` lifted by its partner and mapped back onto its own original range."""
        table = sources.table.copy()
        current = table[self.column]
        lifted = current + self.weight * table[self.partner]
        table[self.column] = self.rescaled(lifted, float(current.min()), float(current.max())).round(DECIMALS)
        return table


@dataclass(frozen=True)
class RubricPull:
    """Cells the host's published rule grades negative, pulled toward the column's own negative floor.

    `residual` is what survives the pull: the probability the flagged cell is positive anyway, which is
    the extractor's measured clean-failure rate. At 0.0 this would be a hard flip and would assert the
    lexical rule is never wrong.
    """

    residual: float

    def flagged(self, sources: LabelSources) -> list[tuple[str, str]]:
        """The `(study, column)` cells the BACKED rules contradict — the population this rule rewrites."""
        gap = RubricGap()
        cells: list[tuple[str, str]] = []
        for rule in RUBRIC:
            marks = gap.table(rule, sources.reports, sources.table)
            if not gap.backed(rule, marks, sources.truth):
                continue
            cells.extend((str(row.study), str(row.column)) for row in gap.contradicted(marks).itertuples())
        return cells

    def applied(self, sources: LabelSources) -> pd.DataFrame:
        """The table with every backed contradiction pulled toward its column's floor by `1 - residual`."""
        table = sources.table.set_index(STUDY)
        by_column: dict[str, list[str]] = {column: [] for column in TARGETS}
        for study, column in self.flagged(sources):
            by_column[column].append(study)
        for column, studies in by_column.items():
            floor = float(table[column].min())
            pulled = floor + self.residual * (table.loc[studies, column] - floor)
            table.loc[studies, column] = pulled.round(DECIMALS)
        return table.reset_index()


FRACTURE_CONTUSION = "fracture_contusion"
RUBRIC_SUBTHRESHOLD = "rubric_subthreshold"

CORRECTIONS: dict[str, Correction] = {
    FRACTURE_CONTUSION: PartnerLift("Fracture", "Contusion", 0.25),
    RUBRIC_SUBTHRESHOLD: RubricPull(0.15),
}


@dataclass(frozen=True)
class CorrectionSequence:
    """Named rules applied in order — itself a `Correction`, so a composition is not a special case."""

    rules: tuple[str, ...]

    def applied(self, sources: LabelSources) -> pd.DataFrame:
        """Each rule in turn, reading the table the one before it wrote."""
        table = sources.table
        for name in self.rules:
            table = CORRECTIONS[name].applied(LabelSources(table, sources.reports, sources.truth))
        return table


def main() -> None:
    """Write a corrected copy of a label table. The source table is never modified in place."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--base", type=Path, required=True, help="the table to correct")
    command.parser.add_argument("--rule", nargs="+", required=True, choices=sorted(CORRECTIONS))
    command.parser.add_argument("--out", type=Path, required=True, help="where the corrected table is written")
    arguments = command.parsed()

    root = command.rooted(arguments)
    truth = pd.read_csv(root.raw() / "train.csv")
    sources = LabelSources(pd.read_csv(arguments.base), truth[[STUDY, "Report"]], truth)
    table = CorrectionSequence(tuple(arguments.rule)).applied(sources)

    moved = [column for column in TARGETS if not table[column].equals(sources.table[column])]
    table.to_csv(arguments.out, index=False)
    command.report(
        [
            f"{len(table)} rows written to {arguments.out} under {', '.join(arguments.rule)}",
            f"  columns moved: {', '.join(moved) or 'none'}",
        ]
    )


if __name__ == "__main__":
    main()
