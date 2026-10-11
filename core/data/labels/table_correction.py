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

`effusion_reread` — a second reader pass over the same reports, grafted in as a whole column. It is here
because the table it built SHIPPED to the board with no builder in the repo, not because it is endorsed;
its own docstring carries the adjudication that prices it.

THE SECOND ORACLE: `--rule effusion_reread` over `blend_hybrid_fx.csv` reproduces `blend_hybrid_fx_v4b.csv`
— measured 2026-09-08 over the 4,406 shared studies, zero difference on eleven of twelve columns, the
Effusion verdict column exact, and 5 of 4,406 Effusion VALUES off (0.9989). Those five all read
`(origin 0.68, donor 0.68)` — a level the re-read did not change and the hand-built table moved anyway,
so the residual is in the shipped table, not in the rule.

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
from core.data.labels.finding_mentions import FindingMentions
from core.data.labels.negation_scope import NegationScope
from core.data.labels.report_language import ReportLanguages
from core.data.labels.rubric_gap import RUBRIC, RubricGap, RuleEvidence
from core.findings import STUDY, TARGETS
from core.paths import LABEL_TABLE

logger = logging.getLogger(__name__)

DECIMALS = 4
"""The written precision of every label table in this project; the rescale is exact above it."""


@dataclass(frozen=True)
class LabelSources:
    """Everything a correction may read: the table it rewrites, the reports behind it, the annotations.

    `labels` is the directory the label tables live in, so a rule whose knowledge is ANOTHER TABLE can name
    its donor by filename instead of by an absolute path baked into a registry entry.
    """

    table: pd.DataFrame
    reports: pd.DataFrame
    truth: pd.DataFrame
    labels: Path


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
            if not gap.backed(RuleEvidence(rule, marks, sources.truth)):
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


@dataclass(frozen=True)
class NegationPull:
    """Positive cells whose report DENIES the finding everywhere it names it, pulled toward the floor.

    `NegationScope` was built, measured and then consumed by nothing that produces a label: its two parts
    took the negation category from 18 to 12 on disputed cells and from 16 to 3 on agreed ones, and the
    only importer is `overcall_audit`. This is the rule that spends it. `report_reader` is a language
    model, so the detector cannot be wired into the read itself without re-reading 4,407 studies — it
    acts here instead, as a post-hoc pull over the table the read produced.

    EVERY located mention must be denied, not one of them. A report that names a finding twice and denies
    it once is describing two things, and the rule declines rather than guessing which. That is also what
    keeps the rule away from `severity_boundary`'s retired territory: it reads DENIAL, never severity.

    `residual` is the extractor's measured clean-failure rate, the same 0.15 `RubricPull` carries, and it
    is what makes this a pull rather than the hard flip the rule is not entitled to.
    """

    residual: float

    def denials(self, sources: LabelSources) -> list[tuple[str, str]]:
        """The `(study, column)` positive cells every one of whose located mentions is denied."""
        mentions = FindingMentions()
        scope = NegationScope(mentions)
        languages = ReportLanguages()
        table = sources.table.set_index(STUDY)
        positive = {column: set(table.index[table[column] > float(table[column].min())]) for column in TARGETS}
        cells: list[tuple[str, str]] = []
        for row in sources.reports.itertuples():
            study, report = str(getattr(row, STUDY)), str(row.Report)
            if study not in table.index:
                continue
            language = languages.language(report)
            if not mentions.covers(language):
                continue
            for column in TARGETS:
                if study not in positive[column]:
                    continue
                spans = mentions.mentions(report, language, column)
                if spans and all(scope.denied(span, language, column) for span in spans):
                    cells.append((study, column))
        return cells

    def applied(self, sources: LabelSources) -> pd.DataFrame:
        """The table with every denied positive pulled toward its column's floor by `1 - residual`."""
        table = sources.table.set_index(STUDY)
        by_column: dict[str, list[str]] = {column: [] for column in TARGETS}
        for study, column in self.denials(sources):
            by_column[column].append(study)
        for column, studies in by_column.items():
            if not studies:
                continue
            floor = float(table[column].min())
            pulled = floor + self.residual * (table.loc[studies, column] - floor)
            table.loc[studies, column] = pulled.round(DECIMALS)
        return table.reset_index()


VERDICT = "__verdict"
CONF = "__conf"


@dataclass(frozen=True)
class ColumnGraft:
    """One column replaced by a DONOR table's re-read of it: `column := donor[column]`, verdict and all.

    This is the third shape of correction and the only one that hard-flips, because a re-read is not a
    rule with a failure rate — it is a second pass by the same reader over the same reports, and half a
    re-read is not a reading. The value is carried through `levels`, the donor's own level ladder mapped
    onto the base column's scale, because a blended base column is continuous while a reader's column
    sits on five levels; grafting the donor's raw value would put the graft on a different scale from the
    eleven columns beside it.

    IT IS RECORDED HERE BECAUSE IT ALREADY SHIPPED, NOT BECAUSE IT IS ENDORSED. `blend_hybrid_fx_v4b` is
    `blend_hybrid_fx` plus exactly this graft of `report_labels_v4`'s Effusion column, and that table
    trained `labels_v4hybrid_s2026` and the board's `kneemri-v4solo` (0.922) with no builder anywhere in
    the repo — `experiments/README.md` recorded it as `blend_hybrid + fracture_contusion`, which is
    `blend_hybrid_fx`, and the graft went unwritten. D200's blinded adjudication then priced the grafted
    column: of the 520 Effusion cells the graft moves, 508 go from a stated `YES` to `UNK`, and a reader
    working from the report text alone scored those cells `YES` at 0.983. So the one column this graft
    changes is a recall regression, and the table it built is dominated by its own base.
    """

    column: str
    donor: str
    origin: str
    levels: dict[float, float]

    def applied(self, sources: LabelSources) -> pd.DataFrame:
        """The table with `column`'s value, verdict and confidence taken from the donor's re-read.

        A cell's VALUE moves only where the re-read landed on a different level than the `origin` reader
        did — the base table is a blend, so where the two passes agree its blended value is the better
        number and is carried. Verdict and confidence come from the donor everywhere.
        """
        table = sources.table.set_index(STUDY)
        donor = pd.read_csv(sources.labels / self.donor).set_index(STUDY)
        origin = pd.read_csv(sources.labels / self.origin).set_index(STUDY)
        shared = table.index.intersection(donor.index).intersection(origin.index)
        reread = donor.loc[shared, self.column].astype(float).round(DECIMALS)
        moved = shared[reread.to_numpy() != origin.loc[shared, self.column].astype(float).round(DECIMALS).to_numpy()]
        table.loc[shared, self.column + VERDICT] = donor.loc[shared, self.column + VERDICT]
        table.loc[shared, self.column + CONF] = donor.loc[shared, self.column + CONF]
        table.loc[moved, self.column] = [round(self.levels[level], DECIMALS) for level in reread.loc[moved]]
        return table.reset_index()


FRACTURE_CONTUSION = "fracture_contusion"
RUBRIC_SUBTHRESHOLD = "rubric_subthreshold"
EFFUSION_REREAD = "effusion_reread"
MISSED_NEGATION = "missed_negation"

EFFUSION_LEVELS = {0.08: 0.0, 0.28: 0.265, 0.68: 0.795, 0.82: 0.9, 0.94: 0.98}
"""`report_labels_v4`'s five Effusion levels on `blend_hybrid_fx`'s scale. Only the middle two are
CONSTRAINED by the shipped table — every one of the 559 cells the graft moves carries a donor value of
0.28 or 0.68 — so the outer three reproduce the ladder's shape and are not evidence of anything."""

CORRECTIONS: dict[str, Correction] = {
    FRACTURE_CONTUSION: PartnerLift("Fracture", "Contusion", 0.25),
    RUBRIC_SUBTHRESHOLD: RubricPull(0.15),
    EFFUSION_REREAD: ColumnGraft("Effusion", "report_labels_v4.csv", LABEL_TABLE, EFFUSION_LEVELS),
    MISSED_NEGATION: NegationPull(0.15),
}


@dataclass(frozen=True)
class CorrectionSequence:
    """Named rules applied in order — itself a `Correction`, so a composition is not a special case."""

    rules: tuple[str, ...]

    def applied(self, sources: LabelSources) -> pd.DataFrame:
        """Each rule in turn, reading the table the one before it wrote."""
        table = sources.table
        for name in self.rules:
            table = CORRECTIONS[name].applied(LabelSources(table, sources.reports, sources.truth, sources.labels))
        return table


def main() -> None:
    """Write a corrected copy of a label table. The source table is never modified in place."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--base", type=Path, required=True, help="the table to correct")
    command.parser.add_argument("--rule", nargs="+", required=True, choices=sorted(CORRECTIONS))
    command.parser.add_argument("--out", type=Path, required=True, help="where the corrected table is written")
    arguments = command.parsed()

    root = command.rooted(arguments)
    truth = pd.read_csv(root.train_table())
    sources = LabelSources(pd.read_csv(arguments.base), truth[[STUDY, "Report"]], truth, arguments.base.parent)
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
