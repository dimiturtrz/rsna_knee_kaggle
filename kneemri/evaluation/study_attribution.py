"""WHY are the wrong studies wrong? `python -m kneemri.evaluation.study_attribution --run <name>`

`study_errors` ranks the holdout by the pairs each study loses and stops there. A ranking is not an
attribution: it says the worst tenth is 84% over-ranked and carries MORE positives than the body, and
those are facts about the ordering, not about a cause. This module asks the three questions that can
actually be settled from evidence already on disk, in the order that makes each one admissible.

FIRST, IS IT THE STUDY AT ALL? A study that only THIS member ranks badly is a fact about the member —
initialisation, augmentation draw, the epoch it stopped at — and no property of the study can explain it.
So the first reading is the overlap between this run's tail and OTHER members' tails. Under independence
a tail of a tenth reshuffles to a tenth, and anything materially above that is a shared difficulty that
belongs to the study. Only if the tail is shared does asking about reports or acquisition mean anything.

SECOND, IS THE LABEL WRONG? The tail is over-ranked, and over-ranking is the report extractor's dominant
error shape: `reader_agreement` counts 108 confident false positives against 32 misses on the 58 gold
studies. `overcall_audit` already names WHY a reader over-called — a negation whose scope was missed, a
hedge dropped, the other knee, a historical mention, a subthreshold one — and it is reused here rather
than rebuilt, over exactly the same code path, because a control classified by a second implementation
would not be one. GENUINE is the category that matters most: the report plainly asserts the finding and
the model still lost the pairs, so the defect is OURS.

THE CONTROL IS THE WHOLE MEASUREMENT. Every category rate is quoted against the SAME construction over
the body — the study's worst negative cell, classified the same way — because these cue rates are high
everywhere and a tail number alone would read as a cause when it is a base rate.

THIRD, IS IT ACQUISITION? Plane coverage, fluid sensitivity and fat suppression are recorded per series,
so what a study was imaged WITH is a covariate the tail can be compared on. This is the axis most likely
to come back null: the sequence axis has been swept and re-swept, and it is read here to be RETIRED as an
explanation rather than in hope.

WHAT THIS CANNOT DO. Only 58 studies carry a radiologist annotation and roughly a dozen land in a fold-0
holdout, so nothing here is validated against gold truth on this split. The categories are cue-based
readings of the report text; they say what the TEXT supports, not what the knee contained.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.data.labels.overcall_audit import CATEGORIES, GENUINE, LATERALITY, UNLOCATED, OvercallAudit
from core.data.labels.report_language import ReportLanguages
from core.findings import TARGETS
from kneemri.evaluation.scores import MIDPOINT
from kneemri.evaluation.study_errors import StudyDamage, StudyErrors
from kneemri.runs.holdout_record import SUFFIX, HoldoutRecord

LOGGER = logging.getLogger(__name__)

TAIL_FRACTION = 0.1
"""The tail this module reads, kept equal to `study_errors.TAIL_FRACTION` so the two agree on the word."""

MEMBERS = 12
"""How many other members the shared-tail reading compares against. Twelve is enough for the overlap's
own spread to be visible, and every one of them costs a 909-square pair matrix per finding."""

SIGMA = 2.0
"""How many standard errors a tail-versus-body difference must clear before it is called a separation.

Two, because these are seven categories read at once and the loosest bar would hand back a cause on
noise alone about a third of the time across that many."""

SILENCE = (UNLOCATED, LATERALITY)
"""The categories where a NEGATIVE label rests on nothing being said, rather than on something said.

A report-derived zero has two very different groundings. Either the report asserts the finding absent —
a negation this cell carries — or the finding is simply never named for this knee, which is `UNLOCATED`,
or named only for the other one, which is `LATERALITY`. The first is a label a radiologist stands behind.
The second is a label that means "not dictated", and a finding visible on the images but not worth a
sentence produces exactly it. So this pair is where a report-derived zero can be wrong while the MODEL is
right, and it is read as one bin because seven separate bins split the power seven ways.

POST HOC. This grouping was written after seeing the per-category table on `calib_dense_hold_s2026`,
where the two moved together (+0.061 and +0.052) and neither cleared two sigma alone. It is stated here
so that a reader knows it was not pre-registered, and it is confirmed by running this module on OTHER
members rather than by re-quoting the run that suggested it."""

CELL_FLOOR = 5
"""How many cells the SMALLER stratum needs before a category may be called a separation at all.

The normal approximation behind `spread` needs about five successes on each side, and its FIRST run
without this floor reported `hedge_dropped` as separating at 2.5 sigma off 0 cells in the tail against
4 in the body — a standard error of 0.002 computed at a rate of exactly zero, which is not a small
uncertainty but an inapplicable formula. A category too rare to model is reported as too rare, and
the difference is left unread."""

CHANCE = TAIL_FRACTION
"""What tail overlap means NOTHING. Two independent tenths of the same corpus share a tenth of their
members in expectation, so this is the null the observed overlap is read against — not zero."""


@dataclass(frozen=True)
class SharedTail:
    """How much of one run's worst tenth is also worst for other members."""

    members: tuple[str, ...]
    overlap: float
    private: tuple[str, ...]

    def lines(self) -> list[str]:
        """The overlap against its own null, and what the answer licenses asking next."""
        verdict = (
            "SHARED — the tail is a property of the studies, so report and acquisition evidence is admissible"
            if self.overlap > CHANCE * 2
            else "PRIVATE — the tail is largely this member's own, and no property of the study can explain it"
        )
        return [
            f"tail overlap with {len(self.members)} other members: {self.overlap:.3f} against a chance {CHANCE:.3f}",
            f"studies in this tail and no other member's: {len(self.private)}",
            verdict,
        ]


@dataclass(frozen=True)
class LabelCause:
    """The report-text account of the tail's over-ranked cells, beside the same account of the body's."""

    tail: dict[str, int]
    body: dict[str, int]

    @staticmethod
    def spread(tail_rate: float, tail_total: int, body_rate: float, body_total: int) -> float:
        """The standard error of the difference between two independent proportions.

        WITHOUT THIS THE DIFF COLUMN IS A TRAP. A tail of ninety cells puts a standard error near 0.03
        on any middling rate, so a category can move five points and be indistinguishable from the same
        corpus drawn twice. Every difference below twice this is reported as reading NOTHING, which is
        most of them and is the honest answer rather than a disappointing one.
        """
        return float(
            np.sqrt(
                tail_rate * (1.0 - tail_rate) / max(tail_total, 1) + body_rate * (1.0 - body_rate) / max(body_total, 1)
            )
        )

    def lines(self) -> list[str]:
        """Each category as a rate over its own denominator, the difference, and what carries it."""
        tail_total = max(sum(self.tail.values()), 1)
        body_total = max(sum(self.body.values()), 1)
        rows = [f"cells classified: tail {sum(self.tail.values())}, body {sum(self.body.values())}"]
        separated: list[str] = []
        for category in CATEGORIES:
            tail_rate = self.tail[category] / tail_total
            body_rate = self.body[category] / body_total
            difference = tail_rate - body_rate
            error = self.spread(tail_rate, tail_total, body_rate, body_total)
            modelled = min(self.tail[category], self.body[category]) >= CELL_FLOOR
            carries = modelled and abs(difference) > SIGMA * error
            separated += [category] if carries else []
            rows.append(
                f"  {category:<16} tail {self.tail[category]:>4} ({tail_rate:.3f})"
                f"  body {self.body[category]:>4} ({body_rate:.3f})"
                f"  diff {difference:+.3f} +- {error:.3f}"
                f"{'  SEPARATES' if carries else ''}{'' if modelled else '  TOO RARE TO MODEL'}"
            )
        genuine = self.tail[GENUINE] / tail_total - self.body[GENUINE] / body_total
        rows.append(
            f"GENUINE is the category that indicts US rather than the label, and it moves {genuine:+.3f}"
            " from body to tail"
        )
        rows.append(f"categories separating tail from body at {SIGMA:.0f} sigma: {', '.join(separated) or 'NONE'}")
        rows.append(self.silence(tail_total, body_total))
        return rows

    def silence(self, tail_total: int, body_total: int) -> str:
        """The grouped reading: how much of each stratum's negative labels rest on nothing being said."""
        tail_rate = sum(self.tail[category] for category in SILENCE) / tail_total
        body_rate = sum(self.body[category] for category in SILENCE) / body_total
        difference = tail_rate - body_rate
        error = self.spread(tail_rate, tail_total, body_rate, body_total)
        verdict = "SEPARATES" if abs(difference) > SIGMA * error else "reads nothing"
        return (
            f"negative labels grounded in SILENCE ({'+'.join(SILENCE)}): tail {tail_rate:.3f}"
            f"  body {body_rate:.3f}  diff {difference:+.3f} +- {error:.3f}  {verdict} (post hoc)"
        )


@dataclass(frozen=True)
class AcquisitionCause:
    """What the tail studies were imaged with, against what the body studies were imaged with."""

    columns: tuple[str, ...]
    tail: dict[str, float]
    body: dict[str, float]

    def lines(self) -> list[str]:
        """One line per acquisition column, both means and the difference."""
        return [
            f"  {column:<18} tail {self.tail[column]:.3f}  body {self.body[column]:.3f}"
            f"  diff {self.tail[column] - self.body[column]:+.3f}"
            for column in self.columns
        ]


class StudyAttribution:
    """The three readings, each against its own control, over one run's ranked holdout."""

    def __init__(self, audit: OvercallAudit | None = None, languages: ReportLanguages | None = None) -> None:
        self.languages = languages or ReportLanguages()
        self.audit = audit or OvercallAudit(languages=self.languages)

    @staticmethod
    def split(rows: list[StudyDamage]) -> tuple[list[StudyDamage], list[StudyDamage]]:
        """The worst tenth and the rest, cut exactly where `study_errors` cuts its own profile."""
        edge = max(1, int(len(rows) * TAIL_FRACTION))
        return rows[:edge], rows[edge:]

    def shared(self, rows: list[StudyDamage], others: dict[str, list[StudyDamage]]) -> SharedTail:
        """How often this run's tail studies are ALSO in another member's tail.

        Each member is cut at its own tenth rather than at a shared damage threshold, because members
        differ in overall skill and a fixed threshold would read the weaker member as agreeing more.
        """
        tail = {row.study for row in self.split(rows)[0]}
        elsewhere = {name: {row.study for row in self.split(member)[0]} for name, member in others.items()}
        counts = [len(tail & seen) / max(len(tail), 1) for seen in elsewhere.values()]
        private = tail - set().union(*elsewhere.values()) if elsewhere else tail
        return SharedTail(
            members=tuple(sorted(elsewhere)),
            overlap=float(np.mean(counts)) if counts else 0.0,
            private=tuple(sorted(private)),
        )

    @staticmethod
    def negative_cells(
        rows: Sequence[StudyDamage],
        studies: list[str],
        table: Float[NDArray[np.float64], "studies findings"],
        truth: Float[NDArray[np.floating], "studies findings"],
        findings: Sequence[str],
    ) -> pd.DataFrame:
        """Each study's worst NEGATIVE cell — the over-ranking the label says should not be there.

        Negative cells only, because an over-ranked positive cell is not something a report can be wrong
        about in the direction this is testing.
        """
        position = {study: index for index, study in enumerate(studies)}
        picked = []
        for row in rows:
            index = position[row.study]
            negative = truth[index] <= MIDPOINT
            if not negative.any():
                continue
            damages = np.where(negative, table[index], -1.0)
            column = int(np.argmax(damages))
            picked.append({"study": row.study, "target": findings[column], "damage": float(table[index, column])})
        return pd.DataFrame(picked, columns=["study", "target", "damage"])

    def labelled(self, cells: pd.DataFrame, reports: dict[str, str]) -> pd.DataFrame:
        """Attach each cell's report language, which every text detector needs before it can read."""
        languages = [self.languages.language(reports[str(study)]) for study in cells["study"]]
        return cells.assign(language=languages)

    def caused(self, tail: pd.DataFrame, body: pd.DataFrame, reports: dict[str, str]) -> LabelCause:
        """Classify both strata through the SAME auditor and census them side by side."""
        return LabelCause(
            tail=self.audit.census(self.audit.audit(self.labelled(tail, reports), reports)),
            body=self.audit.census(self.audit.audit(self.labelled(body, reports), reports)),
        )

    @staticmethod
    def acquired(tail: set[str], body: set[str], series: pd.DataFrame) -> AcquisitionCause:
        """Per-study acquisition means for both strata: series count, planes, fluid and fat suppression."""
        # NUMERIC, NOT TEXTUAL. Both flags are int64 0/1 in this corpus, and the first version of this
        # read them as `"TRUE"` strings — which matches nothing, so every study scored 0.0 on both and
        # the acquisition axis printed a perfect null in tail and body alike. A covariate that reads
        # identically everywhere is the shape a broken parse takes, never the shape of evidence.
        counted = series.assign(
            fluid=series["Fluid_Sensitive"].astype(float),
            fat=series["Fat_Suppression"].astype(float),
        )
        grouped = counted.groupby("StudyInstanceUID").agg(
            series_count=("SeriesInstanceUID", "count"),
            planes=("Anatomical_Plane", "nunique"),
            fluid=("fluid", "mean"),
            fat=("fat", "mean"),
        )
        columns = ("series_count", "planes", "fluid", "fat")
        return AcquisitionCause(
            columns=columns,
            tail={column: float(grouped.loc[list(tail & set(grouped.index)), column].mean()) for column in columns},
            body={column: float(grouped.loc[list(body & set(grouped.index)), column].mean()) for column in columns},
        )


@dataclass(frozen=True)
class AttributionCorpus:
    """Everything on disk this reading joins: the run, the other members, the reports, the series."""

    models: Path
    raw: Path

    def ranked(
        self, run: str
    ) -> tuple[
        list[StudyDamage],
        Float[NDArray[np.float64], "studies findings"],
        list[str],
        Float[NDArray[np.floating], "studies findings"],
    ]:
        """One run's ranking, its damage table, its study order and its truth — all four are needed."""
        reading = HoldoutRecord(self.models / f"{run}.pt").read()
        if reading is None:
            message = f"no holdout record beside {run}"
            raise SystemExit(message)
        errors = StudyErrors()
        table = errors.damages(reading.predictions, reading.truth)
        return errors.ranked(reading.studies, table, reading.truth), table, list(reading.studies), reading.truth

    def members(self, run: str, studies: list[str], wanted: int) -> dict[str, list[StudyDamage]]:
        """Other members ranked the same way, kept only where they cover the SAME studies in the same order.

        A member on a different split cannot agree or disagree about a tail — its tail is a different
        population — so it is skipped rather than aligned, which would be a join this reading invents.
        """
        errors = StudyErrors()
        found: dict[str, list[StudyDamage]] = {}
        for path in sorted(self.models.glob(f"*{SUFFIX}")):
            name = path.name.removesuffix(SUFFIX)
            if name == run or len(found) >= wanted:
                continue
            member = HoldoutRecord(self.models / f"{name}.pt").read()
            if member is None or list(member.studies) != studies:
                continue
            found[name] = errors.ranked(member.studies, errors.damages(member.predictions, member.truth), member.truth)
        return found

    def reports(self) -> dict[str, str]:
        """Study id to report text, which is the evidence every label category is read out of."""
        frame = pd.read_csv(self.raw / "train.csv")
        return dict(zip(frame["StudyInstanceUID"].astype(str), frame["Report"].astype(str), strict=True))

    def series(self) -> pd.DataFrame:
        """The per-series acquisition table, unaggregated — the strata decide how it is grouped."""
        return pd.read_csv(self.raw / "train_series.csv")


def main() -> None:
    """Attribute one run's worst tenth: shared or private, label or model, and acquisition."""
    command = cli.Command(__doc__, LOGGER)
    command.parser.add_argument("--run", type=str, required=True, help="checkpoint stem whose holdout is attributed")
    command.parser.add_argument("--members", type=int, default=MEMBERS, help="how many other members to compare tails")
    arguments = command.parsed()
    root = command.rooted(arguments)
    corpus = AttributionCorpus(models=root.models(), raw=root.raw())
    rows, table, studies, truth = corpus.ranked(arguments.run)
    attribution = StudyAttribution()
    tail_rows, body_rows = attribution.split(rows)
    LOGGER.info(f"{arguments.run}: {len(rows)} studies, tail {len(tail_rows)}, body {len(body_rows)}")

    others = corpus.members(arguments.run, studies, arguments.members)
    command.report(attribution.shared(rows, others).lines())

    reports = corpus.reports()
    tail_cells = attribution.negative_cells(tail_rows, studies, table, truth, TARGETS)
    body_cells = attribution.negative_cells(body_rows, studies, table, truth, TARGETS)
    command.report(attribution.caused(tail_cells, body_cells, reports).lines())

    command.report(
        attribution.acquired(
            {row.study for row in tail_rows}, {row.study for row in body_rows}, corpus.series()
        ).lines()
    )


if __name__ == "__main__":
    main()
