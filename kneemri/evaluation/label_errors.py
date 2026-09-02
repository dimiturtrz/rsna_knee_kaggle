"""Which (study, finding) cells does an INDEPENDENT signal say the reader got wrong?

`python -m kneemri.evaluation.label_errors`

EVERY LABEL-QUALITY MECHANISM THIS PROJECT SHIPS IS READER-SIDE. `Supervision.applied` maps the reader's
certainty level onto measured agreement, and `weight_floor` sets a cell's loss weight to
`0.25 + 0.75*conf` from the reader's own `__conf`. Both were measured and both are spent —
`finding_calibration` read flat, and `report_reader.CONFIDENCE` is a CONSTANT per verdict (0.95 on every
YES), so the floor is not reading a certainty at all.

THE DEFECT THEY CANNOT SEE. Over the 58 radiologist-annotated studies `reader_agreement.errors` counts
108 false positives against 32 misses, and all 108 are confident YES. A weight that rises with the
reader's confidence therefore UP-WEIGHTS the reader's dominant error by construction. No function of
`__conf`, `__verdict` or the graded level can separate those cells, because the reader is equally
confident on the ones it got right.

CONFIDENT LEARNING USES A SIGNAL THAT IS NOT THE READER'S. Northcutt, Jiang and Chuang (JAIR 70:1373,
2021) estimate the joint of given-versus-true label from a model's OUT-OF-FOLD predicted probabilities:
a cell whose held-out probability disagrees with the label it was given is a candidate error, and the
confident joint decides how many such cells to believe rather than leaving it to a threshold. The
arithmetic is `cleanlab`'s, not ours — the thresholding and its calibration are exactly where a
hand-rolled version turns a bug into a finding, and the library carries the authors' own tests.

WHY TWELVE INDEPENDENT BINARY PROBLEMS AND NOT `multilabel_classification`. The defect has a CELL as its
unit — 108 over-called (study, finding) pairs — and cleanlab's multilabel surface does not produce one:
`multilabel_classification.get_label_quality_scores` computes a per-class score internally and then
AGGREGATES the twelve away into one score per example, which would rank studies rather than the cells
this is looking for. Twelve binary `filter.find_label_issues` calls keep the unit, let each finding drop
the rows it has no verdict for, and match what the task already is — the competition metric is the
unweighted mean of twelve independent per-label AUCs, and the reader emits a verdict per finding.

WHAT COUNTS AS THE GIVEN LABEL. The verdict, collapsed the way the census collapses it: `YES` is a
positive and both `NO` and `UNK` are negatives. That is not a convenience — `reader_agreement.errors`
calls a `UNK` cell the radiologist graded positive a MISS, and `Targets.load` gives `UNK` the graded
value 0.28, which binarises to absent. A different collapse here would score the detector against a
census it does not share. Cells the table carries no verdict for at all are dropped.

THE PREDICTIONS ARE THEIRS, WHICH IS THE POINT. `PublishedRuler` stacks twenty published checkpoints —
four seeds over five folds whose `oof` id-sets partition all 4,407 studies — into one held-out
prediction per seed for every study. Nothing here was fitted on our labels by us, and no study is scored
by a model that trained on it. Using our own in-sample predictions would make the detector agree with
the labels it is auditing and prove nothing.

THIS BUILDS A DETECTOR AND TRAINS NOTHING. Whether re-weighting the flagged cells moves a score is a
separate, card-costing question, and the label lane has closed once already on a treatment that was
never gated first (`interpretations/kneemri/2026-08-30_label_lane_closes.md`). The gate below is what
that lane did not have: the flags are scored against 140 disagreements a radiologist independently
recorded, and against a random detector spending the same budget.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
from jaxtyping import Bool, Float
from numpy.typing import NDArray

from core import cli
from core.data.labels.reader_agreement import POSITIVE, SILENT, ReaderAgreement
from core.findings import STUDY, TARGETS
from kneemri.evaluation.published_ruler import PublishedRuler
from kneemri.evaluation.scores import LabelScores

logger = logging.getLogger(__name__)

DEFAULT_LABELS = Path("labels") / "report_labels_v2.csv"
"""The reader's table, under the processed root. The one every trained member was fitted against."""

VERDICT_SUFFIX = "__verdict"

BINARY_CLASSES = 2
"""A finding is present or absent. Named because cleanlab wants the column count stated, not implied."""

FALSE_POSITIVE = "false positive"
MISSED = "missed"
"""`reader_agreement.errors`' own `kind` values, so one census cannot be counted under two vocabularies."""

DISAGREEMENT = "disagreement"
"""Either direction. Not a `kind` there — no cell is one — which is why it is named here and not imported."""

BUDGETS: tuple[int, ...] = (25, 50, 100, 150, 200)
"""Cell budgets the ranking is read at, bracketing the 140 disagreements the gold 58 actually carry.

A CURVE RATHER THAN A CHOSEN OPERATING POINT. Picking the budget that flatters the detector is how a
threshold gets tuned on the only 58 studies that could ever have validated it.
"""


@dataclass(frozen=True)
class ReaderTable:
    """The reader's verdicts as a binary label matrix, with its abstentions marked unread.

    BESIDE ITS CONSUMER because the one decision it encodes belongs to this measurement: the reader's
    three verdicts are collapsed to the two the census scores them under, `YES` against everything else.
    A detector fitted on a different collapse would be scored against a census it does not share.
    """

    identifiers: list[str]
    given: Bool[np.ndarray, "studies targets"]
    """`YES` — the label the model was fitted against, and the one the census scores."""

    read: Bool[np.ndarray, "studies targets"]
    """Whether the table states a verdict at all. A study it does not carry is unread, never negative."""

    @classmethod
    def of(cls, table: pd.DataFrame, identifiers: list[str]) -> "ReaderTable":
        """The verdict matrix for exactly these studies, in their order.

        A study the table does not carry, or a finding it holds no verdict column for, lands UNREAD.
        `UNK` is a stated verdict and so a read negative — see the module docstring on the collapse.
        """
        rows = table.set_index(STUDY).reindex(identifiers)
        given = np.zeros((len(identifiers), len(TARGETS)), dtype=bool)
        read = np.zeros_like(given)
        for column, target in enumerate(TARGETS):
            name = f"{target}{VERDICT_SUFFIX}"
            if name not in rows.columns:
                continue
            verdict = rows[name].astype("string").str.upper()
            read[:, column] = verdict.notna().to_numpy(dtype=bool)
            given[:, column] = (verdict.fillna(SILENT) == POSITIVE).to_numpy(dtype=bool)
        return cls(identifiers, given, read)


@dataclass(frozen=True)
class LabelErrorDetector:
    """cleanlab's confident joint over held-out probabilities, run once per finding.

    Holds the two aligned matrices rather than recomputing them per question: the quality ranking and the
    issue mask are two readings of one estimate, and a caller that built them from separate arrays is how
    a study's score gets compared against another study's label.
    """

    predicted: Float[np.ndarray, "studies targets"]
    """The held-out probability of the finding, averaged over the seeds that did not train on the study."""

    reader: ReaderTable

    def quality(self) -> Float[np.ndarray, "studies targets"]:
        """Per-cell self-confidence: the held-out probability of the label the reader actually gave.

        LOW IS SUSPECT. A cell the reader called YES and the models put at 0.03 scores 0.03; the same
        cell called NO scores 0.97. Unread cells are NaN, never 0 — a zero would rank every abstention
        above every real disagreement.
        """
        scores = np.full(self.predicted.shape, np.nan, dtype=np.float64)
        for column in range(len(TARGETS)):
            rows = np.flatnonzero(self.reader.read[:, column])
            if rows.size == 0:
                continue
            labels = self.reader.given[rows, column].astype(np.int64)
            scores[rows, column] = get_label_quality_scores(labels, self._probabilities(rows, column))
        return scores

    def flagged(self) -> Bool[np.ndarray, "studies targets"]:
        """cleanlab's own issue mask — which cells, and HOW MANY, decided by the confident joint.

        The count is the part a ranking cannot supply. `quality` orders every cell; the confident joint
        estimates the noise rate itself, so the size of the flagged set is an estimate rather than a knob.
        """
        issues = np.zeros(self.predicted.shape, dtype=bool)
        for column in range(len(TARGETS)):
            rows = np.flatnonzero(self.reader.read[:, column])
            labels = self.reader.given[rows, column].astype(np.int64)
            if rows.size == 0 or len(np.unique(labels)) < BINARY_CLASSES:
                continue
            issues[rows, column] = find_label_issues(labels, self._probabilities(rows, column))
        return issues

    def _probabilities(self, rows: NDArray[np.int64], column: int) -> Float[np.ndarray, "rows classes"]:
        """One finding's held-out probabilities as the two-column distribution cleanlab reads."""
        positive = self.predicted[rows, column]
        return np.column_stack([1.0 - positive, positive])


@dataclass(frozen=True)
class GateRow:
    """What a detector spending one budget actually caught, beside what spending it blindly would.

    BESIDE ITS PRODUCER rather than returned as a tuple, because the verdict is a RATIO between two of
    these numbers and a caller reconstructing it from positions is how a comparison gets read backwards.
    """

    budget: int
    caught: int
    errors: int
    scoreable: int

    def precision(self) -> float:
        """The share of the flagged cells that really are disagreements."""
        return self.caught / self.budget if self.budget else float("nan")

    def recall(self) -> float:
        """The share of the known disagreements this budget reached."""
        return self.caught / self.errors if self.errors else float("nan")

    def baseline(self) -> float:
        """What a random detector spending the SAME budget catches, per flagged cell.

        Prevalence, exactly: drawing `budget` of `scoreable` cells without replacement catches
        `budget * errors / scoreable` in expectation, so its precision is the error rate whatever the
        budget. That is the number a lift must be taken against, not 0.5 and not the flagged count.
        """
        return self.errors / self.scoreable if self.scoreable else float("nan")

    def lift(self) -> float:
        """Precision over the prevalence-matched random baseline. 1.0 is chance."""
        base = self.baseline()
        return self.precision() / base if base else float("nan")

    def line(self) -> str:
        """One row of the gate table, denominators included."""
        return (
            f"    budget {self.budget:4d}   caught {self.caught:4d}/{self.errors:<4d} "
            f"precision {self.precision():.3f}   random {self.baseline():.3f}   "
            f"lift {self.lift():.2f}x   recall {self.recall():.3f}"
        )


@dataclass(frozen=True)
class DetectorGate:
    """Do the lowest-quality cells land on disagreements a radiologist independently recorded?

    Flat over the gold cells rather than per finding: 140 errors over twelve findings is as few as three
    per finding, and a per-finding lift on three cells is a number with no resolution at all.
    """

    quality: Float[np.ndarray, "cells"]
    error: Bool[np.ndarray, "cells"]

    def ranked(self, budgets: tuple[int, ...] = BUDGETS) -> list[GateRow]:
        """The catch at each budget, taking the WORST-scoring cells first."""
        order = np.argsort(self.quality, kind="mergesort")
        errors, scoreable = int(self.error.sum()), int(self.error.size)
        caught = np.cumsum(self.error[order].astype(np.int64))
        return [
            GateRow(budget, int(caught[budget - 1]), errors, scoreable) for budget in budgets if 0 < budget <= scoreable
        ]

    def separation(self) -> float:
        """AUC of the quality ranking as an error detector — threshold-free, so nothing is tuned.

        0.5 is chance. Reported beside the budget curve because a curve can be read optimistically at
        whichever budget flatters it and this cannot.
        """
        return LabelScores.area_under_curve(-self.quality, self.error.astype(np.int64))

    def at_mask(self, flagged: Bool[np.ndarray, "cells"]) -> GateRow:
        """The catch of cleanlab's OWN flagged set, whose size it chose rather than we did."""
        budget = int(flagged.sum())
        return GateRow(budget, int((flagged & self.error).sum()), int(self.error.sum()), int(self.error.size))


@dataclass(frozen=True)
class GoldCensus:
    """The radiologist's disagreements with the reader, as flat masks over the gold cells.

    Built from `reader_agreement.cells`/`errors` rather than recomputed: those are where the 108-against-32
    census this project quotes is defined, and a second definition of which cells are wrong would let the
    detector be scored against a truth nobody else uses.
    """

    study: NDArray[np.str_]
    target: NDArray[np.str_]
    false_positive: Bool[np.ndarray, "cells"]
    missed: Bool[np.ndarray, "cells"]

    @classmethod
    def of(cls, cells: pd.DataFrame, errors: pd.DataFrame) -> "GoldCensus":
        """Mark each scored cell as an over-call, a miss, or an agreement."""
        keys = list(zip(cells["study"], cells["target"], strict=True))
        position = {key: index for index, key in enumerate(keys)}
        false_positive = np.zeros(len(keys), dtype=bool)
        missed = np.zeros(len(keys), dtype=bool)
        for _, row in errors.iterrows():
            index = position[(row["study"], row["target"])]
            if row["verdict"] == POSITIVE:
                false_positive[index] = True
            else:
                missed[index] = True
        return cls(np.array([key[0] for key in keys]), np.array([key[1] for key in keys]), false_positive, missed)

    def disagreements(self) -> Bool[np.ndarray, "cells"]:
        """Either direction — every cell the reader and the radiologist do not share."""
        return self.false_positive | self.missed

    def counts(self) -> dict[str, int]:
        """How many cells of each kind, under `reader_agreement.errors`' own vocabulary for them.

        The census counts itself rather than handing its masks out to be summed: a caller reaching through
        for `.false_positive.sum()` is one that can sum three arrays of different lengths and say nothing.
        """
        return {
            FALSE_POSITIVE: int(self.false_positive.sum()),
            MISSED: int(self.missed.sum()),
            DISAGREEMENT: int(self.disagreements().sum()),
        }

    def restricted(self, keep: Bool[np.ndarray, "cells"]) -> "GoldCensus":
        """The same census over a subset of its cells — all four arrays cut together.

        One method rather than four subscripts at the call site: the arrays are positionally aligned and
        cutting three of them is a census whose masks describe different cells.
        """
        return GoldCensus(self.study[keep], self.target[keep], self.false_positive[keep], self.missed[keep])


@dataclass(frozen=True)
class LabelErrorReport:
    """The coverage, the census and the gate as lines — the rendering half of the measurement.

    Split from the detector for the reason the other reports here are: producing the quality matrix reads
    twenty checkpoints off disk, and rendering needs two 1-D arrays.
    """

    corpus_studies: int
    corpus_flagged: int
    corpus_read: int
    annotated_studies: int
    """Studies `train.csv` grades — the outermost denominator, before any join drops anything."""

    gold_studies: int
    """Of those, the ones the reader table also carries a verdict for. The census is defined here."""

    covered_studies: int
    """Of those, the ones the out-of-fold ruler predicts. Only these reach the gate."""
    census: GoldCensus
    quality: Float[np.ndarray, "cells"]
    flagged: Bool[np.ndarray, "cells"]

    def lines(self) -> list[str]:
        """Coverage first, then the census, then one gate per error class."""
        counts = self.census.counts()
        lines = [
            f"corpus: {self.corpus_studies} studies with held-out predictions, "
            f"{self.corpus_read} read cells of {self.corpus_studies * len(TARGETS)}, "
            f"{self.corpus_flagged} flagged ({self.corpus_flagged / max(self.corpus_read, 1):.3%} of read)",
            f"gold: {self.annotated_studies} radiologist-annotated studies, of which {self.gold_studies} "
            f"also carry a reader verdict, of which {self.covered_studies} carry held-out predictions; "
            f"{len(self.quality)} scoreable cells",
            f"  census: {counts[FALSE_POSITIVE]} false positives, {counts[MISSED]} misses, "
            f"{counts[DISAGREEMENT]} disagreements of {len(self.quality)} cells",
        ]
        for name, truth in self._classes():
            lines += self._gate_lines(name, truth)
        return lines

    def _classes(self) -> list[tuple[str, Bool[np.ndarray, "cells"]]]:
        """The three populations the detector is asked to separate, each against ALL scoreable cells.

        The two directions are kept apart because they are different defects and the whole premise is
        that the confident false positives are the findable ones. A detector that reads well on the
        merged census and no better than chance on the 108 has not found what it was built for.
        """
        return [
            ("all disagreements", self.census.disagreements()),
            ("confident false positives", self.census.false_positive),
            ("misses", self.census.missed),
        ]

    def _gate_lines(self, name: str, truth: Bool[np.ndarray, "cells"]) -> list[str]:
        """One class's separation, its budget curve, and cleanlab's own operating point."""
        gate = DetectorGate(self.quality, truth)
        lines = [f"  {name} (n={int(truth.sum())}): separation AUC {gate.separation():.4f} (0.5 = chance)"]
        lines += [row.line() for row in gate.ranked()]
        lines.append(f"    cleanlab's own mask ->{gate.at_mask(self.flagged).line().split('caught', 1)[1]}")
        return lines


@dataclass(frozen=True)
class DetectedCells:
    """One run of the detector over the corpus, before anything is asked of it.

    The gate reads these four values on the annotated 58 and the training table reads them on all 4,407,
    and they are the SAME estimate — `PublishedRuler.per_seed` is minutes of I/O, and a second call would
    also let the flags an arm trains against drift from the flags the gate passed.
    """

    identifiers: list[str]
    reader: ReaderTable
    quality: Float[np.ndarray, "studies targets"]
    flagged: Bool[np.ndarray, "studies targets"]

    def frame(self) -> pd.DataFrame:
        """The flagged cells as the two-column table `core.data.labels.targets.Supervision.pruned` reads."""
        rows, columns = np.nonzero(self.flagged)
        return pd.DataFrame(
            {
                "StudyInstanceUID": [self.identifiers[row] for row in rows],
                "target": [TARGETS[column] for column in columns],
            }
        )


@dataclass(frozen=True)
class LabelErrorAudit:
    """The whole measurement: a corpus-wide detector, restricted to the cells a radiologist scored.

    THE DETECTOR IS FITTED ON THE CORPUS AND READ ON THE 58. Estimating the confident joint from the 58
    annotated studies alone would fit the noise rate on the same cells the gate then scores, and 684
    cells is far too few for the estimate anyway. The 4,407-study fit never sees the annotations.
    """

    table: pd.DataFrame
    """The reader's table under audit."""

    annotated: pd.DataFrame
    """The radiologist's rows, in the columns `ReaderAgreement.cells` joins on."""

    published: Path
    """Where the twenty published checkpoints carrying the `oof` blocks were unzipped."""

    def detected(self) -> DetectedCells:
        """Run the detector over every study the ruler covers."""
        identifiers, per_seed = PublishedRuler(self.published).per_seed()
        reader = ReaderTable.of(self.table, identifiers)
        detector = LabelErrorDetector(np.mean(np.stack(per_seed), axis=0), reader)
        return DetectedCells(identifiers, reader, detector.quality(), detector.flagged())

    def report(self) -> LabelErrorReport:
        """Run the detector over every study the ruler covers, then gate it on the annotated cells."""
        found = self.detected()
        return self.gated(found.identifiers, found.reader, found.quality, found.flagged)

    def gated(
        self,
        identifiers: list[str],
        reader: ReaderTable,
        quality: Float[np.ndarray, "studies targets"],
        flagged: Bool[np.ndarray, "studies targets"],
    ) -> LabelErrorReport:
        """Pair the detector's per-cell output with the census, over the cells present in BOTH.

        THE DENOMINATOR IS CARRIED, not assumed. A gold cell whose study the ruler does not cover is
        dropped from the gate and counted in the coverage line, because a hit rate computed inside the
        subset that happened to align is not a hit rate over the annotated set.
        """
        agreement = ReaderAgreement()
        cells = agreement.cells(self.annotated, self.table)
        census = GoldCensus.of(cells, agreement.errors(cells))
        row = {name: index for index, name in enumerate(identifiers)}
        column = {name: index for index, name in enumerate(TARGETS)}
        inside = np.array([study in row for study in census.study], dtype=bool)
        rows = np.array([row[study] for study in census.study[inside]], dtype=np.int64)
        columns = np.array([column[target] for target in census.target[inside]], dtype=np.int64)
        return LabelErrorReport(
            corpus_studies=len(identifiers),
            corpus_flagged=int(flagged.sum()),
            corpus_read=int(reader.read.sum()),
            annotated_studies=len(self.annotated),
            gold_studies=int(cells["study"].nunique()),
            covered_studies=int(np.unique(census.study[inside]).size),
            census=census.restricted(inside),
            quality=quality[rows, columns],
            flagged=flagged[rows, columns],
        )


def main() -> None:
    """Score the reader's cells by confident learning, and gate the flags on the annotated 58."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, default=None, help="the reader table to audit")
    command.parser.add_argument("--flags", type=Path, default=None, help="write the flagged cells here")
    arguments = command.parsed()

    root = command.rooted(arguments)
    labels = arguments.labels or root.processed() / DEFAULT_LABELS
    logger.info(f"auditing {labels} against the published out-of-fold ruler")
    audit = LabelErrorAudit(pd.read_csv(labels), ReaderAgreement.annotated(root), root.processed() / "published")
    found = audit.detected()
    command.report(audit.gated(found.identifiers, found.reader, found.quality, found.flagged).lines())
    if arguments.flags is not None:
        frame = found.frame()
        frame.to_csv(arguments.flags, index=False)
        logger.info(f"wrote {len(frame)} flagged cells to {arguments.flags}")


if __name__ == "__main__":
    main()
