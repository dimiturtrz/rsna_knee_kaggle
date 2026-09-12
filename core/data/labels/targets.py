"""The training targets: twelve graded scores per study, each with the confidence it was read at.

Only 58 of 4,407 studies carry the twelve annotations — 1.3%. Every study carries a radiology report,
and `test.csv` has no `Report` column, so text is available when fitting and absent when predicting.
Reports are therefore usable as targets, never as an inference-time input.

Two properties of the labels shape the loss:

**Graded, not thresholded.** The reporting radiologist and the annotator do not share a threshold — a
report saying *small joint effusion* can sit against a negative annotation because the annotator marked
only effusions they judged significant. "Term present implies positive" is wrong by construction; the
grade costs nothing, because the metric reads only order.

**Confidence is a weight, not a value.** A report that never mentions synovitis should pull weakly on
that head rather than assert a negative there, so the confidence multiplies the loss for that
(study, target) pair rather than moving the target toward 0.5.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from core.data.corpus.merged_cache import CORPUS
from core.findings import TARGETS

CONFIDENCE_FLOOR = 0.25
"""What a finding the report never mentions still pulls with, as a fraction of a fully-confident one.

Ours dropped such targets entirely (weight 0.0); the public baseline uses `0.25 + 0.75*conf`. Silence
in a radiology report is weak evidence of ABSENCE — a knee report lists what was checked and found
intact — so dropping it discards evidence rather than declining to guess. Kept as a parameter because
the two behaviours are a measurable difference, not a matter of taste."""

MEASURED_AGREEMENT: tuple[tuple[float, float], ...] = (
    (0.08, 0.041),
    (0.28, 0.136),
    (0.68, 0.378),
    (0.82, 0.688),
    (0.94, 0.958),
)
"""Each level the reader emits, beside the rate the radiologist agreed at. See `Calibration`."""

FINDING_OFFSETS: tuple[tuple[str, float], ...] = (
    ("ACL", -0.622),
    ("MCL", -0.474),
    ("Medial Meniscus", 0.166),
    ("Lateral Meniscus", 0.054),
    ("Medial OA", -0.279),
    ("Lateral OA", -0.460),
    ("PF OA", 0.207),
    ("Effusion", 0.432),
    ("Synovitis", 0.985),
    ("Baker's", -0.177),
    ("Contusion", -0.531),
    ("Fracture", 0.056),
)
"""A logit shift per finding ON TOP of `MEASURED_AGREEMENT`, because that table is pooled over all twelve.

Fitted by `core.data.labels.finding_calibration` over the same 684 annotated cells the pooled table was
fitted on — one maximum-likelihood shift per finding, drawn toward zero by an empirical-Bayes weight so
a rate on 57 cells cannot manufacture a spread out of sampling noise. The between-finding variance
survived that subtraction at 0.294 (sd 0.543 in logits), so the findings do NOT share one scale: a 0.68
cell trains at 0.246 on ACL and 0.619 on Synovitis against the pooled 0.378.

Read the two failure directions in the signs. Negative is a finding the reader OVER-calls — Contusion,
where the radiologist agrees with only 44.4% of positive calls. Positive is one the reader MISSES —
Synovitis, silent on 41 cells of which 14 are truly positive, and the only finding whose net error runs
under rather than over. One parameter carries both because it is fitted on every cell, not on the
positive calls alone.

WHAT THIS CANNOT DO, stated where the constants live. A monotone map of one head's targets leaves that
head's ranking unchanged, so in the well-specified limit per-head AUC does not move. Whatever this buys
comes through the SHARED TRUNK — twelve heads competing for one representation, with the mis-scaled
ones pulling too hard — which is the same indirect route the pooled calibration must already work
through. Price it as second-order to a correction that was itself worth +0.0061."""

ANNOTATED_WEIGHT = 3.0
"""Image-read annotations outweigh any derived row: they are the only labels read from the pixels rather
than from text, and there are 58 of them."""

ASSERTS_PRESENCE = 0.5
"""Which side of this a level sits on decides what agreeing with it MEANS. A 0.94 cell is right when the
radiologist finds the thing; a 0.08 cell is right when they do not."""

UNCERTAIN_LEVEL = 0.28
"""The one level that asserts nothing, so it has no reliability to compute — see `Supervision.trusted`."""

NO_SHIFT = 0.0
LOGIT_FLOOR = 1e-6
"""Keeps the logit finite on a table that may hold an exact 0 or 1. See `Supervision.per_finding`."""

WEIGHT_FLOOR = "weight_floor"
CALIBRATED = "calibrated_targets"
FINDING = "finding_calibration"
SEVERITY_WEIGHT = "severity_weight"
ANNOTATIONS = "annotations"
SUSPECT = "label_errors"
"""The six label switches, named where the object that answers for them lives. See `Targets.supervised`."""

UNREAD = "unread_targets"
"""The seventh switch. Named `unread` and not `teacher` because `landed` already spends that word on
distillation, and because it says what the cells have in common rather than where their value comes from."""

WITHHELD = "hold_annotation_studies"
"""The eighth switch, and the one that exists because the seventh was read as doing its job.

`annotations` withholds the annotator's VALUE. This withholds the STUDY. They were conflated in prose
for weeks — see `Supervision.withheld` for the measurement that separates them."""

SUSPECT_STUDY = "StudyInstanceUID"
SUSPECT_TARGET = "target"
"""The two columns of the flag table `kneemri.evaluation.label_errors --flags` writes."""

UNREAD_VALUE = "value"
"""The third column of the unread table, which carries a score where the flag table carries only a name."""


class UncalibratedFindingShiftError(Exception):
    """A per-finding offset asked for without the pooled calibration it was fitted as a correction to."""


@dataclass(frozen=True)
class Supervision:
    """What each report-derived level is WORTH, measured against the image annotations.

    The reader emits five values and we trained on them as if they were probabilities. Measured over
    the 57 fully-annotated studies in train.csv (696 cells, every label annotated), how often the
    radiologist actually agrees the finding is present:

        assigned   measured   95% CI            n
          0.08      0.041     [0.013, 0.068]   196
          0.28      0.136     [0.085, 0.186]   177
          0.68      0.378     [0.282, 0.474]    98
          0.82      0.688     [0.611, 0.764]   141
          0.94      0.958     [0.912, 1.004]    72

    FOUR OF FIVE fall outside their own interval, all four too high: the model was told 0.68 for cells
    that are positive 38% of the time. The levels themselves are sound — agreement is monotone across a
    span of 0.58, and the reader's `__conf` column is constant per verdict, so this gradation is real
    per-study information rather than certainty noise. It is the VALUES that were wrong.

    IT DOES NOT LEAK INTO THE NUMBER IT WILL BE JUDGED BY. The calibration is fitted on the 58
    image-read annotations; an arm is scored on the fold-0 holdout of the REPORT labels. Different
    rulers, so the comparison against wsfold0_e6 stays honest. What it does spend is some independence
    of the annotation-based check (`annot`), which is stated rather than hidden.
    """

    floor: float = 0.0
    """What a finding the report never mentions still pulls with. See CONFIDENCE_FLOOR."""

    calibrated: bool = False
    """Whether to replace each level with the rate it was measured at."""

    severity: bool = False
    """Whether a cell's WEIGHT comes from the grade the reader assigned rather than its `__conf`."""

    annotations: bool = True
    """Whether the 58 image-read annotations are read into the TRAINING table at all.

    IT WITHHOLDS THE ANNOTATOR'S VALUE, NOT THE STUDY, and the difference was misread for weeks. Off
    skips exactly one block in `load` — the one that overwrites those cells at `ANNOTATED_WEIGHT`. The
    ROWS stay in the training table at their report-derived values and keep pulling on the loss.
    Measured 2026-09-12: of the 58 annotated studies, **45 sit in the fold-0 TRAINING split and 13 in
    the 909 holdout**, so an annotated read taken off a member fitted with this flag is 78% in-sample.
    Every header claiming this flag makes the gold reading "clean" or "held out" was wrong, including
    `recal_holdannot.running.sh`. `withheld` is the switch that does what those headers said. See
    knee-x4muo.

    Off makes a member whose predictions on those studies never saw their radiologist labels, which is
    the one input `kneemri.evaluation.supervision_gap` needs and cannot get from any member we fit: the
    override weights them at `ANNOTATED_WEIGHT`, so every member is scored against a ruler it trained
    on. Measured on `report_labels_v2.csv`: 684 cells at weight 3.0 carry 2052.0 of 36579.2 total, so
    gold is **5.61%** of training weight — the cheapest thing in the project to give up, and it buys
    back the only radiologist-truth ruler there is. (This said "696 cells ... ~0.66%" until 2026-09-07;
    both halves were wrong, and the 0.66% was quoted into a card plan as an 8x share deficit against
    mattiaangeli's 8.1%. The real deficit is 1.44x. Re-measure before quoting it again.)
    """

    finding: bool = False
    """Whether each finding also carries its own shift on top of the pooled level map. Requires `calibrated`."""

    suspect: Path | None = None
    """A flag table naming the cells an INDEPENDENT signal says the reader got wrong; they pull with nothing.

    Every other switch on this object is a function of what the reader says about ITSELF — its level, its
    grade, its `__conf`. The defect they cannot reach is the cell where the reader was confident and
    wrong: over the 58 annotated studies `reader_agreement.errors` counts 108 confident false positives
    against 32 misses, and `0.25 + 0.75*conf` weights all 108 near 1.0 by construction.

    The flags come from confident learning over out-of-fold predictions the reader never saw
    (`kneemri.evaluation.label_errors`), which beats a random flagger of the same size 4.05x on those 108.
    PRUNING rather than down-weighting is cleanlab's own operating point: the confident joint estimates
    how many cells to disbelieve, so the size of the flagged set is an estimate and not a threshold this
    project would otherwise have to fit on its own holdout.

    An annotated cell is never pruned. The radiologist's reading is the truth the detector was gated
    against; dropping it because a model disagrees would delete the evidence.
    """

    unread: Path | None = None
    """A (study, target, value) table replacing the SCORE on the cells it names, weight untouched.

    IT IS THE VALUE-OVERRIDE CHANNEL, and it has two producers rather than one. `unread_teacher` writes
    the cells the report never read; `suspect_teacher` writes a CORRECTED level for the cells an
    independent signal says the reader got wrong — the same cells `suspect` prunes, kept in the loss and
    pointed somewhere else instead. Both are (study, target, value) tables read the same way, so this
    stays one field; which defect an arm is treating is stated by the arm, and the two are never mixed
    in one run because that would be two variables.

    The paragraphs below describe the FIRST producer, which is the one that named the field.

    The complement of `suspect`, and the reason both exist on one object. That switch reaches the cell
    where the reader was confident and wrong; this one reaches the cell where the reader said NOTHING.
    27.3% of this project's supervision — 14,445 of 52,872 pairs — carries the constant 0.28 at verdict
    UNK, concentrated in Synovitis (84.2%), Fracture (56.4%) and Baker's (45.9%).

    THE CLAIM THAT MAKES IT WORTH A SWITCH. Those cells train as weak absence, and `report_reader`
    measured that the report does not name the structure at all on them (Synovitis 1.8% of 1,424) — so
    the weak absence is inferred from a silence about something never examined. `supervision_gap` reads
    the model at 0.794 on Synovitis against the label's 0.687 over the 58 annotated studies, which is a
    model exceeding a ceiling made of constants rather than out-reading a mis-read label.

    THE WEIGHT IS DELIBERATELY NOT TOUCHED. An unread cell keeps the 0.2875 it pulls with today, so this
    switch changes what the cell SAYS and never how loudly — otherwise it would be two variables, and
    `severity` already owns the other one.

    WHAT IT CANNOT BE READ AS. The values come from a member fitted on these same cells at 0.28, so its
    read of them is partly the prior it was pulled toward. That is bounded rather than absent — the
    member reads pixels — and it is why the gate is on the pairs the lexicon left BLANK, against
    radiologist truth the teacher never saw, and never on an average over all 52,872 pairs where the two
    tables agree by construction. See knee-s6s9r.
    """

    withheld: bool = False
    """Whether the annotated STUDIES contribute nothing to the loss at all.

    THE SWITCH `annotations` WAS READ AS BEING. Turning `annotations` off leaves all 58 rows training at
    their report-derived values, and 45 of them sit in the fold-0 training split, so every gold reading
    this project has taken was measured on a ruler that is 78% in-sample. That is not a small correction:
    `knee-8wa0`'s +0.0105 gold move, and every arm header calling its gold read "clean", rest on it.

    On, every cell of an annotated study is zeroed in the WEIGHT array — the row stays in the table and
    in the study order, contributing exactly 0.0 to the loss. Zeroing rather than dropping, for the
    reason `pruned` gives: the arrays are aligned to a caller-supplied study order, and removing rows
    would make this switch change the batch composition and the epoch length as well as the supervision,
    which would be three variables wearing one flag.

    IT IS A COSTLY RULER TO BUY, and the price is stated rather than hidden: 58 of 3,498 training
    studies is 1.66% of the corpus, on top of the 5.61% of training WEIGHT that `annotations` already
    gives up. What it buys back is the only radiologist-truth ruler that is honestly out of sample — and
    at n=13 usable studies today, a ruler this thin may still be unreadable. Price its noise floor before
    reading a number off it.

    IT DOES NOT REPLACE `annotations`; it implies it. A run with `withheld` on and `annotations` on would
    write the gold values into cells whose weight is then zeroed, which is a no-op wearing a
    contradiction, so `load` refuses the pair rather than silently picking one.
    """

    measured: tuple[tuple[float, float], ...] = MEASURED_AGREEMENT

    offsets: tuple[tuple[str, float], ...] = FINDING_OFFSETS

    def moves_the_ruler(self) -> bool:
        """Whether this supervision would also change the table an arm is SCORED against.

        All three switches do, by different routes — a calibration moves the VALUES a holdout AUC ranks
        against, a severity weight moves which cells the holdout counts and how much, and withholding
        the annotations moves 696 cells back to what the report said about them. Either way the
        ruler has to be re-read plain or the arm earns credit for the labels agreeing with themselves.
        The floor alone does not: it is applied identically to every arm and the control.

        A prune does, and by the most direct route of the four: it sets some cells' weight to zero, and
        a holdout AUC over a table missing those cells is not the table the control was read on.

        An unread override does too, and it is the one that would be easiest to miss: it moves 27.3% of
        the table's VALUES, so a ruler read from the same file would rank against scores a member was
        just fitted toward. Pin `--eval-labels`.
        """
        return (
            self.calibrated
            or self.finding
            or self.severity
            or not self.annotations
            or self.withheld
            or self.suspect is not None
            or self.unread is not None
        )

    def pruned(self, identifiers: list[str]) -> NDArray[np.bool_]:
        """The flagged cells as a mask over this table's study order.

        A flag naming a study this table does not carry is DROPPED rather than raised: the detector runs
        over the whole corpus and an arm may be fitted on any subset of it, so a smoke run on sixty
        studies would otherwise die on the 4,347 flags it has no row for.
        """
        mask = np.zeros((len(identifiers), len(TARGETS)), dtype=np.bool_)
        if self.suspect is None:
            return mask
        row = {identifier: index for index, identifier in enumerate(identifiers)}
        column = {target: index for index, target in enumerate(TARGETS)}
        flags = pd.read_csv(self.suspect)
        for study, target in zip(flags[SUSPECT_STUDY], flags[SUSPECT_TARGET], strict=True):
            if study in row and target in column:
                mask[row[study], column[target]] = True
        return mask

    def taught(self, identifiers: list[str]) -> NDArray[np.float32]:
        """The override values as a table over this table's study order, NaN where the cell is untouched.

        NaN and not a sentinel, because `load` already spells "this cell was not said" that way and a
        second spelling of absence is the drift `Supervision` exists to keep out of the label path.

        A row naming a study this table does not carry is DROPPED rather than raised, for the reason
        `pruned` gives: the teacher runs over the whole corpus and an arm may be fitted on any subset,
        so a sixty-study smoke would otherwise die on the thousands of rows it has no place for.
        """
        taught = np.full((len(identifiers), len(TARGETS)), np.nan, dtype=np.float32)
        if self.unread is None:
            return taught
        row = {identifier: index for index, identifier in enumerate(identifiers)}
        column = {target: index for index, target in enumerate(TARGETS)}
        table = pd.read_csv(self.unread)
        for study, target, value in zip(table[SUSPECT_STUDY], table[SUSPECT_TARGET], table[UNREAD_VALUE], strict=True):
            if study in row and target in column:
                taught[row[study], column[target]] = value
        return taught

    def __post_init__(self) -> None:
        """A per-finding shift without the pooled map under it is a correction applied to the wrong table."""
        if self.finding and not self.calibrated:
            message = "finding calibration corrects the POOLED rates, so it requires calibrated targets"
            raise UncalibratedFindingShiftError(message)

    def reliability(self) -> tuple[tuple[float, float], ...]:
        """Each asserting level beside how often that assertion held, which is not the same as its rate.

        A 0.68 cell claims the finding is THERE and was right 37.8% of the time; a 0.08 cell claims it is
        ABSENT and was right 95.9% of the time — the same table read from the other side. Derived from
        `measured` rather than typed out again so the two cannot drift apart.

        0.28 is dropped rather than mapped: it asserts neither side, so there is nothing for a
        radiologist to have agreed WITH, and its 0.136 is a base rate rather than a hit rate. Those cells
        keep the near-zero `__conf` the reader already gives them.
        """
        return tuple(
            (level, rate if level > ASSERTS_PRESENCE else 1.0 - rate)
            for level, rate in self.measured
            if not np.isclose(level, UNCERTAIN_LEVEL)
        )

    def trusted(self, value: NDArray[np.float32], confidence: NDArray[np.float64]) -> NDArray[np.float64]:
        """How hard each cell pulls, BEFORE the floor is applied.

        WHY THE READER'S OWN COLUMN CANNOT ANSWER THIS: `__conf` is constant per verdict — 0.95 on all
        52,872 populated cells at every one of the three YES grades — so mild, moderate and severe train
        at identical weight today. The grade is per-study information the loss currently discards, and
        the annotations say it separates: 0.378 / 0.688 / 0.958, three intervals that do not overlap.

        DISTINCT FROM `applied`, which is the same table spent on the VALUE and lost 0.0020 as
        `calib_span`. That moved 6,461 mild cells across 0.5 and inverted their sign; this leaves every
        target where it is and only decides how loudly it speaks.
        """
        if not self.severity:
            return confidence
        trusted = confidence.copy()
        for level, rate in self.reliability():
            trusted[np.isclose(value, level)] = rate
        return trusted

    def applied(self, value: NDArray[np.float32]) -> NDArray[np.float32]:
        """Each level replaced by what it was measured to be worth; anything else left alone.

        Values the reader never emits pass through untouched rather than being snapped to a nearest
        level — a table with a different scale would otherwise be silently reinterpreted as this one.
        """
        if not self.calibrated:
            return value
        applied = value.copy()
        for level, rate in self.measured:
            applied[np.isclose(value, level)] = rate
        return applied

    def per_finding(self, value: NDArray[np.float32], target: str) -> NDArray[np.float32]:
        """This finding's own shift applied in logit space, after the pooled level map and never instead.

        Composes ONLY after `applied`, and the flag that turns it on requires the one that turns that
        on: the offsets were fitted as corrections TO the pooled rates, so shifting the reader's raw
        levels by them would be applying a correction to a table it was not measured against.

        A finding absent from the table shifts by nothing rather than raising, which is what keeps a
        twelve-name contract from becoming a thirteenth place to break when a finding is renamed —
        `FINDING_OFFSETS` is checked against `TARGETS` in its own unit test instead.
        """
        if not self.finding:
            return value
        shift = dict(self.offsets).get(target, NO_SHIFT)
        if shift == NO_SHIFT:
            return value
        bounded = np.clip(value.astype(np.float64), LOGIT_FLOOR, 1.0 - LOGIT_FLOOR)
        moved = 1.0 / (1.0 + np.exp(-(np.log(bounded / (1.0 - bounded)) + shift)))
        return np.where(np.isnan(value), value, moved.astype(np.float32))


class Targets:
    """Per-study scores and weights, from the report-derived table and the few image annotations."""

    def __init__(
        self,
        scores: NDArray[np.float32],
        weights: NDArray[np.float32],
        identifiers: list[str],
        supervision: "Supervision | None" = None,
        corpus: NDArray[np.str_] | None = None,
    ) -> None:
        self.scores = scores
        self.weights = weights
        self.identifiers = identifiers
        self.supervision = supervision or Supervision()
        self.position = {identifier: index for index, identifier in enumerate(identifiers)}
        self.corpus = corpus

    def supervised(self) -> dict[str, object]:
        """The label switches this table was actually built under, for the pre-flight landed check.

        WHAT THIS CATCHES is the wiring, which is where the whole class of defect lives: `pair` and
        `load` both take `supervision` with a `or Supervision()` default, so a caller that forgot to
        pass the run's own value trains on the plain table while the config records the ask — the same
        no-op shape as the report supervision that was wired to nothing for a week. `applied` itself is
        asserted directly in the unit tests; it is the path from the flag to here that nothing watched.

        Recorded rather than re-derived from the arrays because the arrays cannot answer robustly: a
        calibration is visible only as the ABSENCE of the five reader levels from the scores, so a
        subset that happens to contain none of them would read as calibrated and fail a smoke run for a
        property of the sample. A stored value that a smoke cannot false-positive on is worth more here
        than a derivation that can.
        """
        return {
            WEIGHT_FLOOR: self.supervision.floor,
            CALIBRATED: self.supervision.calibrated,
            FINDING: self.supervision.finding,
            SEVERITY_WEIGHT: self.supervision.severity,
            ANNOTATIONS: self.supervision.annotations,
            WITHHELD: self.supervision.withheld,
            SUSPECT: str(self.supervision.suspect) if self.supervision.suspect else "",
            UNREAD: str(self.supervision.unread) if self.supervision.unread else "",
        }

    @classmethod
    def load(
        cls, report_labels: Path, train_table: Path, identifiers: list[str], supervision: "Supervision | None" = None
    ) -> "Targets":
        """Read the derived table, then overwrite with annotations where a study has them.

        `floor` sets what an unmentioned finding still pulls with: 0.0 drops it (our original
        behaviour), `CONFIDENCE_FLOOR` reproduces the public baseline's `0.25 + 0.75*conf`.
        """
        # Aligned to OUR study order once per table, rather than looked up per (study, target). The
        # loop this replaces ran 4,407 x 12 = 52,884 pandas scalar lookups and re-scanned `.columns`
        # inside the inner loop; it also needed three `float(Any)` suppressions, because a `.at` scalar
        # is untyped. Whole columns are typed, so both the cost and the suppressions go together.
        supervision = supervision or Supervision()
        if supervision.withheld and supervision.annotations:
            raise ValueError(
                "hold_annotation_studies with annotations on would write the gold values into cells "
                "whose weight is then zeroed: a no-op wearing a contradiction. Pass --hold-annotations too."
            )
        floor = supervision.floor
        index = pd.Index(identifiers)
        derived = pd.read_csv(report_labels).set_index("StudyInstanceUID").reindex(index)
        annotated = pd.read_csv(train_table).set_index("StudyInstanceUID").reindex(index)

        scores = np.full((len(identifiers), len(TARGETS)), 0.5, dtype=np.float32)
        weights = np.zeros((len(identifiers), len(TARGETS)), dtype=np.float32)
        # float64 for the confidence arithmetic, not float32. The loop this replaces computed
        # `floor + (1 - floor) * raw` in Python floats and stored the float32 result; doing the
        # multiply in float32 instead lands one ULP away on 22,102 of the pairs. Harmless to training,
        # but it would make this refactor unprovable, and an equality oracle is worth more than an ULP.
        ones = np.ones(len(identifiers), dtype=np.float64)
        read_by_radiologist = np.zeros((len(identifiers), len(TARGETS)), dtype=np.bool_)
        for column, target in enumerate(TARGETS):
            if target in derived.columns:
                value = derived[target].to_numpy(dtype=np.float32)
                confidence = f"{target}__conf"
                # A missing confidence COLUMN and a missing confidence CELL both mean "fully
                # confident": the reader either does not report confidence at all, or reported none
                # for this row. Neither is evidence of doubt.
                raw = derived[confidence].to_numpy(dtype=np.float64) if confidence in derived.columns else ones
                # READ OFF THE LEVEL THE READER WROTE, which is why this precedes `applied`: a
                # calibration rewrites 0.68 to 0.378 in place, and a grade looked up afterwards would
                # match nothing and silently fall back to `__conf`. The two switches compose only in
                # this order.
                trust = supervision.trusted(value, np.nan_to_num(raw, nan=1.0))
                value = supervision.per_finding(supervision.applied(value), target)
                said = ~np.isnan(value)
                scores[said, column] = value[said]
                weights[said, column] = floor + (1.0 - floor) * trust[said]
            if supervision.annotations and target in annotated.columns:
                truth = annotated[target].to_numpy(dtype=np.float32)
                read = ~np.isnan(truth)
                scores[read, column] = truth[read]
                weights[read, column] = ANNOTATED_WEIGHT
                read_by_radiologist[read, column] = True
        # After the loop and not inside it, so the radiologist still wins: an annotated cell is never
        # an unread one, and the mask says so rather than the statement order.
        taught = supervision.taught(identifiers)
        override = ~np.isnan(taught) & ~read_by_radiologist
        scores[override] = taught[override]
        weights[supervision.pruned(identifiers) & ~read_by_radiologist] = 0.0
        # LAST, and after the prune, because it is the only switch that overrides the radiologist rather
        # than deferring to them: withholding the STUDY means the annotated cells are exactly the ones
        # that must stop pulling, so `~read_by_radiologist` would invert the intent.
        if supervision.withheld:
            # The columns PRESENT, not all twelve: a foreign-corpus or smoke table carries a subset, and
            # asking for a column it has no row for would kill the run on a property of the sample.
            graded = [target for target in TARGETS if target in annotated.columns]
            weights[annotated[graded].notna().to_numpy(dtype=np.bool_).any(axis=1)] = 0.0
        corpus = None
        if CORPUS in derived.columns:
            corpus = derived[CORPUS].fillna("").to_numpy(dtype=str)
        return cls(scores, weights, identifiers, supervision, corpus=corpus)

    @classmethod
    def pair(
        cls,
        labels: Path,
        eval_labels: Path | None,
        train_table: Path,
        identifiers: list[str],
        supervision: "Supervision | None" = None,
    ) -> tuple["Targets", "Targets"]:
        """The table to TRAIN on and the table to SCORE against, which are not always the same.

        Scoring a run against the labels it trained on makes two arms incomparable: a holdout AUC can
        rise because the labels agree with themselves more, not because the model improved. When the
        training labels are the variable under test, the ruler has to be held fixed — and ideally held
        at the table the interesting arm did NOT train on, so it earns no self-agreement credit.

        A CALIBRATION IS EXACTLY THAT KIND OF VARIABLE, so it is applied to the training table ONLY.
        Recalibrating the ruler too would move every arm's score by construction and make the change
        unmeasurable — the AUC would rise because the targets agree with themselves more. That also
        means the two tables stop being the same object when a calibration is given, even though they
        are read from the same file.
        """
        supervision = supervision or Supervision()
        training = cls.load(labels, train_table, identifiers, supervision)
        if eval_labels is None:
            plain = Supervision(floor=supervision.floor)
            ruler = cls.load(labels, train_table, identifiers, plain) if supervision.moves_the_ruler() else training
            return training, ruler
        return training, cls.load(eval_labels, train_table, identifiers)

    def rows(self, positions: NDArray[np.int64]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Scores and weights for a batch of study positions."""
        return self.scores[positions], self.weights[positions]

    def coverage(self) -> dict[str, float]:
        """The share of studies carrying any weight per target — where the vocabulary is thin shows here."""
        return {target: float((self.weights[:, column] > 0).mean()) for column, target in enumerate(TARGETS)}
