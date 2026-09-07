"""Read several runs' metrics files side by side: `python -m kneemri.evaluation.run_comparison`."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from core import cli
from core.data.corpus.anatomy import ScaleProfile
from core.data.labels.label_power import FOLDS, LabelPower
from kneemri.evaluation.gold_rows import GoldRows
from kneemri.evaluation.holdout_metrics import HoldoutMetrics, ScoredEpoch
from kneemri.evaluation.scores import LabelScores
from kneemri.evaluation.supervision_gap import Bottleneck, SupervisionGap
from kneemri.runs.holdout_record import TRAINED, HoldoutRecord
from kneemri.runs.recorded_run import RecordedRun
from kneemri.runs.run_metrics import METRICS_SUFFIX, EpochRow, RunMetrics

logger = logging.getLogger(__name__)

NOT_AN_AXIS = {
    "out",
    "vram_gb",
    "paths",
    "labels",
    "eval_labels",
    "compile_encoder",
    "recompute_activations",
}
"""Options that differ between any two runs without making them unmatched: where the member is written,
how much of the card it was allowed, which files it read them from, and the two throughput flags. Listed
rather than inferred, so a genuinely confounding option can never be dropped by a heuristic — the whole
value of this line is that everything it does NOT name is a difference somebody has to justify.

The throughput pair is here on a MEASUREMENT, not on the claim that recompute and compile are
mathematically neutral. The `throughput` arm ran all four combinations for one epoch on the same fold:
plain 0.7604, compiled 0.7606, checkpointing on 0.7612, off 0.7601 — a spread of 0.0011 against a
0.0053 noise floor. Reporting them as confounds made every older run unmatched against every newer one
on a difference smaller than the instrument can see, which is the failure mode this list exists to
prevent in the other direction."""


VIEW_AXES = frozenset({"crop_mm", "centre", "img", "train_size"})
"""The keys that change HOW MUCH OF THE KNEE, and at what density, reaches the encoder.

Spelled as the checkpoint's CONFIG spells them, not as the flags do, because the config is the only
record that spans both eras of the ledger: `wsfold0_e6`'s options block carries no crop at all while its
config states `crop_mm 130.0`. Read off the recipe instead, a crop arm against that control looked like
an option the control merely predated, and got no verdict at all — the false negative that replaced the
false positive.

`ScaleProfile`'s four verdicts are all sentences about the view, so they may only be printed under an
arm that moved one of these. Anything else — schedule, loss, depth, pooling, encoder — leaves the field
of view and the pixel density exactly as the control had them, and its focal/regional split is then a
description of where its deltas landed, not evidence about resolution."""


OFF_VALUES = frozenset({"False", "0", "0.0", "None", ""})
"""What a recipe value reads as when the switch it names is not doing anything.

Recipes are recorded as strings, so a flag that is off and a rate that is zero are the same statement in
two spellings; grouping them here is what lets one rule read every axis."""

SUPERVISION_AXES = frozenset(
    {
        "labels",
        "eval_labels",
        "calibrated_targets",
        "severity_weight",
        "hold_annotations",
        "per_label_loss",
        "rank_weight",
        "report_weight",
        "report_vectors",
        "weight_floor",
        "normalise_by_count",
    }
)
"""The options that change what the model is TOLD. Everything else changes how it looks at the pixels.

The split is exhaustive on that test rather than on a taxonomy of arms: a flag either alters the target
or the weight attached to it, or it acts somewhere on the path from voxels to prediction. Resolution,
depth, pooling, capacity, rate and schedule are all the second kind, which is why they are not listed —
naming them would make this a list somebody has to keep complete, and it would silently misclassify the
next flag nobody remembered to add. The named set is the small one, and the default is the safe one.

WHY THE KIND MATTERS: `SupervisionGap` measures that half the findings are held back by the report
reader and half by the model, and an arm is only allowed to be judged on the half it can move. Averaged
over all twelve, either change is diluted by the six it cannot affect. The comparison INFERS this from
the recorded recipes rather than having each arm declare it — the arms are shell scripts and a
declaration would only cover runs made after it landed, while the recipes are already on disk for every
run in the ledger."""


@dataclass(frozen=True)
class PairedWidths:
    """One pair's own per-label half-widths, and whether the arrays behind them are the exact ones.

    The two travel together because the number alone cannot be read: a floor measured on a back-filled
    run is a floor measured on a re-derivation, and the re-derivation's own tolerance (4x eps across
    precisions) is wider than the half-widths being quoted.
    """

    widths: dict[str, float]
    exact: bool

    def named(self) -> str:
        """The sentence the report prints under the arm, saying what marked it and how far to trust it."""
        caveat = "" if self.exact else " - INDICATIVE, one of these runs had its predictions re-derived"
        return f"paired floor, this pair's own: half-widths up to +/-{max(self.widths.values()):.4f}{caveat}"


class RunComparison:
    """Several runs read side by side, per label, against one control.

    The missing half of this module. Every arm has written a metrics file since the recorder landed and
    nothing read one, so comparing arms still meant grepping `best holdout` out of log files by hand —
    the exact practice that once credited a seed change as a depth result.

    PER LABEL, not just the mean, because that is what makes the current arms falsifiable. The
    resolution arm predicts a gain CONCENTRATED in MCL, ACL and the menisci and near zero for the large
    fluid collections; the extended-slots arm predicts MCL and LCL specifically. A uniform lift refutes
    both stories while leaving the mean looking like a win, and a mean is exactly what hides that.
    """

    def __init__(
        self,
        control: Path,
        arms: list[Path],
        labels: Path | None = None,
        bottleneck: Bottleneck | None = None,
        excluded: frozenset[str] | None = None,
    ) -> None:
        self.control = RunMetrics(control)
        self.arms = [RunMetrics(arm) for arm in arms]
        self.errors = self._intervals(labels) if labels else {}
        # PASSED IN, never measured here: the split costs the published corpus and its alignment to the
        # annotations, which is `SupervisionGap`'s subject and not this module's. Absent, every arm still
        # reports which half it moved — it just does not get the deltas restricted to that half.
        self.bottleneck = bottleneck
        self.scores = LabelScores()
        # The studies every number in this report is taken WITHOUT. Empty for every comparison between
        # two of our own runs; non-empty when one side chose its weights on studies the other held out,
        # which is a property of the foreign arm and not of the fold.
        self.excluded = excluded or frozenset()
        self.holdout = HoldoutMetrics(self.scores, self.excluded)
        # CACHED because reading a run's account can cost a `torch.load` of the whole checkpoint, and
        # the report asks each arm for its recipe twice — once to say whether the pair is matched and
        # once to say which half of the pipeline it moved.
        self._recorded: dict[Path, RecordedRun] = {}

    @staticmethod
    def resolved(run: Path, processed: Path) -> Path:
        """A run given by NAME becomes the checkpoint of that name; a run given by path is left alone.

        The ledger loop reads every arm through this tool, and the name is what the ledger, the arm
        script and the log all call a run — the checkpoint's absolute path appears nowhere a reader
        would have it to hand. Given a bare name this used to look for
        `centre_tissue_s2026.metrics.jsonl` in the CURRENT DIRECTORY and report "control does not
        exist", which reads as a missing run rather than as a path that was never going to resolve.

        A name is anything with no directory part and no suffix — plus the `.metrics` form, which is a
        name wearing a suffix. THIS TOOL PRINTS THAT FORM: every arm in its own report and every row in
        the ledger reads `lr4e-5_ws_s2026.metrics`, because that is the record the score was taken from.
        Pasting a line of the output back in as a control resolved it against the CURRENT DIRECTORY and
        reported "does not exist" — the same failure the paragraph above describes, surviving in the one
        spelling a reader is most likely to have to hand.

        That leaves `./x.pt`, `x.pt` and an absolute path working exactly as before, so nothing that
        already ran has to change.
        """
        if run.parent != Path() or run.suffix not in ("", METRICS_SUFFIX):
            return run
        return processed / f"{run.stem}.pt"

    @staticmethod
    def _intervals(labels: Path) -> dict[str, float]:
        """Each label's 95% sampling interval, so no delta is printed without its own precision.

        `LabelPower` has carried these since it was written and NOTHING read them at the point deltas
        are quoted — which is how four arms with unrelated mechanisms were each recorded improving
        Fracture by +0.012 to +0.037 against an interval of +/-0.061. The knowledge existed; it did not
        reach the place it was needed. Same shape as the config values this project keeps recording and
        not reading, in prose rather than in code.
        """
        power = LabelPower(labels)
        studies = len(power.table) // FOLDS
        return {
            target: 1.96 * power.standard_error(total // FOLDS, studies - total // FOLDS)
            for target, total in power.positives().items()
        }

    def scored(self, run: RunMetrics) -> ScoredEpoch | None:
        """One run's best epoch from whichever of its two records can state it.

        A run that predates the recorder still kept its predictions, and `wsfold0_e6` — the control
        knee-bot's whole argument rests on — is exactly that run: this comparison used to refuse it as
        "does not exist" while the arrays that answer it sat beside the checkpoint under another name.
        """
        return self.holdout.of(run)

    def recipe(self, run: RunMetrics) -> dict[str, str] | None:
        """One run's recipe from whichever of its two records carries it.

        Its metrics file OR its checkpoint stamp, the same rule the back-fill reads splits by. Reading
        only the first is what made this comparison call `wsfold0_e6` unrecorded — the control the
        whole extended-slots finding rests on, whose checkpoint stamps every option it ran under.
        """
        if run.path not in self._recorded:
            self._recorded[run.path] = RecordedRun.of(run)
        return self._recorded[run.path].recipe

    def deltas(self) -> dict[str, dict[str, float]]:
        """Per-arm, per-label difference against the control's best epoch."""
        control = self.scored(self.control)
        if control is None:
            # Distinguished, because they need different fixes: a missing file is usually the wrong
            # path, and an empty one is a run that died before its first scored epoch.
            missing = "recorded no scored epoch and kept no predictions" if self.control.exists() else "does not exist"
            message = f"control {self.control.path} {missing}"
            raise ValueError(message)
        reference = control.row
        table: dict[str, dict[str, float]] = {}
        for arm in self.arms:
            best = self.scored(arm)
            if best is None:
                continue
            table[arm.path.stem] = {
                label: round(score - reference["per_label"].get(label, float("nan")), 4)
                for label, score in best.row["per_label"].items()
            }
            table[arm.path.stem]["MEAN"] = round(best.row["auc"] - reference["auc"], 4)
        return table

    def report(self) -> str:
        """The comparison as a table, widest delta first so the concentration is visible at a glance."""
        # deltas() raises on a control with no scored epoch, so it is called BEFORE the header is
        # formatted - reading `reference` first would crash on None instead of saying which run is empty.
        table = self.deltas()
        control = self.scored(self.control)
        if control is None:  # pragma: no cover - deltas() has already raised
            message = f"control {self.control.path} recorded no scored epoch"
            raise ValueError(message)
        reference = control.row
        lines = [
            f"control {self.control.path.stem}: mean {reference['auc']:.4f} at epoch {reference['epoch']}",
            f"  {control.named()}",
            *([f"  taken WITHOUT {len(self.excluded)} excluded studies"] if self.excluded else []),
            *self._stalled(self.control),
        ]
        for arm in self.arms:
            lines.extend(self._arm(arm, table.get(arm.path.stem)))
        if self.errors:
            lines.append(
                "\n* the delta exceeds that label's own 95% sampling interval; unmarked deltas are "
                "smaller than the precision of the instrument that produced them. Fracture carries 62 "
                "holdout positives and an interval of +/-0.061, which is why four arms with unrelated "
                "mechanisms all 'improved' it."
            )
            lines.append(
                "  WHICH BOUND is stated per arm above. The unpaired one is deliberately conservative: "
                "two arms scored on the SAME studies share their sampling error, so the paired floor is "
                "tighter - 2.2x on a null pair of twins, and less than that on any real pair. A run has "
                "kept its per-study predictions since knee-mx17 and is marked against its own paired "
                "floor; an arm that predates them is UNREAD against the wide bound, not refuted by it."
            )
        return "\n".join(lines)

    def _arm(self, arm: RunMetrics, deltas: dict[str, float] | None) -> list[str]:
        """One arm's block of the table — its deltas, the floor they are marked against, and its stalls.

        Its own method because `report` was doing two jobs at one indent: assembling the document, and
        rendering a row of it. The row is the part that grows every time a run learns to record something
        new, and growing it was pushing the assembler towards the complexity gate for reasons that had
        nothing to do with assembling anything.
        """
        if deltas is None:
            # NAMED, never skipped. A mistyped arm used to vanish from the report entirely, so the
            # reader saw a shorter table and no reason for it — `intensity392` (the file is
            # `intensity392_ws`) came back as a report of the control alone, which reads as an arm
            # that moved nothing rather than as a path that matches no run.
            why = "recorded no scored epoch and kept no predictions" if arm.exists() else "does not exist"
            return [f"\n{arm.path.stem}  NOT READ: {why}"]
        ordered = sorted(deltas.items(), key=lambda item: -abs(item[1]))
        intervals, named = self.floor(arm)
        profile = ScaleProfile.of(deltas)
        scored = self.scored(arm)
        return [
            f"\n{arm.path.stem}  " + "  ".join(f"{label} {self._marked(label, v, intervals)}" for label, v in ordered),
            *self._peaks_line(arm, scored),
            f"  marked against the {named}",
            profile.judged() if self.moved_view(arm) else profile.line(),
            f"  vs control: {self.unmatched(arm)}",
            f"  {self.reader_change(arm)}",
            *self.judged_on(arm, deltas),
            *self._stalled(arm),
            *([f"  {scored.named()}"] if scored is not None and scored.reconstructed else []),
        ]

    @staticmethod
    def _peaks(run: RunMetrics) -> dict[str, tuple[float, int]]:
        """Each label's OWN best AUC across every recorded epoch, and the epoch index it peaked at.

        The delta line reads every label at ONE epoch — the one the run's MEAN peaked at. A label that
        peaked two epochs earlier and gave it back is invisible there, which is the shape blendlabels
        wore: a mean of +0.0034 over seven helped and five diluted labels. The per-label trajectory was
        being recorded all along; this is the reader it never had.
        """
        peaks: dict[str, tuple[float, int]] = {}
        for row in run.rows():
            for label, value in row["per_label"].items():
                if np.isnan(value):
                    continue
                if label not in peaks or value > peaks[label][0]:
                    peaks[label] = (value, row["epoch"])
        return peaks

    def _peaks_line(self, arm: RunMetrics, scored: ScoredEpoch | None) -> list[str]:
        """Each label's own peak beside its value at the global-best epoch, or nothing when unrecorded.

        A LIST rather than a string, like `_stalled`: a run with no epoch rows contributes no line and
        needs no branch to say so. A reconstructed arm keeps only its best epoch, so its peak IS the
        value at best and the line would restate the delta line.
        """
        if scored is None:
            return []
        peaks = self._peaks(arm)
        if not peaks:
            return []
        at_best = scored.row["per_label"]
        shown = ", ".join(
            f"{label} {peak:.4f}@e{epoch} ({at_best.get(label, float('nan')):.4f} at best)"
            for label, (peak, epoch) in sorted(peaks.items())
        )
        return [f"  label peaks across epochs: {shown}"]

    @staticmethod
    def _stalled(run: RunMetrics) -> list[str]:
        """One line per stalled epoch, or nothing at all for a run that has none.

        Here because this report is where a run's cost is READ, and a stall is exactly the thing that
        makes a cost line a lie: `slices16_ws_s2026` carries one 30,684 s epoch among epochs of 404-530,
        and a reader who saw only the total would record the arm as 8.8x its control rather than 1.3x.
        Three runs on disk lost 20.8 hours this way and none of them said so.

        A LIST rather than a string, so the report appends it without asking whether there was anything
        to append — a run with no stall contributes no line and needs no branch to say so.
        """
        return [
            f"  STALLED epoch {row['epoch']}: {row['seconds']}s against a median of "
            f"{run.epoch_seconds():.0f}s{RunComparison._began(row)} — wall-clock the card spent doing nothing"
            for row in run.stalls()
        ]

    @staticmethod
    def _began(row: EpochRow) -> str:
        """When the epoch started, in UTC, or nothing for a run recorded before epochs carried the field.

        The clock time is the whole point of naming the stall: `seconds` says an epoch took eight hours
        and only a timestamp can line that up against the suspend/resume the machine is suspected of.
        Runs that predate the field still report their stall — without the one thing that would explain
        it, which is the state this change exists to end rather than to hide.
        """
        started = row.get("started")
        return "" if started is None else f", from {datetime.fromtimestamp(started, tz=UTC):%Y-%m-%d %H:%M UTC}"

    def paired(self, arm: RunMetrics) -> PairedWidths | None:
        """This pair's OWN per-label 95% half-widths, or None when either run kept no predictions.

        The unpaired bound is each label's sampling error from its positive count, and it is far too
        conservative to read a knob by: two arms scored on the same studies share that error almost
        entirely, so the DIFFERENCE keeps only what is left. Measured on a null pair of seed twins the
        paired interval came out 2.2x tighter — but twins are the most correlated pair that exists here,
        which makes 2.2x a ceiling rather than this pair's number. This computes this pair's number.

        None rather than a fallback, because "the floor could not be measured" and "the floor is wide"
        are different statements and the report says which one it is printing.
        """
        control, theirs = HoldoutRecord(self.control.path).read(), HoldoutRecord(arm.path).read()
        if control is None or theirs is None:
            return None
        # THE SAME STUDIES the deltas were taken on. A floor measured over rows the delta above does not
        # include is a floor for a different comparison, and it would be quoted beside this one.
        first, second, truth = self.holdout.kept(control).aligned(self.holdout.kept(theirs))
        interval = self.scores.paired_interval(first, second, truth)
        return PairedWidths(
            widths={label: (high - low) / 2 for label, (low, high) in interval.items()},
            exact=control.source == TRAINED and theirs.source == TRAINED,
        )

    def floor(self, arm: RunMetrics) -> tuple[dict[str, float], str]:
        """The intervals this arm's deltas are marked against, and the sentence naming which they are."""
        paired = self.paired(arm)
        if paired is None:
            return self.errors, "UNPAIRED bound (one of these runs kept no per-study predictions)"
        return paired.widths, paired.named()

    @staticmethod
    def disagreements(control: dict[str, str], theirs: dict[str, str]) -> dict[str, str]:
        """The keys BOTH records carry and state differently — the only kind of difference either can prove.

        One home, because three judgements here ask the same question and two of them once answered it
        differently. A key only the arm recorded is an option the control PREDATES: it ran whatever that
        day's default was, which nothing on disk can read back. Counting that absence as a disagreement is
        what made a schedule arm and a loss arm both print "READER change (calibrated_targets,
        per_label_loss)" and collect a resolution verdict they had not earned.
        """
        return {name: value for name, value in theirs.items() if name in control and control[name] != value}

    def unmatched(self, arm: RunMetrics) -> str:
        """What this arm changed BESIDES the axis under test, read from both runs' recorded recipes.

        A per-label delta is a claim about one axis, and it is only that if nothing else moved. Two of
        the arms in the ledger changed depth and encoder together and were read as capacity results.
        Runs recorded before the recipe existed say so plainly rather than silently comparing as
        matched — an unknown control is a worse failure than a mismatched one.
        """
        control, theirs = self.recipe(self.control), self.recipe(arm)
        if control is None or theirs is None:
            return "recipe unrecorded, so nothing here can say the two runs are matched"
        axes = {name: value for name, value in theirs.items() if name not in NOT_AN_AXIS}
        # Two different statements, kept apart because they need different judgements. An option BOTH
        # runs recorded and disagree on is a confound outright. An option only the arm recorded is a
        # question — the control predates the flag, so it ran whatever the default was that day, which
        # this cannot read. Merging them would let `train_size unset→0`, a no-op on every arm, sit in
        # the same list as `backbone_lr 8e-06→4e-05`, which is a second axis on a resolution result.
        moved = [f"{name} {control[name]}→{value}" for name, value in self.disagreements(control, axes).items()]
        newer = [f"{name}={value}" for name, value in axes.items() if name not in control]
        said = ", ".join(moved) if moved else "matched on every option both runs recorded"
        return said if not newer else f"{said}; control predates {', '.join(newer)}"

    def view(self, run: RunMetrics) -> dict[str, str] | None:
        """The reading one run was fitted under, from the checkpoint block that records it in both eras."""
        if run.path not in self._recorded:
            self._recorded[run.path] = RecordedRun.of(run)
        return self._recorded[run.path].view

    def moved_view(self, arm: RunMetrics) -> bool:
        """Whether this arm changed the field of view or the pixel density the control read at.

        Read off the two CONFIGS rather than the two recipes. The recipe is the flag list, and the flag
        list is era-dependent: most of the ledger's controls predate `crop_mm` and `centre` as flags, so
        against a recipe every crop arm reads either as a difference the control cannot confirm (a
        verdict it has not earned) or as an option the control merely predates (no verdict for an arm
        that plainly moved the view). The config carries the DERIVED reading — `crop_mm 130.0` for a
        control whose flags never mention a crop — so both runs can state the axis and the comparison is
        real either way.

        A run whose config is unrecorded counts as NOT having moved the view: the verdict is the strong
        claim, and an unknown reading is not grounds to make it.
        """
        control, theirs = self.view(self.control), self.view(arm)
        if control is None or theirs is None:
            return False
        return bool(self.disagreements(control, {name: value for name, value in theirs.items() if name in VIEW_AXES}))

    def moved(self, arm: RunMetrics) -> set[str] | None:
        """Every option this arm changed against the control, or None when either recipe is unrecorded.

        The recipe diff PLUS the view axes the two configs disagree on, because a crop arm against a
        control that predates the crop flag moved the view without either recipe being able to say so.
        One home, because the kind of change and the half it may be judged on are two readings of this
        same set, and computing it twice is how they would come to disagree.

        PLUS the switches the control PREDATES and this arm turned on. `disagreements` refuses those on
        purpose — a key only the arm recorded is usually an option that ran at the day's default, and
        counting `train_size unset→0` as a change is what once handed a schedule arm a resolution
        verdict. But an arm cannot turn a flag ON by default: `calibrated_targets=True` against a
        control predating the flag IS the axis under test, and the blanket refusal printed this
        project's canonical supervision arm as a MODEL change and judged it on the wrong six findings.
        Only a non-off value counts, which is the same reading `switched_on` gives every flag.
        """
        control, theirs = self.recipe(self.control), self.recipe(arm)
        if control is None or theirs is None:
            return None
        seen, read = self.view(self.control), self.view(arm)
        moved = set(self.disagreements(control, theirs)) | self.switched_on(control, theirs)
        if seen is not None and read is not None:
            moved |= set(self.disagreements(seen, read)) & VIEW_AXES
        return moved

    @staticmethod
    def switched_on(control: dict[str, str], theirs: dict[str, str]) -> set[str]:
        """The axes only the ARM recorded, and set to something other than off.

        An option the control predates ran at that day's default, which nothing on disk can read back —
        so it is normally not a difference either record can prove. The exception is a flag the arm set
        to a non-default: no control, whatever its era, had `calibrated_targets` on before the flag
        existed. `OFF_VALUES` is what a switch reads as when it is not doing anything, so the numeric
        and boolean forms are one rule rather than two.
        """
        return {name for name, value in theirs.items() if name not in control and value not in OFF_VALUES}

    def judged_on(self, arm: RunMetrics, deltas: dict[str, float]) -> list[str]:
        """The same deltas, restricted to the findings this arm's kind of change could actually move.

        NOT A NEW SCORE, and the line says so: the ledger number stays the twelve-label mean, or the
        ledger stops being comparable with itself. This is the half that carries the arm's claim, printed
        beside the headline rather than instead of it — a reader change diluted over the six findings
        where the model already beats its own labels reads as noise at a floor of 0.0053, and that
        dilution has been the standing objection to every label-side arm in the ledger.

        Nothing at all without a measured `Bottleneck`: the split is a measurement over the published
        held-out corpus, and a guessed one would be a grouping edited until it agreed with the arm.
        """
        moved = self.moved(arm)
        if self.bottleneck is None or not moved:
            return []
        supervision = bool(moved & SUPERVISION_AXES)
        side = self.bottleneck.label_side if supervision else self.bottleneck.model_side
        scored = {label: deltas[label] for label in side if label in deltas}
        if not scored:
            return ["  its half of the findings is not among this run's labels, so there is nothing to restrict to"]
        half = "label-bottlenecked" if supervision else "model-bottlenecked"
        mean = sum(scored.values()) / len(scored)
        return [
            f"  on its own {len(scored)} {half}: mean {mean:+.4f} ({', '.join(sorted(scored))})",
            "  — the arm's CLAIM, not its score; the ledger number stays the twelve-label mean above",
        ]

    def reader_change(self, arm: RunMetrics) -> str:
        """Which half of the pipeline this arm moved, read from the two recipes rather than declared.

        NOT filtered through `NOT_AN_AXIS`: that list answers "is this pair matched", and a swapped
        label table is the clearest reader change there is while being exactly the sort of option a
        confound check waves through. Kind and confound are different questions asked of the same diff.

        An arm that moves both is reported as a reader change, because the reader half is the one that
        cannot be judged on the model-bottlenecked findings — the narrower claim is the honest one.

        The view is asked of the config as well as the recipe, so a crop arm against a control that
        predates the crop FLAG still reports the change it made rather than "moves nothing".
        """
        moved = self.moved(arm)
        if moved is None:
            return "kind unknown, one of these runs recorded no recipe"
        if not moved:
            return "moves nothing, so there is no axis to judge it on"
        supervision = sorted(moved & SUPERVISION_AXES)
        if supervision:
            return f"READER change ({', '.join(supervision)}); judge on the label-bottlenecked findings"
        return "MODEL change (pixels to prediction); judge on the model-bottlenecked findings"

    @staticmethod
    def _marked(label: str, value: float, intervals: dict[str, float]) -> str:
        """A delta with a star when it clears its own label's 95% interval, and bare when it cannot."""
        interval = intervals.get(label)
        star = "*" if interval is not None and abs(value) > interval else ""
        return f"{value:+.4f}{star}"


def main() -> None:
    """Compare arms per label: `python -m kneemri.evaluation.run_comparison --control a --arms b c`."""
    command = cli.Command(RunComparison.__doc__, logger)
    command.parser.add_argument(
        "--control", type=Path, required=True, help="the checkpoint the arms are measured against"
    )
    command.parser.add_argument("--arms", type=Path, nargs="+", required=True)
    command.parser.add_argument("--labels", type=Path, help="the label table, to print each delta's own error bar")
    command.parser.add_argument(
        "--split",
        action="store_true",
        help="also print each arm's deltas over the half of the findings its kind of change can move",
    )
    command.parser.add_argument(
        "--exclude-gold",
        action="store_true",
        help="score without the radiologist-annotated studies, for an arm that selected its epoch on them",
    )
    arguments = command.parsed()
    # OPT-IN, because the split costs the twenty published checkpoints and their alignment to the
    # annotations — seconds, and a `--labels` table this comparison does not otherwise need. Asking for
    # it without the table is REFUSED rather than silently skipped: a restriction that quietly did not
    # happen reads as an arm whose claim was not worth printing.

    root = command.rooted(arguments)
    control = RunComparison.resolved(arguments.control, root.models())
    arms = [RunComparison.resolved(arm, root.models()) for arm in arguments.arms]
    split = None
    if arguments.split:
        if arguments.labels is None:
            message = "--split measures the report labels against the annotations, so it needs --labels"
            raise SystemExit(message)
        split = SupervisionGap().measured(arguments.labels, root.raw() / "train.csv", root.processed() / "published")
    # WHY A FLAG AND NOT A DEFAULT. Between two of our own runs the gold studies are held out by both
    # and dropping them would throw away 58 of the fold's rows for nothing. It earns its keep against a
    # FOREIGN arm whose trainer validated on gold-58 and kept its best epoch by that number: every gold
    # study inside our holdout then chose that arm's weights, and the delta credits it with a selection
    # it made. Which studies overlap is read from train.csv, never assumed to be a count.
    excluded = frozenset(GoldRows().annotated(root.raw() / "train.csv")[0]) if arguments.exclude_gold else None
    logger.info("%s", RunComparison(control, arms, arguments.labels, split, excluded).report())


if __name__ == "__main__":
    main()
