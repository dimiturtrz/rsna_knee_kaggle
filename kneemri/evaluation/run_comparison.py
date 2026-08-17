"""Read several runs' metrics files side by side: `python -m kneemri.evaluation.run_comparison`."""

import argparse
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kneemri.corpus.anatomy import ScaleProfile
from kneemri.evaluation.holdout_metrics import HoldoutMetrics, ScoredEpoch
from kneemri.evaluation.scores import LabelScores
from kneemri.labels.label_power import FOLDS, LabelPower
from kneemri.runs.holdout_record import TRAINED, HoldoutRecord
from kneemri.runs.recorded_run import RecordedRun
from kneemri.runs.run_metrics import EpochRow, RunMetrics

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


SUPERVISION_AXES = frozenset(
    {
        "labels",
        "eval_labels",
        "calibrated_targets",
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

    def __init__(self, control: Path, arms: list[Path], labels: Path | None = None) -> None:
        self.control = RunMetrics(control)
        self.arms = [RunMetrics(arm) for arm in arms]
        self.errors = self._intervals(labels) if labels else {}
        self.scores = LabelScores()
        self.holdout = HoldoutMetrics(self.scores)
        # CACHED because reading a run's account can cost a `torch.load` of the whole checkpoint, and
        # the report asks each arm for its recipe twice — once to say whether the pair is matched and
        # once to say which half of the pipeline it moved.
        self._recorded: dict[Path, RecordedRun] = {}

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
            *self._stalled(self.control),
        ]
        for arm in self.arms:
            deltas = table.get(arm.path.stem)
            if deltas is None:
                # NAMED, never skipped. A mistyped arm used to vanish from the report entirely, so the
                # reader saw a shorter table and no reason for it — `intensity392` (the file is
                # `intensity392_ws`) came back as a report of the control alone, which reads as an arm
                # that moved nothing rather than as a path that matches no run.
                why = "recorded no scored epoch and kept no predictions" if arm.exists() else "does not exist"
                lines.append(f"\n{arm.path.stem}  NOT READ: {why}")
                continue
            ordered = sorted(deltas.items(), key=lambda item: -abs(item[1]))
            intervals, named = self.floor(arm)
            lines.append(
                f"\n{arm.path.stem}  "
                + "  ".join(f"{label} {self._marked(label, v, intervals)}" for label, v in ordered)
            )
            lines.append(f"  marked against the {named}")
            profile = ScaleProfile.of(deltas)
            lines.append(profile.judged() if self.moved_view(arm) else profile.line())
            lines.append(f"  vs control: {self.unmatched(arm)}")
            lines.append(f"  {self.reader_change(arm)}")
            lines.extend(self._stalled(arm))
            scored = self.scored(arm)
            if scored is not None and scored.reconstructed:
                lines.append(f"  {scored.named()}")
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
        first, second, truth = control.aligned(theirs)
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
        control, theirs = self.recipe(self.control), self.recipe(arm)
        if control is None or theirs is None:
            return "kind unknown, one of these runs recorded no recipe"
        seen, read = self.view(self.control), self.view(arm)
        moved = set(self.disagreements(control, theirs))
        if seen is not None and read is not None:
            moved |= set(self.disagreements(seen, read)) & VIEW_AXES
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
    """Compare arms per label: `python -m kneemri.evaluation.run_comparison --control a.pt --arms b.pt c.pt`."""
    parser = argparse.ArgumentParser(description=RunComparison.__doc__)
    parser.add_argument("--control", type=Path, required=True, help="the checkpoint the arms are measured against")
    parser.add_argument("--arms", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=Path, help="the label table, to print each delta's own error bar")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("%s", RunComparison(arguments.control, arguments.arms, arguments.labels).report())


if __name__ == "__main__":
    main()
