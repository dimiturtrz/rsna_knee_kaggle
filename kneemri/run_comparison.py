"""Read several runs' metrics files side by side: `python -m kneemri.run_comparison`."""

import argparse
import logging
from pathlib import Path

from kneemri.anatomy import ScaleProfile
from kneemri.holdout_record import HoldoutRecord
from kneemri.label_power import FOLDS, LabelPower
from kneemri.run_metrics import RunMetrics
from kneemri.scores import LabelScores

logger = logging.getLogger(__name__)

NOT_AN_AXIS = {"out", "vram_gb", "paths", "labels", "eval_labels"}
"""Options that differ between any two runs without making them unmatched: where the member is written,
how much of the card it was allowed, and which files it read them from. Listed rather than inferred, so
a genuinely confounding option can never be dropped by a heuristic — the whole value of this line is
that everything it does NOT name is a difference somebody has to justify."""


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

    def deltas(self) -> dict[str, dict[str, float]]:
        """Per-arm, per-label difference against the control's best epoch."""
        reference = self.control.best()
        if reference is None:
            # Distinguished, because they need different fixes: a missing file is usually the wrong
            # path, and an empty one is a run that died before its first scored epoch.
            missing = "recorded no scored epoch" if self.control.exists() else "does not exist"
            message = f"control {self.control.path} {missing}"
            raise ValueError(message)
        table: dict[str, dict[str, float]] = {}
        for arm in self.arms:
            best = arm.best()
            if best is None:
                continue
            table[arm.path.stem] = {
                label: round(score - reference["per_label"].get(label, float("nan")), 4)
                for label, score in best["per_label"].items()
            }
            table[arm.path.stem]["MEAN"] = round(best["auc"] - reference["auc"], 4)
        return table

    def report(self) -> str:
        """The comparison as a table, widest delta first so the concentration is visible at a glance."""
        # deltas() raises on a control with no scored epoch, so it is called BEFORE the header is
        # formatted - reading `reference` first would crash on None instead of saying which run is empty.
        table = self.deltas()
        reference = self.control.best()
        if reference is None:  # pragma: no cover - deltas() has already raised
            message = f"control {self.control.path} recorded no scored epoch"
            raise ValueError(message)
        lines = [f"control {self.control.path.stem}: mean {reference['auc']:.4f} at epoch {reference['epoch']}"]
        for arm in self.arms:
            deltas = table.get(arm.path.stem)
            if deltas is None:
                continue
            ordered = sorted(deltas.items(), key=lambda item: -abs(item[1]))
            intervals, named = self.floor(arm)
            lines.append(
                f"\n{arm.path.stem}  "
                + "  ".join(f"{label} {self._marked(label, v, intervals)}" for label, v in ordered)
            )
            lines.append(f"  marked against the {named}")
            lines.append(ScaleProfile.of(deltas).line())
            lines.append(f"  vs control: {self.unmatched(arm)}")
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

    def paired(self, arm: RunMetrics) -> dict[str, float] | None:
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
        return {label: (high - low) / 2 for label, (low, high) in interval.items()}

    def floor(self, arm: RunMetrics) -> tuple[dict[str, float], str]:
        """The intervals this arm's deltas are marked against, and the sentence naming which they are."""
        paired = self.paired(arm)
        if paired is None:
            return self.errors, "UNPAIRED bound (one of these runs kept no per-study predictions)"
        widest = max(paired.values())
        return paired, f"paired floor, this pair's own: half-widths up to +/-{widest:.4f}"

    def unmatched(self, arm: RunMetrics) -> str:
        """What this arm changed BESIDES the axis under test, read from both runs' recorded recipes.

        A per-label delta is a claim about one axis, and it is only that if nothing else moved. Two of
        the arms in the ledger changed depth and encoder together and were read as capacity results.
        Runs recorded before the recipe existed say so plainly rather than silently comparing as
        matched — an unknown control is a worse failure than a mismatched one.
        """
        control, theirs = self.control.recipe(), arm.recipe()
        if control is None or theirs is None:
            return "recipe unrecorded, so nothing here can say the two runs are matched"
        axes = [(name, value) for name, value in theirs.items() if name not in NOT_AN_AXIS]
        # Two different statements, kept apart because they need different judgements. An option BOTH
        # runs recorded and disagree on is a confound outright. An option only the arm recorded is a
        # question — the control predates the flag, so it ran whatever the default was that day, which
        # this cannot read. Merging them would let `train_size unset→0`, a no-op on every arm, sit in
        # the same list as `backbone_lr 8e-06→4e-05`, which is a second axis on a resolution result.
        moved = [
            f"{name} {control[name]}→{value}" for name, value in axes if name in control and control[name] != value
        ]
        newer = [f"{name}={value}" for name, value in axes if name not in control]
        said = ", ".join(moved) if moved else "matched on every option both runs recorded"
        return said if not newer else f"{said}; control predates {', '.join(newer)}"

    @staticmethod
    def _marked(label: str, value: float, intervals: dict[str, float]) -> str:
        """A delta with a star when it clears its own label's 95% interval, and bare when it cannot."""
        interval = intervals.get(label)
        star = "*" if interval is not None and abs(value) > interval else ""
        return f"{value:+.4f}{star}"


def main() -> None:
    """Compare arms per label: `python -m kneemri.run_comparison --control a.pt --arms b.pt c.pt`."""
    parser = argparse.ArgumentParser(description=RunComparison.__doc__)
    parser.add_argument("--control", type=Path, required=True, help="the checkpoint the arms are measured against")
    parser.add_argument("--arms", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=Path, help="the label table, to print each delta's own error bar")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("%s", RunComparison(arguments.control, arguments.arms, arguments.labels).report())


if __name__ == "__main__":
    main()
