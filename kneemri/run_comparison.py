"""Read several runs' metrics files side by side: `python -m kneemri.run_comparison`."""

import argparse
import logging
from pathlib import Path

from kneemri.label_power import FOLDS, LabelPower
from kneemri.run_metrics import RunMetrics

logger = logging.getLogger(__name__)


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
        for name, deltas in table.items():
            ordered = sorted(deltas.items(), key=lambda item: -abs(item[1]))
            lines.append(f"\n{name}  " + "  ".join(f"{label} {self._marked(label, value)}" for label, value in ordered))
        if self.errors:
            lines.append(
                "\n* the delta exceeds that label's own 95% sampling interval; unmarked deltas are "
                "smaller than the precision of the instrument that produced them. Fracture carries 62 "
                "holdout positives and an interval of +/-0.061, which is why four arms with unrelated "
                "mechanisms all 'improved' it."
            )
            lines.append(
                "  IT IS AN UNPAIRED BOUND, deliberately conservative: two arms scored on the SAME "
                "studies share their sampling error, so a paired delta has a tighter floor than this. "
                "An unmarked delta is not refuted, it is UNREAD - what it needs is the paired floor."
            )
        return "\n".join(lines)

    def _marked(self, label: str, value: float) -> str:
        """A delta with a star when it clears its own label's 95% interval, and bare when it cannot."""
        interval = self.errors.get(label)
        star = "*" if interval is not None and abs(value) > interval else ""
        return f"{value:+.4f}{star}"


def main() -> None:
    """Compare arms per label: `python -m kneemri.run_metrics --control a.pt --arms b.pt c.pt`."""
    parser = argparse.ArgumentParser(description=RunComparison.__doc__)
    parser.add_argument("--control", type=Path, required=True, help="the checkpoint the arms are measured against")
    parser.add_argument("--arms", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=Path, help="the label table, to print each delta's own error bar")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("%s", RunComparison(arguments.control, arguments.arms, arguments.labels).report())


if __name__ == "__main__":
    main()
