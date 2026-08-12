"""Read several runs' metrics files side by side: `python -m kneemri.run_comparison`."""

import argparse
import logging
from pathlib import Path

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

    def __init__(self, control: Path, arms: list[Path]) -> None:
        self.control = RunMetrics(control)
        self.arms = [RunMetrics(arm) for arm in arms]

    def deltas(self) -> dict[str, dict[str, float]]:
        """Per-arm, per-label difference against the control's best epoch."""
        reference = self.control.best()
        if reference is None:
            message = f"control {self.control.path} recorded no scored epoch"
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
            lines.append(f"\n{name}  " + "  ".join(f"{label} {value:+.4f}" for label, value in ordered))
        return "\n".join(lines)


def main() -> None:
    """Compare arms per label: `python -m kneemri.run_metrics --control a.pt --arms b.pt c.pt`."""
    parser = argparse.ArgumentParser(description=RunComparison.__doc__)
    parser.add_argument("--control", type=Path, required=True, help="the checkpoint the arms are measured against")
    parser.add_argument("--arms", type=Path, nargs="+", required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("%s", RunComparison(arguments.control, arguments.arms).report())


if __name__ == "__main__":
    main()
