"""Per-member, per-finding AUC over the shared holdout: `python -m kneemri.evaluation.seat_profile`.

WHY A MEAN OF TWELVE HIDES THE THING THAT PICKS SEATS. Every member number this project quotes is one
scalar, and `competence_axis` reads 81.7-82% of what separates 105 members as a SINGLE latent — which is
the mechanism under seven board flats, because rearranging members that differ along one axis cannot buy
diversity. The 18% that is NOT that axis is per-finding, and no tool here has ever printed it. A member
that is mid-table overall while leading Fracture is invisible to `seat_value`, `member_curve` and the
solo ordering alike, and Fracture alone is 27% of the variance of every holdout number quoted.

WHAT IT CANNOT SEE, STATED FIRST. A member fitted at `--fold -1` held nothing out, so it has no record
here and never will. That is most of the champion: this profile reads the fold-0 pool, and a leader
named below is the best of what can be MEASURED, not the best that could be seated.

The AUCs come out of `LabelScores.per_label` against the truth stored beside each member's predictions,
so they are the same arrays the ledger's numbers were read off, not a re-derivation.
"""

import logging
from pathlib import Path

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.scores import LabelScores
from kneemri.runs.holdout_record import HoldoutReading, HoldoutRecord

logger = logging.getLogger(__name__)


class SeatProfile:
    """The finding-by-finding profile of each member that kept a holdout record."""

    def reading(self, path: Path) -> HoldoutReading:
        """One member's kept predictions, its truth and the epoch they came from."""
        record = HoldoutRecord(path)
        reading = record.read() if record.exists() else None
        if reading is None:
            message = f"{path} left no holdout record - a --fold -1 member has none and can never have one"
            raise FileNotFoundError(message)
        return reading

    def scored(self, path: Path) -> dict[str, float]:
        """One member's twelve AUCs, keyed by finding."""
        reading = self.reading(path)
        return LabelScores().per_label(reading.predictions, reading.truth)

    def table(self, paths: list[Path]) -> dict[str, dict[str, float]]:
        """Every member that has a record, keyed by the checkpoint's stem then by finding.

        THE POOL IS CHECKED ONTO ONE RULER BEFORE A SINGLE NUMBER IS COMPARED. This project holds several
        label tables and members were fitted against different ones, so a column-by-column comparison
        across members is only meaningful where the truth beside each is the same truth. `aligned` is
        what already refuses that case for a pair; running it against the first reading extends the
        refusal to the whole set, which is the shape this table is read in.
        """
        readings = {path.stem: self.reading(path) for path in paths}
        first = next(iter(readings.values()))
        for reading in readings.values():
            first.aligned(reading)
        return {name: LabelScores().per_label(reading.predictions, reading.truth) for name, reading in readings.items()}

    def leaders(self, table: dict[str, dict[str, float]]) -> dict[str, tuple[str, float]]:
        """The member that scores highest on each finding, which is the question a mean cannot answer."""
        return {
            target: max(((name, row[target]) for name, row in table.items()), key=lambda pair: pair[1])
            for target in TARGETS
        }

    def spread(self, table: dict[str, dict[str, float]]) -> dict[str, float]:
        """Best minus worst per finding — where the pool disagrees, and so where a seat can still pay."""
        return {
            target: max(row[target] for row in table.values()) - min(row[target] for row in table.values())
            for target in TARGETS
        }


def main() -> None:
    """Print the member-by-finding matrix, the per-finding leader, and where the pool spreads widest."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--members", type=Path, nargs="+", required=True, help="member checkpoints")
    arguments = command.parsed()

    profile = SeatProfile()
    table = profile.table(list(arguments.members))
    leaders, spread = profile.leaders(table), profile.spread(table)

    lines = [f"{'finding':22s} {'leader':28s} {'best':>7s} {'spread':>7s}"]
    lines += [
        f"{target:22s} {leaders[target][0][:28]:28s} {leaders[target][1]:7.4f} {spread[target]:7.4f}"
        for target in sorted(TARGETS, key=lambda name: spread[name], reverse=True)
    ]
    lines.append("")
    for name, row in sorted(table.items(), key=lambda item: -sum(item[1].values())):
        mean = sum(row.values()) / len(row)
        worst = min(TARGETS, key=lambda target: row[target])
        lines.append(f"{name[:34]:34s} mean {mean:.4f}  weakest {worst} {row[worst]:.4f}")
    command.report(lines)


if __name__ == "__main__":
    main()
