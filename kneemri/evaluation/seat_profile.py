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

THE CUT IS LEDGER75, matching `seat_add` and `seat_weight`, and the header names it. A per-finding leader
is an ORDERING, and the midpoint is the cut that orders pairs the other way from the board: on the 39
ensemble entries carrying both, midpoint inverts 12 of 414 resolved pairs and ledger75 inverts 0 of 397.
`--midpoint` asks the other question deliberately.
"""

import logging
from pathlib import Path

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.ledger75 import LedgerSeventyFive
from kneemri.evaluation.member_pool import DECLINES, MemberPool
from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.runs.holdout_record import SUFFIX, HoldoutReading, HoldoutRecord

logger = logging.getLogger(__name__)


class SeatProfile:
    """The finding-by-finding profile of each member that kept a holdout record."""

    def __init__(self, scores: LabelScores | None = None) -> None:
        """`scores` defaults to LEDGER75, the cut the seat instruments answer on."""
        self.scores = scores if scores is not None else LedgerSeventyFive().scorer()

    def scored(self, path: Path) -> dict[str, float]:
        """One member's twelve AUCs, keyed by finding."""
        reading = HoldoutRecord(path).required()
        return self.scores.per_label(reading.predictions, reading.truth)

    def margins(
        self, readings: dict[str, HoldoutReading], table: dict[str, dict[str, float]], against: str
    ) -> dict[str, tuple[str, float, tuple[float, float]]]:
        """Each finding's leader, its margin over one named member, and the 95% interval of that margin.

        A leader is the maximum over the whole pool on the same studies that price it, so its margin is
        selection-optimistic by construction and a positive one proves nothing. The interval is what makes
        the table decidable in ONE direction: a margin whose interval spans zero is not a per-finding edge
        even before the selection is discounted, so that column is CLOSED, and only a margin that clears
        its own interval survives as a candidate still owing an out-of-pool confirmation.

        `spread` says where the pool disagrees; this says whether the disagreement is larger than the
        studies it was read on, which is the question that decides whether a column is worth an arm.
        """
        baseline = readings[against]
        intervals: dict[str, dict[str, tuple[float, float]]] = {}
        margins = {}
        for target in TARGETS:
            name, best = max(
                ((name, row[target]) for name, row in table.items() if name != against), key=lambda pair: pair[1]
            )
            if name not in intervals:
                # `paired_interval` reports SECOND minus FIRST, so the baseline goes first for the
                # interval to bracket `best - table[against]` rather than its negative.
                intervals[name] = self.scores.paired_interval(
                    baseline.predictions, readings[name].predictions, baseline.truth
                )
            margins[target] = (name, best - table[against][target], intervals[name][target])
        return margins

    def table(self, paths: list[Path]) -> dict[str, dict[str, float]]:
        """Every member that has a record, keyed by the checkpoint's stem then by finding.

        THE POOL IS CHECKED ONTO ONE RULER BEFORE A SINGLE NUMBER IS COMPARED. This project holds several
        label tables and members were fitted against different ones, so a column-by-column comparison
        across members is only meaningful where the truth beside each is the same truth. `aligned` is
        what already refuses that case for a pair; running it against the first reading extends the
        refusal to the whole set, which is the shape this table is read in.
        """
        return self.profile(self.readings(paths))

    def profile(self, readings: dict[str, HoldoutReading]) -> dict[str, dict[str, float]]:
        """The twelve AUCs of each member in an already-read set, so a caller needing both pays one read."""
        return {name: self.scores.per_label(reading.predictions, reading.truth) for name, reading in readings.items()}

    def readings(self, paths: list[Path]) -> dict[str, HoldoutReading]:
        """Every member's kept arrays, checked onto one ruler, keyed by the member name the ledger quotes."""
        readings = {
            path.name.removesuffix(SUFFIX).removesuffix(".pt"): HoldoutRecord(path).required() for path in paths
        }
        first = next(iter(readings.values()))
        for reading in readings.values():
            first.aligned(reading)
        return readings

    def population(self, paths: list[Path], against: str) -> list[Path]:
        """The members answering the SAME question as `against` — same studies, same table that graded them.

        A pool directory holds records from several populations, and two of this project's carry a moved
        label table (knee-g2vm9): `recal_holdannot_s2026` reads 0.9004 on its own truth and 0.8888 on the
        corpus table, which a flat per-finding rank prints in the same column as everybody else. Filtering
        by `LedgerSeventyFive.population` is the same digest the ledger divides its rows by, so a member
        this drops is one no ledger row ever ranked against the baseline either.
        """
        ruler = LedgerSeventyFive()
        wanted = ruler.population(ruler.reading(next(path for path in paths if against in path.name)))
        kept = [path for path in paths if ruler.population(ruler.reading(path)) == wanted]
        if len(kept) < len(paths):
            logger.info(f"population @{wanted}: {len(kept)} of {len(paths)} members answer {against}'s question")
        return kept

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
    source = command.parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--members", type=Path, nargs="+", help="member checkpoints")
    source.add_argument("--holdouts", type=Path, help="directory of .holdout.npz records, void ones dropped")
    command.parser.add_argument("--declines", type=Path, default=DECLINES, help="the declines file to read void from")
    command.parser.add_argument("--midpoint", action="store_true", help="read at the midpoint instead of LEDGER75")
    command.parser.add_argument("--against", help="member each finding's leader is priced against")
    arguments = command.parsed()

    profile = SeatProfile(LabelScores(MIDPOINT) if arguments.midpoint else None)
    if arguments.holdouts:
        pool = MemberPool.at(arguments.holdouts, arguments.declines)
        paths = [arguments.holdouts / f"{name}{SUFFIX}" for name in pool.names()]
    else:
        paths = list(arguments.members)
    if arguments.against:
        paths = profile.population(paths, arguments.against)
    readings = profile.readings(paths)
    table = profile.profile(readings)
    leaders, spread = profile.leaders(table), profile.spread(table)
    margins: dict[str, tuple[str, float, tuple[float, float]]] = (
        profile.margins(readings, table, arguments.against) if arguments.against else {}
    )

    lines = [f"--- per finding @{profile.scores.name()}"]
    header = f"{'finding':22s} {'leader':28s} {'best':>7s} {'spread':>7s}"
    lines += [header + (f"  {'margin':>8s} {'95% interval':>18s}  clears" if margins else "")]
    for target in sorted(TARGETS, key=lambda name: spread[name], reverse=True):
        leader = margins[target][0] if margins else leaders[target][0]
        row = f"{target:22s} {leader[:28]:28s} {table[leader][target]:7.4f} {spread[target]:7.4f}"
        if margins:
            _, margin, (low, high) = margins[target]
            row += f"  {margin:8.4f} [{low:8.4f},{high:8.4f}]  {'YES' if low > 0 else 'no'}"
        lines.append(row)
    lines.append("")
    for name, row in sorted(table.items(), key=lambda item: -sum(item[1].values())):
        mean = sum(row.values()) / len(row)
        worst = min(TARGETS, key=lambda target: row[target])
        lines.append(f"{name[:34]:34s} mean {mean:.4f}  weakest {worst} {row[worst]:.4f}")
    command.report(lines)


if __name__ == "__main__":
    main()
