"""What a seat is worth: the board gain of an ensemble over the SOLO entry of its own first member.

The question this answers is the one every roster decision turns on — if a single model scores S on the
board, what does an ensemble built around it score? — and the project has been answering it from three
hand-picked pairings quoted in prose (`knee-urvf` planned on +0.0143). The complete `comp/board.jsonl`
answers it far better than that, because two members were each scored ALONE and each lead a family of
scored ensembles. Member identity is then held fixed across a whole curve and only the seat count moves.
The `final_ws_cf` family, nine entries deep (`--anchor final_ws_cf`):

    n=1   calibsolo       0.923   +0.000
    n=2   onepair_v1      0.936   +0.013
    n=3   trioseat_v1     0.938   +0.015
    n=4   fourseat_v1     0.937   +0.014
    n=6   calibsix_v1     0.941   +0.018
    n=7   raptorseat_v1   0.943   +0.020
    n=8   fleettwo_v1     0.942   +0.019
    n=8   pixgroup_v1     0.940   +0.017
    n=11  calibmix_v1     0.941   +0.018

THE BONUS SATURATES. Two seats buy two thirds of everything eight seats buy, and past seven seats the
curve is flat or falling. That is the number that prices a roster: not "more members is more score" but
"a seventh seat is worth +0.001 and an eleventh is worth nothing".

WHAT THIS IS NOT. It is not a regression and it is not a law with an error bar. Three caveats, each of
which would be invisible if this lived in prose:

  * KITS DIFFER. The nine entries span three kits (the other anchor's fourteen span seven), and the
    kit is not held fixed the way the anchor
    member is. `soloseat_v1` scores 0.925 on kit 8243a95 with `final_ws_e12` where `calibsolo` scores
    0.923 on a642ba8 — so the ZERO of this curve is itself uncertain by about the size of one seat.
  * THE SPREADS ARE UNDER THE FLOOR. n=3, n=4, n=6, n=8 and n=11 differ by 0.001-0.002, and
    `BOARD_RESOLVABLE` is 0.0015. The SATURATION is real (0.013 -> 0.020 is not); the ORDERING among
    the saturated points is not a reading.
  * TWO ENTRIES BUY THEIR TOP SEATS WITH A FOREIGN CHECKPOINT. `raptorseat_v1` and `fleettwo_v1` seat
    published foreign members, which is a different kind of diversity from another of our own arms.

THERE IS MORE THAN ONE ANCHOR AND THEY DISAGREE. `final_ws_e12` leads fourteen scored entries and was
itself scored alone at 0.925; its curve saturates at **+0.014**. `final_ws_cf` leads nine and was scored
alone at 0.923; its curve saturates at **+0.020**. Both anchors' ensembles land in the same 0.936-0.943
band, so most of that 0.006 disagreement is the two ZEROS differing, not the seats buying more. This is
why `anchors()` and `alternatives()` are public and the report prints both: a module that quoted the
larger candidate would have turned a confound into a constant.

WHAT IT PRICES. A 0.950 board entry needs a single at `TARGET - saturation()`: **0.930 to 0.936**
depending on the anchor, against our best measured single of 0.925. So the standing target on ONE MEMBER
is +0.005 to +0.011 — the bead's +0.011 is the pessimistic end of the range rather than the answer, and
the range itself is the deliverable. That is the whole reason this module exists rather than a
paragraph: this project has now had two constants misquoted from prose.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from core import cli

logger = logging.getLogger(__name__)

RECORD = Path("comp/board.jsonl")
"""The machine-readable half of `comp/SUBMISSIONS.md`, joined on the Kaggle submission ref."""

ENSEMBLE = "ensemble"
"""The only scope whose board numbers are subtractable from one another."""

SOLO_SEATS = 1
"""How many members make an entry the ANCHOR rather than a point on the curve."""

TARGET = 0.950
"""The board score being planned toward: one point above the tenth place of the prize band."""

BOARD_RESOLVABLE = 0.0015
"""Below this two board numbers are the same number, so a difference of one seat's spread is not read."""


@dataclass(frozen=True)
class SeatEntry:
    """One scored ensemble submission, with the roster it was built from in seating order."""

    id: str
    board: float
    members: tuple[str, ...]
    kit: str

    @property
    def seats(self) -> int:
        """How many members were averaged into this entry."""
        return len(self.members)

    @property
    def first(self) -> str:
        """The member seated first, which is the anchor a bonus is read against."""
        return self.members[0]


@dataclass(frozen=True)
class Seating:
    """One point on the curve: an entry, and what its seats bought over the anchor alone."""

    entry: SeatEntry
    bonus: float


class SeatBonus:
    """The board's own answer to what an ensemble seat is worth, read against one fixed anchor member.

    It refuses rather than guesses. If no member is seated first by more than one entry, or if the most
    common anchor has no SOLO scored entry to subtract, there is no zero for the curve and the reading
    does not exist — which is the honest outcome for a record that has not yet spent a slot on the
    single. Every previous version of this number was a subtraction between two entries picked by hand.
    """

    def __init__(self, record: Path = RECORD, scope: str = ENSEMBLE, chosen: str | None = None) -> None:
        self.record = record
        self.scope = scope
        self.chosen = chosen

    def _rows(self) -> list[dict[str, object]]:
        """Every recorded line, exactly as the file holds them."""
        return [json.loads(line) for line in self.record.read_text(encoding="utf-8").splitlines() if line]

    @staticmethod
    def _seatable(row: dict[str, object]) -> bool:
        """Whether a row carries a board number AND the roster that produced it, in order.

        A row missing `members_named` is not a defect: excluded scopes and the older entries predate the
        field. It simply cannot be a point on a curve about seat counts.
        """
        return row.get("board") is not None and bool(row.get("members_named"))

    def entries(self) -> list[SeatEntry]:
        """Every scored entry on this scope that names its roster, fewest seats first."""
        found = [
            SeatEntry(
                id=str(row["id"]),
                board=float(str(row["board"])),
                members=tuple(str(member) for member in list(row["members_named"])),  # type: ignore[call-overload]
                kit=str(row.get("kit", "")),
            )
            for row in self._rows()
            if row.get("scope") == self.scope and self._seatable(row)
        ]
        return sorted(found, key=lambda entry: entry.seats)

    def solos(self) -> dict[str, SeatEntry]:
        """Every member the board has scored ALONE, by name — the only members a curve can have a zero at."""
        return {entry.first: entry for entry in self.entries() if entry.seats == SOLO_SEATS}

    def anchor(self) -> str:
        """The member seated first by the most entries AMONG THOSE SCORED ALONE.

        The two conditions are not the same and the record separates them: `band_dense_s2026` leads the
        first seat outright, and no slot was ever spent on it alone, so it can anchor nothing. Picking
        the most-seated member and then discovering it has no zero is how this reading fails silently.
        """
        if self.chosen is not None:
            if self.chosen not in self.solos():
                message = f"asked to anchor on {self.chosen!r}, which the board never scored alone"
                raise ValueError(message)
            return self.chosen
        common = self.anchors()
        if not common:
            message = f"no member on scope {self.scope!r} is both seated first and scored alone"
            raise ValueError(message)
        return common[0]

    def anchors(self) -> list[str]:
        """Every member that both leads a roster and was scored alone, most-seated first.

        There is more than one, and they do not agree: `final_ws_e12` anchors fourteen entries and
        saturates at +0.014, `final_ws_cf` anchors nine and saturates at +0.020. The bonus is a property
        of the anchor and its company, not a constant of ensembling, and a module that reported only the
        largest candidate would have hidden exactly that.
        """
        alone = self.solos()
        counted = Counter(entry.first for entry in self.entries() if entry.first in alone)
        return [member for member, _ in counted.most_common()]

    def alternatives(self) -> dict[str, float]:
        """Each candidate anchor's own saturation, so the chosen one is read next to the others."""
        return {
            member: SeatBonus(self.record, self.scope, member).saturation()
            for member in self.anchors()
            if SeatBonus(self.record, self.scope, member).curve()
        }

    def solo(self) -> SeatEntry:
        """The scored entry that is the anchor ALONE, which is the curve's zero.

        Without it every bonus would be measured against another ensemble and the curve would describe a
        difference of rosters rather than a number of seats.
        """
        return self.solos()[self.anchor()]

    def curve(self) -> list[Seating]:
        """Every entry anchored on that member, with what its extra seats bought, fewest seats first."""
        anchor = self.anchor()
        zero = self.solo()
        return [
            Seating(entry=entry, bonus=entry.board - zero.board)
            for entry in self.entries()
            if entry.first == anchor and entry.seats > SOLO_SEATS
        ]

    def saturation(self) -> float:
        """The largest bonus any seat count bought — what the board says more members is worth at best."""
        return max(seating.bonus for seating in self.curve())

    def knee(self) -> Seating:
        """The FEWEST seats that reach within the board's resolution of the saturation.

        This is the roster-size decision: seats past it are unpaid weight, and weight is runtime, which
        is the efficiency prize's whole currency.
        """
        best = self.saturation()
        return next(seating for seating in self.curve() if seating.bonus >= best - BOARD_RESOLVABLE)

    def required_single(self, target: float = TARGET) -> float:
        """The solo board score a member needs for an ensemble around it to reach `target`."""
        return target - self.saturation()

    def kits(self) -> set[str]:
        """The distinct kits the curve's points were built with — the confound this reading cannot hold."""
        return {seating.entry.kit for seating in self.curve()} | {self.solo().kit}

    def report(self, target: float = TARGET) -> list[str]:
        """The curve, the saturation, and what a single would have to score."""
        zero = self.solo()
        lines = [
            f"anchor {self.anchor()}, zero {zero.id} at board {zero.board:.3f} (kit {zero.kit})",
            f"  n={zero.seats:<3} {zero.id:<16} {zero.board:.3f}  +0.000  {zero.kit}",
        ]
        lines += [
            f"  n={seating.entry.seats:<3} {seating.entry.id:<16} {seating.entry.board:.3f}  "
            f"{seating.bonus:+.3f}  {seating.entry.kit}"
            for seating in self.curve()
        ]
        knee = self.knee()
        lines += [
            f"saturation {self.saturation():+.3f}, first reached at {knee.entry.seats} seats "
            f"({knee.entry.id}); seats past that are runtime the board does not pay for",
            f"a {target:.3f} entry needs a single at {self.required_single(target):.3f} board",
            f"{len(self.kits())} distinct kits across these points, so the zero itself is not held fixed; "
            f"spreads under {BOARD_RESOLVABLE} are not orderings",
            "other anchors: " + ", ".join(f"{member} {value:+.3f}" for member, value in self.alternatives().items()),
        ]
        return lines


def main() -> None:
    """Price a seat: `python -m kneemri.evaluation.seat_bonus`."""
    command = cli.Command(SeatBonus.__doc__, logger)
    command.parser.add_argument("--record", type=Path, default=RECORD, help="the scored-submission record")
    command.parser.add_argument("--scope", default=ENSEMBLE, help="which denominator to read within")
    command.parser.add_argument(
        "--target",
        type=float,
        default=TARGET,
        help="the board score being planned toward; the report says what single reaches it",
    )
    command.parser.add_argument(
        "--anchor",
        default=None,
        help="read the curve against this member instead of the most-seated one that was scored alone",
    )
    arguments = command.parsed()
    command.report(SeatBonus(arguments.record, arguments.scope, arguments.anchor).report(arguments.target))


if __name__ == "__main__":
    main()
