"""The three distance rulers read TOGETHER, because each alone names a different member as the good one.

`python -m kneemri.evaluation.member_axes --roster <records> --candidates <records>`

WHY A JOIN RATHER THAN A THIRD COLUMN. The repo now owns three ways to say two members differ, and each
was written with an honest counter saying it cannot be read alone:

- `MemberDistance` — per-label Spearman rank correlation. Invariant to any monotone rescaling, so it is
  blind to confidence, and it cannot tell a member that disagrees for a REASON from one that is merely
  loud: both decorrelate, by construction.
- `MemberDivergence` — Jensen-Shannon in bits over the same predictions read as Bernoulli beliefs. Sees
  confidence, which is what a logit average reads and what the champion ships on. Its counter, written
  in its own docstring: high divergence with high rank correlation is a member that is only differently
  SCALED, and seating it buys what a temperature knob would have.
- `MemberMahalanobis` — the same predictions with the POOL'S covariance divided out. Its counter is that
  it is taken in prediction space rather than on embeddings, and that its shrinkage is a knob.

Read singly they disagree about who to seat, and each disagreement is information the other two hold the
answer to. The JOIN is the instrument; the columns were only ever its inputs.

THE FOUR CELLS, and what each says to do:

  rank-near  + mahalanobis-near   TWIN      — seating it buys nothing. If divergence is FAR it is
                                              SCALED instead: same opinion, different confidence, which
                                              a temperature knob buys more cheaply than a seat.
  rank-FAR   + mahalanobis-near   NOISY     — it decorrelates, but along a direction the pool is
                                              already wide in, so the disagreement is variance rather
                                              than a second opinion. This is the cell `MemberDistance`
                                              alone cannot see, and it is the cell that most flatters a
                                              weak member.
  rank-FAR   + mahalanobis-FAR    DISTINCT  — decorrelated along an axis the pool does not span. The
                                              seat worth buying, and the only cell that is one.
  rank-near  + mahalanobis-FAR    PUZZLE    — orders like the roster yet sits off its span. Reported,
                                              never acted on: it is most likely a member whose ranks
                                              agree on the easy studies and scatter on the thin ones.

WHERE THE THRESHOLDS COME FROM, since a cell boundary is exactly the sort of number this project refuses
to invent. Each ruler's boundary is the ROSTER'S OWN median nearest-neighbour reading on that ruler: a
candidate counts as near when it is nearer to the roster than a typical seated pair already is. That
scale is the one `MemberDistance.spread` already prints and the one the seat argument is already taken
on, it moves when the roster moves, which is correct, and it means no cell can be defined without a
roster big enough to have neighbours.

IT TAKES NO DECISION. `knee-iyad`'s adoption gate stands: the Mahalanobis column must first reproduce
board-validated seat decisions before it may overturn one, and until it has, this report is a way of
reading candidates rather than a way of ranking them. `record_mix` does not consume it.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core import cli
from kneemri.evaluation.member_distance import MemberDistance
from kneemri.evaluation.member_divergence import MemberDivergence
from kneemri.evaluation.member_mahalanobis import MemberMahalanobis
from kneemri.evaluation.member_space import MemberSpace
from kneemri.evaluation.pool import Pool
from kneemri.evaluation.record_mix import RecordMix

logger = logging.getLogger(__name__)

NEIGHBOURED = 2
"""The smallest roster whose spread is a scale — same bar and same reason as the three rulers it joins."""


@dataclass(frozen=True)
class MemberReading:
    """One candidate on all three rulers, with the cell those readings put it in."""

    name: str
    agreement: float
    """Nearest seated member by rank correlation — HIGHER is nearer."""
    divergence: float
    """Nearest seated member in Jensen-Shannon bits — LOWER is nearer."""
    separation: float
    """Nearest seated member in whitened pool coordinates — LOWER is nearer."""
    cell: str

    def line(self) -> str:
        """One row, the cell first because it is what the reader is here for."""
        readings = f"rank {self.agreement:.4f}  js {self.divergence:.4f}  maha {self.separation:.3f}"
        return f"  {self.cell:9s} {self.name}: {readings}"


class MemberAxes:
    """Roster and candidates on three rulers at once, each candidate placed in the cell they agree on."""

    def __init__(self, roster: Pool, candidates: Pool) -> None:
        self.roster = roster
        self.candidates = candidates

    def whitened(self) -> tuple[list[list[float]], list[list[float]]]:
        """Nearest-in-pool distances for the seated members and for the candidates, in one whitening.

        ONE space for both, which is the point: whitening the roster alone would divide out a covariance
        the candidates never entered, and every candidate would then read as far along the axes the
        roster happens to be narrow in. The pool that defines the axes has to be the pool being judged.
        """
        combined = Pool(
            self.roster.names + self.candidates.names,
            self.roster.predictions + self.candidates.predictions,
            self.roster.truth,
        )
        matrix = MemberMahalanobis(MemberSpace(combined)).distances()
        seats = len(self.roster.names)
        seated = [[float(matrix[index, other]) for other in range(seats) if other != index] for index in range(seats)]
        candidates = [[float(value) for value in matrix[index, :seats]] for index in range(seats, len(combined.names))]
        return seated, candidates

    def scale(self) -> tuple[float, float, float]:
        """The roster's own median nearest reading on each ruler — the three cell boundaries.

        Median rather than mean because one seated twin pair would drag a mean and redefine every cell,
        and the boundary is meant to describe the typical seated neighbour rather than the closest one.
        """
        distance, divergence = MemberDistance(self.roster, self.roster), MemberDivergence(self.roster, self.roster)
        seated, _ = self.whitened()
        agreements = [max(distance.against(member, skip=index)) for index, member in enumerate(self.roster.predictions)]
        divergences = [
            min(divergence.against(member, skip=index)) for index, member in enumerate(self.roster.predictions)
        ]
        spans = [min(row) for row in seated]
        return float(np.median(agreements)), float(np.median(divergences)), float(np.median(spans))

    @staticmethod
    def celled(agreement: float, divergence: float, separation: float, scale: tuple[float, float, float]) -> str:
        """Which of the four cells these three readings put a candidate in.

        PURE, and separate from the pool arithmetic above, because this is the only judgement in the
        module and it should be readable — and testable — without a pool to build it from.
        """
        near_rank, near_span = agreement >= scale[0], separation <= scale[2]
        if near_rank and near_span:
            return "SCALED" if divergence > scale[1] else "TWIN"
        if near_span:
            return "NOISY"
        return "PUZZLE" if near_rank else "DISTINCT"

    def readings(self) -> list[MemberReading]:
        """Every candidate on all three rulers, most distant in rank space first."""
        scale = self.scale()
        distance = MemberDistance(self.roster, self.candidates)
        divergence = MemberDivergence(self.roster, self.candidates)
        _, whitened = self.whitened()
        rows = []
        for index, (name, candidate) in enumerate(zip(self.candidates.names, self.candidates.predictions, strict=True)):
            agreement = max(distance.against(candidate))
            bits = min(divergence.against(candidate))
            separation = min(whitened[index])
            cell = self.celled(agreement, bits, separation, scale)
            rows.append(MemberReading(name, agreement, bits, separation, cell))
        return sorted(rows, key=lambda reading: reading.agreement)

    def report(self) -> list[str]:
        """The candidates by cell, with the roster's own scale printed above them."""
        if len(self.roster.names) < NEIGHBOURED:
            return [f"  no scale: a roster of {len(self.roster.names)} has no seated pair to set a boundary"]
        rank, bits, span = self.scale()
        boundaries = f"rank {rank:.4f}  js {bits:.4f}  maha {span:.3f}"
        lines = [f"member axes: roster of {len(self.roster.names)}, boundaries {boundaries}"]
        lines += [reading.line() for reading in self.readings()]
        lines.append("  DISTINCT is the only cell that is a seat; NOISY is the one a rank ruler alone would flatter")
        lines.append("  ADOPTION GATE (knee-iyad): the maha column must reproduce board-validated seats first")
        return lines


def main() -> None:
    """Read candidates on three rulers: `python -m kneemri.evaluation.member_axes --roster <a> --candidates <b>`."""
    command = cli.Command(MemberAxes.__doc__, logger)
    command.parser.add_argument("--records", type=Path, nargs="+", required=True, help="every member's holdout record")
    command.parser.add_argument("--roster", type=Path, nargs="+", required=True, help="which of them are seated")
    arguments = command.parsed()
    roster, candidates = RecordMix(list(arguments.records)).per_member().split(list(arguments.roster))
    logger.info("%s", "\n".join(MemberAxes(roster, candidates).report()))


if __name__ == "__main__":
    main()
