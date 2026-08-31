"""Distance between members with the POOL'S OWN COVARIANCE divided out — far only along an unspanned axis.

`python -m kneemri.evaluation.member_mahalanobis --roster comp/members.roster --candidates <dir>`

THE INCUMBENT AND ITS GAP (knee-iyad). `member_distance` prices diversity as per-label Spearman rank
correlation, averaged, reported as NEAREST and MEAN. It is the most decision-heavy tool in the repo —
it declined `raptor_distill` and `distill_pairs`, and `knee-for6` closed on its verdict. It has two
blind spots, both of which are what a Mahalanobis distance is FOR:

1. **It does not divide out what the pool already spans.** Two members can each be a fair distance from
   every seated member and yet be far in the SAME direction the pool is already wide in, which buys an
   ensemble nothing. Whitening by the pool covariance makes a member count as far only along an axis the
   pool is narrow in.
2. **It cannot separate DISTINCT from NOISY.** Rank correlation falls when a member disagrees for a
   reason and falls when a member is merely loud; both decorrelate, by construction. Under whitening the
   noise directions are the ones the pool has most variance in, so they are the ones shrunk hardest.

WHAT IT MEASURES HERE, AND WHAT IT DOES NOT. The features are per-label RANK-NORMALISED predictions on
the shared studies — the space members are actually combined in — not encoder embeddings. The bead names
embeddings as the ideal, and they are: representation distance is nearer the decode-path quantity
measured to transfer to the board at ~2.4x its slope. An embedding dump costs a forward pass per member
over the holdout and does not exist yet, so this reads the space we HAVE while `MemberSpace` keeps the
feature source a parameter, which is the seam an embedding dump plugs into without touching the ruler.
The gap between the two spaces is real and is not closed by this module.

THE RANK-DEFICIENCY PROBLEM, ANSWERED IN THE OPEN. With ~121 members and tens of thousands of feature
dimensions the covariance is catastrophically rank-deficient: the members span at most `m - 1`
directions and every other direction has exactly zero variance, where a Mahalanobis distance is
undefined rather than large. So the distance is taken IN THE SPAN, and the eigenvalues of that span are
shrunk toward their own mean in the Ledoit-Wolf manner — `(1 - a) * lambda_k + a * mean(lambda)`. At
`a = 1` this is plain Euclidean distance on the centred features; at `a = 0` it is the raw whitening
that hands the smallest, noisiest eigen-direction unbounded leverage. The bead's honest counter is that
`a` is a knob that can quietly fit the answer, so `sensitivity` reports the ordering at several `a` and
the report prints it: a nearest-neighbour verdict that survives only one shrinkage is not a verdict.

ADOPTION GATE, NON-NEGOTIABLE, AND NOT AUTOMATED HERE. This ruler must first REPRODUCE the seat
decisions the board has already validated before it is trusted to overturn one — a ruler that disagrees
with the board on day one is a bug, not a finding. `report` prints both orderings side by side for
exactly that comparison and takes no decision of its own. Nothing in `record_mix` consumes this module,
and nothing should until that gate is answered in writing on knee-iyad.
"""

import logging
from pathlib import Path

import numpy as np
from jaxtyping import Float

from core import cli
from kneemri.evaluation.member_space import MemberSpace
from kneemri.evaluation.record_mix import RecordMix

logger = logging.getLogger(__name__)

SHRINKAGES = (0.05, 0.2, 0.5)
"""The shrinkages `sensitivity` reads the ordering at. Spread over the decade where the answer can
plausibly change rather than clustered: 0.05 is near-raw whitening, 0.5 is halfway to Euclidean."""

NEIGHBOURED = 2
"""How many nearest members a summary names, matching `member_distance` so the two reports can be read
against each other line for line."""


class MemberMahalanobis:
    """Pairwise member distance with the pool's covariance divided out, at a stated shrinkage."""

    def __init__(self, space: MemberSpace, shrinkage: float = SHRINKAGES[1]) -> None:
        self.space = space
        self.shrinkage = shrinkage

    def coordinates(self, shrinkage: float | None = None) -> Float[np.ndarray, "members components"]:
        """The members in whitened coordinates, where a plain Euclidean distance IS the Mahalanobis one.

        Taken through the SVD of the centred matrix rather than by forming and inverting a covariance:
        the covariance is `features x features` and would be tens of thousands on a side, while its
        non-zero spectrum is exactly the `m - 1` squared singular values already in hand.
        """
        weight = self.shrinkage if shrinkage is None else shrinkage
        matrix = self.space.matrix()
        components, values, _ = np.linalg.svd(matrix, full_matrices=False)
        variance = values**2 / max(len(matrix) - 1, 1)
        kept = variance > variance.max() * np.finfo(np.float64).eps * len(variance)
        shrunk = (1.0 - weight) * variance[kept] + weight * variance[kept].mean()
        return components[:, kept] * values[kept] / np.sqrt(shrunk)

    def distances(self, shrinkage: float | None = None) -> Float[np.ndarray, "members members"]:
        """The full pairwise matrix, diagonal set to infinity so a member is never its own neighbour."""
        points = self.coordinates(shrinkage)
        gaps = points[:, None, :] - points[None, :, :]
        matrix = np.sqrt((gaps**2).sum(axis=-1))
        np.fill_diagonal(matrix, np.inf)
        return matrix

    def nearest(self, shrinkage: float | None = None) -> list[tuple[float, str, str]]:
        """Each member paired with its closest other member, ordered closest first — the twin detector.

        The ordering, not the number, is what a reader takes from this: the units are whitened
        standard deviations of a pool that changes every time a member is added.
        """
        matrix = self.distances(shrinkage)
        names = self.space.pool.names
        closest = matrix.argmin(axis=1)
        pairs = [(float(matrix[index, other]), names[index], names[other]) for index, other in enumerate(closest)]
        return sorted(pairs)

    def sensitivity(self, shrinkages: tuple[float, ...] = SHRINKAGES) -> dict[float, list[str]]:
        """The twin ordering at several shrinkages — the knob the bead warned would quietly fit the answer.

        A member that reads as a twin at every shrinkage is a twin. One that reads as a twin at 0.05 and
        as distinct at 0.5 is a member the ruler cannot price, and saying so is the honest output.
        """
        return {weight: [pair[1] for pair in self.nearest(weight)] for weight in shrinkages}

    def unstable(self, shrinkages: tuple[float, ...] = SHRINKAGES) -> list[str]:
        """The members whose twin RANK moves by more than a neighbourhood across the shrinkages.

        These are the ones no verdict may be taken on. Named rather than counted, because the point of
        the check is to stop a specific decision, not to score the ruler.
        """
        orderings = self.sensitivity(shrinkages)
        places = [{name: place for place, name in enumerate(order)} for order in orderings.values()]

        def travelled(name: str) -> int:
            return max(place[name] for place in places) - min(place[name] for place in places)

        return sorted(name for name in self.space.pool.names if travelled(name) > NEIGHBOURED)

    def report(self) -> list[str]:
        """The closest pairs, plus the members the shrinkage knob disagrees about.

        Prints no seat decision and consumes no roster verdict: knee-iyad's adoption gate is that this
        ruler must first reproduce board-validated seats, and a report that ranked members would invite
        exactly the overturn the gate forbids until the comparison is written down.
        """
        lines = [f"member mahalanobis (shrinkage {self.shrinkage}, {len(self.space.pool)} members)"]
        lines += [f"  {distance:7.3f}  {name} ~ {other}" for distance, name, other in self.nearest()[:NEIGHBOURED]]
        unstable = self.unstable()
        lines.append(f"  shrinkage-unstable ({len(unstable)}): {' '.join(unstable) if unstable else 'none'}")
        lines.append("  ADOPTION GATE (knee-iyad): reproduce board-validated seats before overturning one")
        return lines


def main() -> None:
    """Read the pool's whitened distances: `python -m kneemri.evaluation.member_mahalanobis --records <paths>`."""
    command = cli.Command(MemberMahalanobis.__doc__, logger)
    command.parser.add_argument("--records", type=Path, nargs="+", required=True, help="member holdout records")
    command.parser.add_argument("--shrinkage", type=float, default=SHRINKAGES[1], help="eigenvalue shrinkage in [0, 1]")
    arguments = command.parsed()
    pool = RecordMix(list(arguments.records)).per_member()
    ruler = MemberMahalanobis(MemberSpace(pool), arguments.shrinkage)
    logger.info("%s", "\n".join(ruler.report()))


if __name__ == "__main__":
    main()
