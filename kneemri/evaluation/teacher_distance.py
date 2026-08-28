"""How far foreign members sit from EACH OTHER, measured with no labels and no submission slot.

`python -m kneemri.evaluation.teacher_distance --teachers <dir>` — every teacher file in a directory,
paired against every other, in the rank space the shipped vote actually averages in.

THE PROBLEM IT SOLVES. Thirteen foreign checkpoints are servable and ten of them are the same backbone at
the same resolution, differing only in the corpus they were fitted on. Their QUALITY cannot be measured
here at all: every one was fitted on all 4,406 studies, so a holdout AUC over our corpus is memorised and
the board is the only honest ruler for it — at one submission slot each, with five slots a day.

Their DIVERSITY has no such problem. Distance is a property of the predictions, so it needs neither
labels nor a slot, and diversity is the term the ensemble bonus is paid for. A blend of ten members that
order the studies identically is a blend of one.

WHY IT IS `RankAverage.agreement` AND NOT A CORRELATION WRITTEN HERE. That is the function the shipped
vote uses, so the number is computed in the space the decision is made in. A second implementation would
be a second thing to keep true, and the one it disagreed with would be the one on the board.

WHAT IT DOES NOT SAY. Nothing about quality, and nothing about how many board ticks a unit of distance
buys — `member_distance` carries that warning for our own roster and it is not weaker here. Two members
far apart are a candidate PAIR, not a better one; a distant member that is simply worse still dilutes.
This ranks the pairings a slot could be spent on; the slot still decides.
"""

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float

from core import cli
from kneemri.inference.vote import RankAverage
from kneemri.teacher_file import TeacherFile

logger = logging.getLogger(__name__)

TEACHERS = "*.npz"
"""Teacher files are found by extension: one per member, named for the checkpoint that wrote it."""


@dataclass(frozen=True)
class TeacherReadings:
    """One member's scores over our corpus, carrying the studies they belong to."""

    name: str
    identifiers: tuple[str, ...]
    scores: Float[np.ndarray, "studies targets"]

    @classmethod
    def read(cls, path: Path) -> "TeacherReadings":
        """Load a teacher file under the two names its writer stored it by."""
        stored = np.load(path, allow_pickle=True)
        return cls(
            name=path.stem,
            identifiers=tuple(str(study) for study in stored[TeacherFile.IDENTIFIERS]),
            scores=stored[TeacherFile.SCORES],
        )

    def aligned(self, studies: tuple[str, ...]) -> Float[np.ndarray, "studies targets"]:
        """This member's rows for the given studies, in the given order.

        Reordered rather than assumed to match: a teacher pass is run as several strided processes and
        joined, so two members' files agree on their CONTENTS and need not agree on their row order.
        Correlating them positionally would read a permutation as disagreement.
        """
        rows = {study: index for index, study in enumerate(self.identifiers)}
        return self.scores[[rows[study] for study in studies]]


@dataclass(frozen=True)
class TeacherDistance:
    """Every pairing among a set of members, in the vote's own rank space."""

    members: tuple[TeacherReadings, ...]

    @classmethod
    def loaded(cls, root: Path) -> "TeacherDistance":
        """Read every teacher file under a directory, in sorted order so two runs' reports diff cleanly."""
        return cls(tuple(TeacherReadings.read(path) for path in sorted(root.glob(TEACHERS))))

    def shared(self) -> tuple[str, ...]:
        """The studies every member scored, sorted.

        An INTERSECTION because members are passed over the corpus at different times and one may have
        been capped for a smoke. Comparing them over a union would compare a member against rows it never
        produced, and comparing over one member's set would silently make that member the reference.
        """
        common = set(self.members[0].identifiers).intersection(*(set(one.identifiers) for one in self.members))
        return tuple(sorted(common))

    def pairs(self) -> list[tuple[float, str, str]]:
        """Every unordered pairing and its agreement, most DISTANT first."""
        studies = self.shared()
        aligned = {one.name: one.aligned(studies) for one in self.members}
        agreements = [
            (RankAverage.agreement(aligned[left.name], aligned[right.name]), left.name, right.name)
            for left, right in itertools.combinations(self.members, 2)
        ]
        return sorted(agreements)

    def nearest(self) -> list[tuple[float, str]]:
        """Each member's closest neighbour — the twin detector, most ISOLATED member first.

        The same summary `member_distance` prints for our roster, and for the same reason: a seat is lost
        to one redundant neighbour rather than to an average, so a member can sit far from the set's
        centre while being a copy of one other member in it.
        """
        closest = {one.name: 0.0 for one in self.members}
        for agreement, left, right in self.pairs():
            closest[left] = max(closest[left], agreement)
            closest[right] = max(closest[right], agreement)
        return sorted((value, name) for name, value in closest.items())

    def report(self) -> list[str]:
        """The isolation ranking, then every pairing — the two questions a foreign blend is chosen on."""
        pairs = self.pairs()
        return [
            f"{len(self.members)} members over {len(self.shared())} shared studies",
            "",
            "most isolated first (nearest neighbour's agreement):",
            *(f"  {agreement:.4f}  {name}" for agreement, name in self.nearest()),
            "",
            "most distant pairings first:",
            *(f"  {agreement:.4f}  {left}  x  {right}" for agreement, left, right in pairs),
        ]


def main() -> None:
    """Report how far a directory of foreign members sits from itself."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--teachers", type=Path, required=True, help="directory of teacher files")
    arguments = command.parsed()

    command.report(TeacherDistance.loaded(Path(arguments.teachers)).report())


if __name__ == "__main__":
    main()
