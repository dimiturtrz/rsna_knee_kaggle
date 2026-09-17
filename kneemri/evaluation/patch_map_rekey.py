"""Re-key a patch sweep that recorded SEATS onto the study names: `python -m kneemri.evaluation.patch_map_rekey`.

`patch_maps` wrote `studies` as positional indices into whichever cache the sweep loaded, while
`PatchErasing.loaded` joins a trainer's corpus by StudyInstanceUID. The sweep and the trainer need not
share a cache — the sweep runs on the member that produced the maps and the arm trains at its own
resolution — so a seat means nothing outside the array it was measured in. Fixed at the source, but the
sweeps already on disk are hours of card each and hold seats.

WHY THIS IS A REPAIR RATHER THAN A GUESS. A seat is only ambiguous about WHICH cache ordered it, and the
npz carries `truth` for those same seats. Re-key under a candidate identifier list, recompute the labels
that list implies, and the stored truth either reproduces exactly or the list is the wrong one. So the
candidate does not have to be known in advance: it has to be CHECKED, and an unverifiable re-key is
refused rather than written. Two candidates that both reproduce the truth carry the same names by
construction, and the writer asserts that rather than trusting it.

A sweep already keyed on names is refused too — rewriting one costs nothing and proves nothing, and a
command that silently accepts its own output is how a repair gets run twice against a file it did not fix.
"""

import json
import logging
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray

from core import cli
from core.data.labels.targets import Targets

logger = logging.getLogger(__name__)

STUDIES = "studies"
MAPS = "maps"
TRUTH = "truth"
TISSUE = "tissue"
REPORTS = "train.csv"


class PatchMapRekey:
    """One positional sweep, and the identifier lists that might be the order it was measured in."""

    def __init__(self, maps: Path, labels: Path, reports: Path) -> None:
        """`labels` is the soft label table the sweep scored against, `reports` the corpus's own csv."""
        self.maps = maps
        self.labels = labels
        self.reports = reports
        self.held = np.load(maps)

    def positions(self) -> Int[NDArray[np.int64], " s"]:
        """The seats the sweep recorded, refusing a sweep that is already keyed on names."""
        recorded = self.held[STUDIES]
        if not np.issubdtype(recorded.dtype, np.integer):
            raise ValueError(f"{self.maps} is already keyed on names ({recorded.dtype}); nothing to repair")
        return recorded.astype(np.int64)

    def named(self, index: Path) -> list[str]:
        """The seats read as names under this cache's order — unverified on its own."""
        identifiers = json.loads(index.read_text(encoding="utf-8"))
        seats = self.positions()
        if int(seats.max()) >= len(identifiers):
            raise ValueError(f"{index} holds {len(identifiers)} studies; the sweep names seat {int(seats.max())}")
        return [identifiers[seat] for seat in seats]

    def reproduces(self, index: Path) -> bool:
        """Whether the labels this order implies for those seats are the ones the sweep stored.

        The whole verification: the sweep's own `truth` was taken at the same seats, so a candidate order
        that is not the one it was measured in disagrees on the labels of almost every study.
        """
        identifiers: list[str] = json.loads(index.read_text(encoding="utf-8"))
        scores: Float[NDArray[np.float32], "n t"] = Targets.load(self.labels, self.reports, identifiers).scores
        return bool(np.array_equal(scores[self.positions()], self.held[TRUTH]))

    def write(self, index: Path, destination: Path) -> Path:
        """Copy the sweep through with its seats replaced by names, every other array untouched."""
        if not self.reproduces(index):
            raise ValueError(f"{index} does not reproduce the truth stored in {self.maps}; refusing to re-key")
        np.savez_compressed(
            destination,
            maps=self.held[MAPS],
            studies=np.asarray(self.named(index)),
            truth=self.held[TRUTH],
            tissue=self.held[TISSUE],
        )
        return destination


def main() -> None:
    """Re-key a positional patch sweep onto study names, verified against the truth it stored."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--maps", type=Path, required=True, help="the .patches.npz keyed on seats")
    command.parser.add_argument("--labels", type=Path, required=True, help="the soft label table the sweep used")
    command.parser.add_argument("--indexes", type=Path, nargs="+", required=True, help="candidate cache index json")
    command.parser.add_argument("--out", type=Path, required=True, help="destination .patches.npz")
    arguments = command.parsed()

    root = command.rooted(arguments)
    rekey = PatchMapRekey(arguments.maps, arguments.labels, root.raw() / REPORTS)
    matching = [index for index in arguments.indexes if rekey.reproduces(index)]
    if not matching:
        command.parser.error(f"none of the {len(arguments.indexes)} candidate orders reproduce the stored truth")

    names = [rekey.named(index) for index in matching]
    if any(candidate != names[0] for candidate in names):
        command.parser.error(f"{len(matching)} orders reproduce the truth and disagree on the names")

    written = rekey.write(matching[0], arguments.out)
    command.report([f"{matching[0].name} reproduces the truth", f"wrote {written} keyed on {len(names[0])} names"])


if __name__ == "__main__":
    main()
