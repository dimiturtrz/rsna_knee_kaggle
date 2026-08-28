"""What a directory of foreign checkpoints actually CONTAINS, read from the checkpoints themselves.

`python -m kneemri.evaluation.raptor_inventory --weights /d/data/external/raptor` — one line per
checkpoint: the backbone it names, the resolution it was fitted at, whether its label order is ours, and
whether it carries the `model` key `RaptorMember.loaded` requires.

WHY IT EXISTS. Fourteen checkpoints were downloaded from one author's Kaggle account across nine
datasets whose TITLES describe corpora ("Knee MRI - Wide FOV Corpus") and whose FILE LISTS are weights.
One of them held five days unnoticed for exactly that reason. Nothing recorded which architecture each
one is, and the seat question — a blend pays for DIVERSITY, so two checkpoints of the same backbone at
the same resolution are a weaker pair than the file count suggests — cannot be asked without that.

IT READS, IT DOES NOT SERVE. No decode, no card, no forward pass: `torch.load` onto the CPU and four
dictionary lookups. That is the point — a serving probe costs a study's decode per checkpoint and this
costs a file read, so the inventory can be re-run whenever the directory grows.

`loadable` is the narrow claim that the file is a raptor checkpoint at all. `rsna-knee-model/model.pt`
from the same account is not: 17 MB, no `model` key, and it fails here in one line instead of inside a
teacher pass forty minutes into a card.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.raptor_model import DEFAULT_ARCHITECTURE, DEFAULT_RESOLUTION

logger = logging.getLogger(__name__)

WEIGHTS = "*.pt"
"""Checkpoints are found by extension, recursively, because one dataset per directory is the download's
shape and the directory names are the author's dataset titles rather than anything we chose."""

MODEL = "model"
"""The state-dict key `RaptorMember.loaded` indexes. Its absence is the whole of `loadable`."""


@dataclass(frozen=True)
class CheckpointEntry:
    """One checkpoint's self-description: what it says it is, and whether it is servable at all."""

    path: Path
    architecture: str
    resolution: int
    labels: tuple[str, ...]
    loadable: bool

    @classmethod
    def read(cls, path: Path) -> "CheckpointEntry":
        """Load the file onto the CPU and take what it says about itself, defaulting as the loader does.

        The defaults are `raptor_model`'s own, imported rather than restated: a checkpoint with no `arch`
        key IS the published arm, and an inventory that guessed differently from the loader would report
        a fleet the serving path cannot build.
        """
        state = torch.load(path, map_location="cpu", weights_only=False)
        return cls(
            path=path,
            architecture=str(state.get("arch", DEFAULT_ARCHITECTURE)),
            resolution=int(state.get("res", DEFAULT_RESOLUTION)),
            labels=tuple(str(name) for name in state.get("lab", TARGETS)),
            loadable=MODEL in state,
        )

    @property
    def family(self) -> str:
        """The pair a blend actually cares about — same backbone at the same resolution is one lens."""
        return f"{self.architecture}@{self.resolution}"

    def line(self, root: Path) -> str:
        """One report row, the path shown relative to the directory that was scanned."""
        state = "ok" if self.loadable else "NOT-A-RAPTOR-CHECKPOINT"
        ours = "ours" if self.labels == TARGETS else "FOREIGN-LABEL-ORDER"
        return f"{state:<24} {self.family:<52} {ours:<20} {self.path.relative_to(root)}"


@dataclass(frozen=True)
class Inventory:
    """Every checkpoint under a directory, and the families they fall into."""

    entries: tuple[CheckpointEntry, ...]

    @classmethod
    def scanned(cls, root: Path) -> "Inventory":
        """Read every `.pt` under the root, in sorted order so two runs' reports diff cleanly."""
        return cls(tuple(CheckpointEntry.read(path) for path in sorted(root.rglob(WEIGHTS))))

    def families(self) -> dict[str, int]:
        """How many loadable checkpoints share each backbone-and-resolution.

        Counted over the LOADABLE ones only: a file that is not a raptor checkpoint has no family here,
        and letting it default into the published arm's would overstate that family by one.
        """
        counts: dict[str, int] = {}
        for entry in self.entries:
            if entry.loadable:
                counts[entry.family] = counts.get(entry.family, 0) + 1
        return counts

    def report(self, root: Path) -> list[str]:
        """The per-checkpoint rows, then the family census — the two questions the fleet is read for."""
        rows = [entry.line(root) for entry in self.entries]
        census = [f"{count:>3}  {family}" for family, count in sorted(self.families().items())]
        loadable = sum(1 for entry in self.entries if entry.loadable)
        return [
            *rows,
            "",
            f"{loadable} of {len(self.entries)} loadable, in {len(self.families())} families:",
            *census,
        ]


def main() -> None:
    """Report what every checkpoint under a directory says it is."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--weights", type=Path, required=True, help="directory of checkpoints")
    arguments = command.parsed()

    root = Path(arguments.weights)
    command.report(Inventory.scanned(root).report(root))


if __name__ == "__main__":
    main()
