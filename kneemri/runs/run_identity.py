"""Which run wrote a record: an id minted once, the code it ran, and the runs it started from.

A run used to be named by its `--out` alone, so everything that asked "which run is this" recovered the
answer from a filename, a rotation index and a recipe diff — and got it wrong whenever a re-run reused
the name. An id minted at the start and stamped into the record AND the checkpoint makes the question a
lookup.
"""

import os
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import torch

from kneemri.runs.run_record import RunRecord

GIT_ENV = "KNEEMRI_GIT_SHA"
"""Set by `experiments/lib.sh:run_snapshot`: an arm runs from a copy of the package with no `.git` beside it."""

UNKNOWN = "unknown"
WEIGHTS = (".pt", ".pth")
OWN_OUTPUT = "out"
REPOSITORY = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RunIdentity:
    """One run's identity, as a stamp both the record and the checkpoint can carry under `weights_only`."""

    id: str
    git: str
    parents: dict[str, dict[str, str | None]] = field(default_factory=dict)
    """Recipe key → the weights file it names and that file's own run id, when the file carries one."""

    @classmethod
    def begun(cls, recipe: dict[str, str], destination: Path, *, resuming: bool) -> "RunIdentity":
        """Mint this run's identity — or, when RESUMING, keep the one the interrupted run was given.

        Read before `RunMetrics.begin` rotates the record: a resume is the same run continuing, and a
        fresh id would split one trajectory into two runs that each look truncated.
        """
        earlier = RunRecord(destination).run() if resuming else None
        minted = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{secrets.token_hex(4)}"
        return cls(
            id=str(earlier["id"]) if earlier and "id" in earlier else minted,
            git=cls.commit(),
            parents={
                key: {"path": value, "run": cls.of(Path(value))}
                for key, value in recipe.items()
                if key != OWN_OUTPUT and value.endswith(WEIGHTS)
            },
        )

    @staticmethod
    def commit(repository: Path = REPOSITORY) -> str:
        """The commit this code came from: the arm's exported sha, else the checkout's HEAD, else unknown."""
        if exported := os.environ.get(GIT_ENV):
            return exported
        head = repository / ".git" / "HEAD"
        if not head.exists():
            return UNKNOWN
        pointer = head.read_text(encoding="utf-8").strip()
        if not pointer.startswith("ref: "):
            return pointer
        branch = repository / ".git" / pointer.removeprefix("ref: ")
        if branch.exists():
            return branch.read_text(encoding="utf-8").strip()
        packed = repository / ".git" / "packed-refs"
        lines: list[str] = packed.read_text(encoding="utf-8").splitlines() if packed.exists() else []
        named = pointer.removeprefix("ref: ")
        return next((line.split()[0] for line in lines if line.endswith(f" {named}")), UNKNOWN)

    @staticmethod
    def of(weights: Path) -> str | None:
        """The run id a weights file's record carries — else the one stamped inside the file — else None.

        The file's own stamp is what makes a COPY resolve: `package_members` copies the checkpoint and
        leaves the record behind, and the stamp travels with the weights. Memory-mapped, so asking a
        multi-gigabyte checkpoint for its id reads the header, not the tensors.
        """
        stamped = RunRecord(weights).run()
        if not stamped and weights.suffix in WEIGHTS and weights.exists():
            saved = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
            stamped = saved.get("run") if isinstance(saved, dict) else None
        return str(stamped["id"]) if isinstance(stamped, dict) and "id" in stamped else None

    def stamp(self) -> dict[str, object]:
        """The identity as plain data, so a `weights_only` load of the checkpoint can read it back."""
        return {"id": self.id, "git": self.git, "parents": self.parents}
