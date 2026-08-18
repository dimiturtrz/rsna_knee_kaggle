"""Can the best entry this project has still be rebuilt? `python -m kneemri.kaggle.champion_check`.

THE SEVENTH CHECK, and the first that guards a RESULT rather than a shipment. The other six ask whether
what is about to be sent is what the repo means to send. This one asks the opposite question, about
something already sent: the entry that scored 0.925 on the hidden set is the only proof this project has
that any of its choices were right, and nothing in the repo held it together. Its members live in a
roster that is edited by hand every time a new one is trained, its serving config lives in a kernel
script that gets copied and rewritten for each arm, and the number itself lived in prose.

Losing it would not announce itself. A roster line dropped while adding a member, or a `chosen` tuple
edited in place instead of copied into a new kernel, leaves everything green: the package builds, the
kit verifies, the kernel runs, and the entry that proved the recipe is simply no longer expressible.

So the champion is pinned in `comp/champion.json` and this compares that record against the two places
it actually has to hold — the roster that decides what is packaged, and the kernel that names what is
loaded. It reads the kernel with `ast` rather than a regex, because the thing being checked is what
Python would bind, and a comment holding an old member name is not a member.

WHAT IT DOES NOT DO IS FORBID CHANGE. Every entry after the champion is a deliberate deviation from it —
that is how the axes get priced — so a differing kernel is not a finding. Only the champion's OWN
reproduction path is asserted.
"""

import argparse
import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from kneemri.kaggle.package_members import MemberRoster

logger = logging.getLogger(__name__)

RECORD = Path("comp/champion.json")
"""The pinned best entry. Prose lives in `comp/SUBMISSIONS.md`; this is the half something can read."""

KERNELS = Path("comp/kernels")
"""Where a kernel's script lives, one directory per entry, named after the kernel."""

CHOSEN = "chosen"
"""The name a kernel binds its member tuple to, which is the only line of it this check reads."""


@dataclass(frozen=True)
class Champion:
    """The best scored entry, and everything needed to build it again."""

    entry: str
    kernel: str
    board: float
    members: tuple[str, ...]
    kit: str

    @classmethod
    def recorded(cls, path: Path = RECORD) -> "Champion":
        """The pinned champion, read from the record rather than from whatever scored most recently."""
        held = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            entry=str(held["entry"]),
            kernel=str(held["kernel"]),
            board=float(str(held["board"])),
            members=tuple(str(name) for name in held["members"]),
            kit=str(held["kit"]),
        )

    def named_by_kernel(self, kernels: Path = KERNELS) -> tuple[str, ...]:
        """The members the champion's kernel binds to `chosen`, as Python would bind them.

        Parsed rather than matched: a regex reads a member name out of a comment or a docstring, and
        this check exists precisely for the case where the code and the prose around it disagree.
        """
        script = kernels / self.kernel / f"{self.kernel}.py"
        tree = ast.parse(script.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Tuple):
                continue
            if any(isinstance(target, ast.Name) and target.id == CHOSEN for target in node.targets):
                return tuple(
                    element.value
                    for element in node.value.elts
                    if isinstance(element, ast.Constant) and isinstance(element.value, str)
                )
        return ()

    def drift(self, roster: Path, kernels: Path = KERNELS) -> list[str]:
        """Every way the champion can no longer be rebuilt, named. Empty means it still can."""
        named = self.named_by_kernel(kernels)
        packaged = {pattern.removesuffix(".pt") for pattern in MemberRoster.read(roster).patterns()}
        found = []
        found += [
            f"{self.kernel} no longer names {name!r}, which the {self.board} entry loaded"
            for name in self.members
            if name not in named
        ]
        found += [
            f"the roster no longer packages {name!r}, so the {self.board} entry cannot be mounted"
            for name in self.members
            if name not in packaged
        ]
        return found


def main() -> None:
    """Assert the champion is still buildable: `python -m kneemri.kaggle.champion_check`."""
    parser = argparse.ArgumentParser(description=Champion.__doc__)
    parser.add_argument("--record", type=Path, default=RECORD, help="the pinned champion")
    parser.add_argument("--roster", type=Path, default=Path("comp/members.roster"), help="what ships")
    parser.add_argument("--kernels", type=Path, default=KERNELS, help="where kernel scripts live")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    champion = Champion.recorded(arguments.record)
    drift = champion.drift(arguments.roster, arguments.kernels)
    for complaint in drift:
        logger.error("%s", complaint)
    if drift:
        message = f"{champion.entry} ({champion.board}) can no longer be rebuilt"
        raise SystemExit(message)
    logger.info(
        "%s (%s, kit %s) is still buildable: %d members named and packaged",
        champion.entry,
        champion.board,
        champion.kit,
        len(champion.members),
    )


if __name__ == "__main__":
    main()
