"""What an arm's header DECLARES, read from the shell that printed the arm's name.

An arm's header is the only place it states anything about itself: the bead that priced it, the run its
number will be read against, the flags it moves on purpose, and the board reading it waits on. Those four
declarations share one hard part — finding the shell text at all — and it is the part that was wrong.

AN ARM IS NOT ONE MEMBER. Every rule here used to resolve a header as `experiments/arms/<name>.sh`, and
`final_fit_six.sh` prints six member names, none of which is its filename. So the arms that spend the most
card were exactly the ones whose declarations nothing read, including the gate whose five unread hours
(2026-08-23, knee-1e3m) are the reason this module exists.
"""

import re
from collections.abc import Iterable
from pathlib import Path

from kneemri.experiments.arm_gate import BOARD, ArmGate

BEAD_LINE = re.compile(r"^#\s*bead:\s*(knee-[a-z0-9]+)\s*$", re.MULTILINE)
"""How an arm names the issue that priced it. One line, anywhere in the shell's header."""

CONTROL_LINE = re.compile(r"^#\s*control\s+([A-Za-z0-9_]+)\s*=", re.MULTILINE)
"""How an arm names the run its number will be read against. `# control  band_dense = 0.8578`.

The name must be followed by `=` because the word also opens prose in several headers ("# control is
the same recipe..."), and a sentence is not a declaration.
"""

DELIBERATE_LINE = re.compile(r"^#\s*deliberate:\s*(--[a-z0-9-]+)", re.MULTILINE)
"""How an arm says IN WRITING which flag it moves away from its control. One flag per line.

An arm exists to move something, so a diff against the control is not by itself a defect — an
UNDECLARED diff is. This is the declaration, and it costs the author one line per variable, which is
also the moment they notice they are moving two."""

ARMS = Path("experiments/arms")


class ArmHeader:
    """Each arm name's shell text, and the declarations that text carries."""

    def __init__(self, arms: Path = ARMS) -> None:
        self.arms = arms
        self._sources: dict[str, Path] = {}
        """Which shell printed each invocation name, filled by `record` as the shells are run.

        Kept here rather than returned, because the invocation mapping is name -> flags in eight places
        and widening it would rewrite all of them to carry a path most of them do not read.
        """

    def record(self, name: str, shell: Path) -> None:
        """Remember that `shell` printed this invocation name — the only link a multi-member arm has."""
        self._sources[name] = shell

    def text(self, name: str) -> str:
        """The shell text of the arm printing this name, or empty where no shell of the tree printed it.

        The file named after the invocation FIRST, because that is the common arm and it holds whether
        or not it has been run through `invocations` yet; the shell that actually printed the name
        second, which is how a six-member arm's header is reached at all. Empty for the synthesised
        switch recipes, which have no shell.
        """
        path = self.arms / f"{name}.sh"
        source = self._sources.get(name)
        if not path.exists() and source is not None and source.exists():
            path = source
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def beads(self, names: Iterable[str]) -> dict[str, str | None]:
        """The bead each arm names in a `# bead:` line, or None where its shell names none.

        The cheapest binding from an arm to the issue that PRICED it. There is no other link: an arm is a
        shell, a bead is a row in a database, and nothing in the tree connects them, which is how a run
        gets launched against a bead whose price was never computed.

        Arms with no shell — the switch recipes `SwitchCoverage` synthesises — are absent rather than
        None, because a recipe that exists to exercise one flag at sixty studies is not an experiment and
        has nothing to price.
        """
        headers = ((name, self.text(name)) for name in names)
        found = ((name, BEAD_LINE.search(header)) for name, header in headers if header)
        return {name: match.group(1) if match else None for name, match in found}

    def controls(self, names: Iterable[str]) -> dict[str, str]:
        """The control each arm declares in a `# control  <name> = <score>` line, by arm name.

        Arms declaring none are absent rather than None: seventy arms predate the convention and most
        of them name a CHECKPOINT (`wsfold0_e6`) rather than another arm, which is a control this tool
        cannot re-derive flags for. Silence here is a missing comparison, not a wrong one.
        """
        found = ((name, CONTROL_LINE.search(self.text(name))) for name in names)
        return {name: match.group(1) for name, match in found if match}

    def deliberate(self, name: str) -> set[str]:
        """The flags this arm declares it moves on purpose, so a diff on them is the experiment."""
        return set(DELIBERATE_LINE.findall(self.text(name)))

    def gated(self, names: Iterable[str], board: Path = BOARD) -> dict[str, str]:
        """Each arm whose declared `# gate:` the board does NOT currently satisfy, and why.

        THE HOLE THIS CLOSES cost five hours of card on 2026-08-23. `final_fit_six.sh` states its own
        stop condition in its header, `kneemri-finalfit` read 0.934 against a 0.936 floor, and the arm
        ran anyway — four members to completion and a fifth to epoch 10 of 12 — because a sentence in a
        bash comment has no consumer. Arms that declare no gate are absent rather than empty: an
        ungated arm is the normal one, and a check that names seventy of them names nothing.
        """
        gates = ((name, ArmGate.declared(self.text(name))) for name in names)
        unmet = ((name, gate.unmet(board)) for name, gate in gates if gate is not None)
        return {name: why for name, why in unmet if why}
