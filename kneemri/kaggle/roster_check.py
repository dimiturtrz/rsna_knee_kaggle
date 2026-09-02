"""Is every scoreable member either packaged or explicitly declined? `python -m kneemri.kaggle.roster_check`.

`comp/members.roster` is the package's ONLY definition: a member absent from it cannot be shipped no
matter what any other file says. Nothing writes it. A run that finishes writes a checkpoint, a holdout
record and usually a ledger section, and then the roster is edited BY HAND — so every member in it is
there because somebody remembered.

WHAT THAT COST, measured 2026-08-29 while building a sixteen-seat entry: five of its sixteen best seats
were not in the roster, `labelwise_dense_s2026` among them — the third-best solo checkpoint the project
owns, and the only member pooling slices labelwise. Five of the sixteen best available seats were
silently unshippable, and nothing anywhere said so.

THE RULE THIS ENFORCES, and it is deliberately a two-sided one. Every checkpoint under the processed root
that has a holdout record is either NAMED in the roster or NAMED in `comp/members.declines` with a
reason. Anything in neither is a finding. A one-sided rule ("everything scoreable must ship") would be
wrong — twelve checkpoints are smoke runs or were scored under an older label table, and they genuinely
should not ship. What was missing is not a policy that everything ships; it is a place to write down that
something does not, so the set difference can be taken by a machine instead of by memory.

A DECLINE WITHOUT PROSE IS ALSO A FINDING. A bare name in the declines file is memory again, one file
over: it records that somebody once decided, not what they decided or why, and the next reader cannot
tell a smoke run from a member that was overlooked twice.
"""

import fnmatch
import logging
from dataclasses import dataclass
from pathlib import Path

from core import cli
from kneemri.kaggle.package_members import MemberRoster
from kneemri.runs.holdout_record import HoldoutRecord

logger = logging.getLogger(__name__)

ROSTER = Path("comp/members.roster")
DECLINES = Path("comp/members.declines")
CHECKPOINT_SUFFIX = ".pt"
CHECKPOINT_GLOB = f"*{CHECKPOINT_SUFFIX}"
COMMENT = "#"


@dataclass(frozen=True)
class MemberDeclines:
    """The committed statement of which scoreable members deliberately do NOT ship, and why each.

    Beside `RosterDrift` rather than beside `MemberRoster`, because the roster is read by the thing that
    PACKAGES and this is read only by the thing that CHECKS: a decline never affects what ships, it
    affects whether an absence is a finding.
    """

    text: str

    @classmethod
    def read(cls, path: Path) -> "MemberDeclines":
        return cls(path.read_text(encoding="utf-8") if path.exists() else "")

    def reasons(self) -> dict[str, str]:
        """Each declined member's name, mapped to the prose that says why it does not ship.

        Keyed WITHOUT the `.pt` suffix, so the file may be written in the roster's spelling — the two
        live side by side and a member named one way in one and another way in the other would read as
        undocumented and declined at once.
        """
        declined: dict[str, str] = {}
        latest = ""
        for line in self.text.splitlines():
            name, separator, reason = line.partition(COMMENT)
            subject = name.strip()
            if not subject:
                continue
            # A CONTINUATION, not an entry. A decline is an argument, and an argument runs past eighty
            # columns — so the arguments in this file are written as wrapped paragraphs. Every wrapped
            # line carries no separator, which read as a fresh bare name: eight of the nine findings in
            # this check's 2026-09-02 run were sentences out of the middle of well-written declines,
            # and a genuinely bare name would have been indistinguishable among them. A name is one
            # token; prose has spaces. That is the whole discriminator, and it keeps the bare-name
            # finding alive rather than trading one hole for another.
            if not separator and " " in subject:
                if latest:
                    declined[latest] = f"{declined[latest]} {subject}".strip()
                continue
            latest = subject.removesuffix(CHECKPOINT_SUFFIX)
            declined[latest] = reason.strip()
        return declined


class RosterDrift:
    """Every way the roster has stopped describing the members on disk."""

    def __init__(self, processed: Path, roster: Path = ROSTER, declines: Path = DECLINES) -> None:
        self.processed = processed
        self.roster = roster
        self.declines = declines

    def checkpoints(self) -> list[str]:
        """Every checkpoint file under the processed root, named without its suffix."""
        return sorted(path.stem for path in self.processed.glob(CHECKPOINT_GLOB))

    def scoreable(self) -> list[str]:
        """The checkpoints that could take a seat: the ones that left holdout predictions to price."""
        return [name for name in self.checkpoints() if HoldoutRecord(self.processed / f"{name}.pt").exists()]

    def patterns(self) -> list[str]:
        return MemberRoster.read(self.roster).patterns()

    def packaged(self) -> set[str]:
        """The scoreable members the roster's globs actually reach."""
        names = self.scoreable()
        return {
            name
            for pattern in self.patterns()
            for name in names
            if fnmatch.fnmatch(f"{name}.pt", pattern) or name == pattern.removesuffix(".pt")
        }

    def undocumented(self) -> list[str]:
        """Scoreable members in neither file — the defect this exists to make impossible."""
        declined = MemberDeclines.read(self.declines).reasons()
        return [name for name in self.scoreable() if name not in self.packaged() and name not in declined]

    def reasonless(self) -> list[str]:
        """Declined members whose decline is a bare name, which records only that somebody once decided."""
        return sorted(name for name, reason in MemberDeclines.read(self.declines).reasons().items() if not reason)

    def contradictory(self) -> list[str]:
        """Members the roster ships and the declines file says do not ship. The roster wins; say so."""
        declined = MemberDeclines.read(self.declines).reasons()
        return sorted(self.packaged() & set(declined))

    def phantom(self) -> list[str]:
        """Roster lines matching no checkpoint at all — `package_members` refuses these, later and louder."""
        present = {f"{name}.pt" for name in self.checkpoints()}
        return [pattern for pattern in self.patterns() if not any(fnmatch.fnmatch(name, pattern) for name in present)]

    def complaints(self) -> list[str]:
        """Each drift, named with the action that clears it. Empty means the roster still describes disk."""
        found = [
            f"{name} is scoreable and in neither file — add a roster line or a decline with its reason"
            for name in self.undocumented()
        ]
        found += [
            f"{name} is declined with no reason — say what it is, not that it was decided" for name in self.reasonless()
        ]
        found += [
            f"{name} is both packaged and declined — the roster ships it, so drop the decline"
            for name in self.contradictory()
        ]
        found += [
            f"roster names {pattern!r}, which matches no checkpoint under {self.processed}"
            for pattern in self.phantom()
        ]
        return found

    def report(self) -> list[str]:
        """The survey, whether or not anything drifted."""
        scoreable = self.scoreable()
        declined = MemberDeclines.read(self.declines).reasons()
        summary = (
            f"{len(scoreable)} scoreable of {len(self.checkpoints())} checkpoints: "
            f"{len(self.packaged())} packaged, {len(declined)} declined, {len(self.undocumented())} undocumented"
        )
        return [*self.complaints(), summary]


def main() -> None:
    """Assert the roster is the scoreable pool minus a documented decline."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--roster", type=Path, default=ROSTER, help="what ships")
    command.parser.add_argument("--declines", type=Path, default=DECLINES, help="what deliberately does not")
    arguments = command.parsed()

    drift = RosterDrift(command.rooted(arguments).models(), arguments.roster, arguments.declines)
    command.report(drift.report())
    if drift.complaints():
        message = f"{len(drift.complaints())} member(s) the roster no longer describes"
        raise SystemExit(message)


if __name__ == "__main__":
    main()
