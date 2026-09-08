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

A ROSTER LINE IS A CLAIM THAT THE MEMBER CAN STILL BE READ, and until 2026-09-03 nothing checked it.
`adjacent224_s2026` sits in the roster with a holdout record, a checkpoint and every gate in this repo
green, and its pixels are gone: `images_224px_12sl_adjacent.npy` is absent while `mask_224px_12sl_adjacent.npy`
survives. Both arrays are asserted rather than one, because that is the shape the defect actually took —
a check reading only the mask would have called this member healthy (knee-dhpo9).

REPORTED, NOT REFUSED, and the first run says why. Nine of 102 packaged members name a cache that is no
longer on disk, over eight distinct arrays, against 24 that survive: these are reclaimed decodes, each
one tens of gigabytes, and deleting them was housekeeping rather than an accident. Such a member can
still SHIP — the kernel decodes Kaggle's studies at serve time and never opens our cache — and its
recorded holdout number is still quotable. What it can never do again is be re-scored, re-derived,
witnessed or curve-read offline, because the pixels it was fitted on are gone. That is worth a named
line every run, and it is not worth a refusal: a gate that fails nine-strong on every invocation is a
gate people learn to skip.
"""

import fnmatch
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from core import cli
from kneemri.inference.members import Member
from kneemri.kaggle.package_members import MemberRoster
from kneemri.runs.holdout_record import SUFFIX as HOLDOUT_SUFFIX

logger = logging.getLogger(__name__)

ROSTER = Path("comp/members.roster")
ROSTERS = (ROSTER, Path("comp/members2.roster"))
"""Every shelf a member may be named on, in mount order.

There are two because the shipped package is ~10 GB: adding one member to it means re-pushing all of
it, so members trained after the last big push live in a SECOND dataset mounted beside the first
(`MemberPackage.spanning`). That is a second shelf, never a second standard — this check reads both and
holds every member on either to the same rule, so moving a line between them changes what gets pushed
and nothing about what is documented. A name on BOTH shelves is a duplicate the mount refuses, which is
why `packaged` is a set and the duplicate is caught there rather than being silently merged."""
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

    def __init__(
        self,
        processed: Path,
        rosters: Sequence[Path] = ROSTERS,
        declines: Path = DECLINES,
        pixels: Path | None = None,
    ) -> None:
        self.processed = processed
        self.rosters = [roster for roster in rosters if roster.is_file()]
        self.declines = declines
        self.pixels = processed if pixels is None else pixels
        """Where the decoded arrays live. Defaults to the checkpoint directory so every existing caller
        keeps working; `main` passes the processed root, which is where a cache is actually written."""

    def checkpoints(self) -> list[str]:
        """Every checkpoint file under the processed root, named without its suffix."""
        return sorted(path.stem for path in self.processed.glob(CHECKPOINT_GLOB))

    def scoreable(self) -> list[str]:
        """Every member that could take a seat: the ones that left holdout predictions to price.

        Enumerated from the HOLDOUT RECORDS, not from the checkpoints. Until 2026-09-08 this walked
        `*.pt` and asked each one for its record, which meant a member with predictions and no local
        checkpoint was never enumerated at all — and that is every FOREIGN member, someone else's
        weights served through `foreign_member` and scored on our fold. So the one gate that guards what
        can be shipped was blind to an entire class of shippable member: the five `raptor_ft_hisrecipe_distill*`
        sat in neither the roster nor the declines for hours with this check green, and `raptor_v4_swa`
        — a seat in the SHIPPED champion — is invisible to it to this day. The record is what makes a
        member scoreable; the checkpoint is only the usual way one arrives (knee-s39kk).
        """
        return sorted(path.name[: -len(HOLDOUT_SUFFIX)] for path in self.processed.glob(f"*{HOLDOUT_SUFFIX}"))

    def holdout_only(self) -> list[str]:
        """Scoreable members with predictions and no checkpoint of ours — the class this check used to miss.

        Named in the survey every run rather than complained about, because being holdout-only is not a
        defect: it is what a foreign member IS. What was the defect is that such a member could be in
        neither file and nothing said so, and `scoreable` now closes that. A roster line for one of these
        is still a `phantom`, correctly — `package_members` ships checkpoints, so the shelf cannot reach
        it and the honest home for it is a decline that says how it reaches Kaggle instead.
        """
        return [name for name in self.scoreable() if not (self.processed / f"{name}{CHECKPOINT_SUFFIX}").exists()]

    def patterns(self) -> list[str]:
        return [pattern for roster in self.rosters for pattern in MemberRoster.read(roster).patterns()]

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

    def unreadable(self) -> list[str]:
        """Each packaged member whose own cache is not on disk, named with the FIRST array that is gone.

        The member is asked which pixels it read rather than the roster being asked what it expects —
        `Member.cache` is the one place that knows the axis list, and a second spelling of it here would
        be the mismatch this check exists to catch, arriving from the checker instead of the reader.

        Both arrays are stated, and the path is printed rather than the member: `adjacent224_s2026` lost
        its images and kept its mask, so a check that stopped at either one alone would have passed it.
        """
        gone: list[str] = []
        for name in sorted(self.packaged()):
            checkpoint = self.processed / f"{name}{CHECKPOINT_SUFFIX}"
            # A HOLDOUT-ONLY MEMBER HAS NOTHING TO ASK. A foreign member is scoreable through its record
            # and carries no checkpoint of ours, so there is no stamped config to name a cache with — and
            # it never read our cache in the first place, having been served through its own decode.
            if not checkpoint.exists():
                continue
            member = Member.from_checkpoint(checkpoint)
            if not member.reads_pixels:
                continue
            cache = member.cache(self.pixels)
            gone += [f"{name} {path}" for path in (cache.images_path, cache.mask_path) if not path.exists()]
        return gone

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
        unreadable = [
            f"{missing} is gone, so that member can never be re-scored, re-derived or witnessed offline"
            for missing in self.unreadable()
        ]
        scoreable = self.scoreable()
        declined = MemberDeclines.read(self.declines).reasons()
        summary = (
            f"{len(scoreable)} scoreable (of which {len(self.holdout_only())} carry no local checkpoint) "
            f"against {len(self.checkpoints())} checkpoints: "
            f"{len(self.packaged())} packaged, {len(declined)} declined, {len(self.undocumented())} undocumented"
        )
        return [*self.complaints(), *unreadable, summary]


def main() -> None:
    """Assert the roster is the scoreable pool minus a documented decline."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--roster",
        type=Path,
        action="append",
        help="a shelf of what ships; repeatable, and every shelf given replaces the default pair "
        "(comp/members.roster plus the second package's comp/members2.roster)",
    )
    command.parser.add_argument("--declines", type=Path, default=DECLINES, help="what deliberately does not")
    arguments = command.parsed()

    root = command.rooted(arguments)
    drift = RosterDrift(root.models(), arguments.roster or ROSTERS, arguments.declines, root.processed())
    command.report(drift.report())
    if drift.complaints():
        message = f"{len(drift.complaints())} member(s) the roster no longer describes"
        raise SystemExit(message)


if __name__ == "__main__":
    main()
