"""How many decodes would a roster cost, and which members share each? `python -m kneemri.kaggle.roster_reads`.

THE COST OF A KAGGLE ENTRY IS ITS DECODE COUNT, not its member count. Decode is the I/O-bound half on a
four-CPU host, and `Ensemble.reads` groups members that share one — so an entry of thirty members on two
reads is cheaper to serve than an entry of seven on four. That is also close to the efficiency track's
own metric, which scores against serve cost.

WHY IT EXISTS. On 2026-08-29 the question "do these checkpoints share a decode?" could only be answered
by building a throwaway members package and running `dry_run` on it: the key lived inside
`Ensemble.reads`, reachable only through a mounted package. That is a package build and a CPU inference
pass to learn a fact that is entirely in the checkpoints' metadata. It also arrives too late — after
packaging, when the choice of what to package is the thing being decided.

WHAT IT IS NOT. It says nothing about whether a seat is worth taking; `seat_value` prices that. This is
the COST half only, and the two are read together: a candidate that prices inside the noise floor is
declined whatever it costs, and a candidate that prices above it is taken more readily when it adds no
read. Nor does it replace `dry_run` before a slot — a decode count derived from metadata is a claim
about what the kernel will do, and `dry_run` is the kernel doing it.

WHAT SPLITS A READ, in the order it bites: the slot scheme (which series are routed), the slice depth,
the tie rule (which of several equally dense candidates fills a slot), and the whole `Reading` — crop
extent, crop centre, band, spacing. RESOLUTION DOES NOT, and neither pooling does. Those are properties
of the model, not of the read.
"""

import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from core import cli
from core.data.corpus.slot_reader import Reading
from core.denominator import Denominator
from core.paths import DataRoot
from kneemri.inference.members import Member

logger = logging.getLogger(__name__)

CHECKPOINT_SUFFIX = ".pt"
"""What a roster line carries and a `--candidates` path may omit. `comp/members.roster` writes the
suffix; a name typed by hand usually does not, and a tool that accepts only one of the two spellings
teaches its operator to try both."""


@dataclass(frozen=True)
class ReadGroup:
    """One decode, and every member that rides it."""

    members: list[Member]

    @property
    def sizes(self) -> list[int]:
        """The resolutions this ONE read serves — the axis that costs nothing, listed so it is visible."""
        return sorted({member.image_size for member in self.members})

    @property
    def description(self) -> str:
        """What makes this read distinct from the others, in the fields that actually split a decode."""
        first = self.members[0]
        crop = first.crop
        band = first.band.recorded()
        return (
            f"{first.scheme} / {first.slices} slices / band {band[0]}-{band[1]} / "
            f"centre {crop.centre} / crop {crop.mm}mm / tie {first.tie} / spacing {first.spacing}"
        )

    def lines(self) -> list[str]:
        """This read's header and its members, sorted by recorded holdout so the best seat reads first."""
        best = sorted(self.members, key=lambda member: member.holdout, reverse=True)
        rows = [f"  {len(self.members)} members at {self.sizes} — {self.description}"]
        rows.extend(
            f"      {member.identifier}  {member.image_size}px  holdout {member.holdout:.4f}" for member in best
        )
        return rows


@dataclass(frozen=True)
class RosterReads:
    """A set of checkpoints, grouped into the decodes serving them would cost."""

    groups: list[ReadGroup]

    @staticmethod
    def keyed(member: Member) -> tuple[str, int, str, Reading] | None:
        """This member's decode, or `None` when the checkpoint does not describe one.

        IT READS `image_size` TOO, though the size is not part of the key and never will be. A
        checkpoint missing `img` groups perfectly well and then takes the report down when its group
        lists the resolutions it serves — the same defect one field over, discovered exactly that way.
        Answering both here means every member that survives the filter is fully reportable.
        """
        try:
            key, _size = member.decode_key, member.image_size
        except KeyError:
            return None
        return key

    @classmethod
    def over(cls, members: list[Member]) -> "RosterReads":
        """Group by `Member.decode_key` — the SAME property `Ensemble.reads` keys on, never a copy.

        Largest group first, because the interesting question is always which read holds the most seats
        and the answer is meant to be readable without counting.

        NOT EVERY `.pt` UNDER THE PROCESSED ROOT IS A MEMBER. Teachers, distillation targets and partial
        saves live beside them and record no slot list, so the key raises `KeyError` on one of them —
        which took the whole grouping down the first time this was pointed at the directory rather than
        at a roster. A file that cannot say which pixels it read cannot be placed in a decode, and
        naming it per file is the only honest handling: dropping it silently would understate a cost.
        """
        gathered: dict[tuple[str, int, str, Reading], list[Member]] = {}
        for member in members:
            key = cls.keyed(member)
            if key is None:
                logger.warning("%s describes no reading, so it names no decode — not a member", member.identifier)
                continue
            gathered.setdefault(key, []).append(member)
        ordered = sorted(gathered.values(), key=len, reverse=True)
        return cls([ReadGroup(group) for group in ordered])

    @property
    def placed(self) -> list[Member]:
        """Every member that answered a decode — the only set the report may speak about.

        `main` counted resolutions over the members it READ instead, which reintroduced the exact
        `KeyError` the filter had just removed, one line below it. A filtered set with an unfiltered
        one still in scope is a trap; this is the filtered one, named.
        """
        return [member for group in self.groups for member in group.members]

    @property
    def seats(self) -> int:
        """Members served. With `len(self.groups)` this is the whole trade the tool reports."""
        return sum(len(group.members) for group in self.groups)

    @property
    def per_decode(self) -> float:
        """Seats bought per decode spent — the number an efficiency entry is trying to raise."""
        return self.seats / len(self.groups)

    @staticmethod
    def read_members(paths: list[Path]) -> list[Member]:
        """Every checkpoint that exists AND records a reading, naming the rest rather than skipping quietly.

        A roster listing a checkpoint we have not trained is the normal state of a candidate list being
        edited, and it is also how a typo looks. Both are reported; neither is fatal, because the grouping
        of what IS here is still the answer to the question asked.

        Whether each of them can be PLACED in a decode group is a separate question, answered by
        `RosterReads.over` at the one point the key is taken.
        """
        found = [path for path in paths if path.is_file()]
        for absent in [path for path in paths if not path.is_file()]:
            logger.warning("no checkpoint at %s", absent)
        return [Member.from_checkpoint(path) for path in found]

    @staticmethod
    def resolved(names: list[str], processed: Path) -> list[Path]:
        """Roster names or bare paths, as checkpoint paths, tolerating a missing `.pt` and a comment.

        `comp/members.roster` writes `name.pt   # why this member exists`, so the comment is stripped here
        rather than by every caller — a roster read with its comments attached resolves to nothing and
        reports every member absent, which reads exactly like an empty roster.
        """
        paths = []
        for line in names:
            name = line.split("#")[0].strip()
            if not name:
                continue
            path = Path(name)
            paths.append(path if path.parent != Path() else processed / f"{Path(name).stem}{CHECKPOINT_SUFFIX}")
        return paths

    def lines(self) -> list[str]:
        """Every read with its members, then the trade."""
        rows = [line for group in self.groups for line in group.lines()]
        rows.append(f"{self.seats} seats on {len(self.groups)} decodes = {self.per_decode:.1f} seats per decode")
        return rows


def main() -> None:
    """Report the decodes a set of checkpoints would cost, and which members share each one."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--roster", type=Path, default=None, help="a roster file, one member per line")
    command.parser.add_argument(
        "--candidates", nargs="*", default=[], help="member names or checkpoint paths, instead of a roster"
    )
    command.parser.add_argument(
        "--max-decodes",
        type=int,
        default=0,
        help="exit non-zero if the set costs more decodes than this; 0 reports without judging",
    )
    arguments = command.parsed()

    processed = DataRoot.from_config(arguments.paths).processed()
    named = arguments.roster.read_text(encoding="utf-8").splitlines() if arguments.roster else arguments.candidates
    members = RosterReads.read_members(RosterReads.resolved(list(named), processed))
    # THE DENOMINATOR, refused rather than divided by: a roster whose every line resolved to nothing
    # would otherwise print "0 seats on 0 decodes" through a ZeroDivisionError, or worse, "0 decodes"
    # as if the entry were free.
    Denominator(len(members), "no checkpoint named by --roster or --candidates is on disk").refuse_empty()

    reads = RosterReads.over(members)
    for line in reads.lines():
        logger.info(line)
    # THE PACKAGED HALF, because a free seat still needs its weights uploaded: a member on an
    # already-paid read costs no decode and a dataset push is not nothing.
    logger.info("resolutions across the whole set: %s", dict(Counter(member.image_size for member in reads.placed)))
    if arguments.max_decodes and len(reads.groups) > arguments.max_decodes:
        logger.error(
            "this set costs %d decodes against a ceiling of %d — the reads above name which members "
            "split off, and a member whose crop, tie or band disagrees is usually the one",
            len(reads.groups),
            arguments.max_decodes,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
