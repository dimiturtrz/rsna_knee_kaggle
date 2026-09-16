"""Which holdout records on disk may be READ — the pool every pooled reading starts from.

A directory of `.holdout.npz` is not a pool. 438 records sit beside each other under the models root and
19 of them carry a reading that is VOID: `dinov3f0_solo`, `raptor_v4_swa` and the DINOv3 decode family
were fitted on weights served to the competition, in-sample on 703 of the 909 holdout studies
(`comp/members.declines`). They are also the HIGHEST-scoring records there — `raptor_v4_swa` reads
LEDGER75 0.9425, above the champion — so any statistic a pooled reading takes over the default glob is
led by exactly the rows that cannot mean anything. That defect has been paid for once already: on
2026-08-29 `raptor_solo`'s 0.8938 was carrying the whole "compression goes negative at the top" reading
plus eleven of fifteen apparent ruler inversions, and all of it was in-sample.

ONLY A VOID DECLINE IS EXCLUDED, and the distinction is the whole rule. Most declines are about SHIPPING
- a smoke run, an older label table, a member below the floor - and `roster_check` states that such a
member's "recorded holdout number is still quotable". A void decline is the other kind: it says the
number itself was never a reading of held-out studies. So 277 of the records here are declined and stay
in the pool; 19 leave it.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from kneemri.runs.holdout_record import SUFFIX as HOLDOUT_SUFFIX

logger = logging.getLogger(__name__)

DECLINES = Path("comp/members.declines")
CHECKPOINT_SUFFIX = ".pt"
COMMENT = "#"
VOID = "void"


@dataclass(frozen=True)
class MemberDeclines:
    """The committed statement of which scoreable members deliberately do NOT ship, and why each.

    Read by the thing that CHECKS the roster and by the thing that builds a POOL, never by the thing that
    packages: a decline never affects what ships, it affects whether an absence is a finding and whether
    a number may be quoted.
    """

    text: str

    @classmethod
    def read(cls, path: Path = DECLINES) -> "MemberDeclines":
        """The file, or an empty statement when a repo has declined nothing yet."""
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

    def void(self) -> set[str]:
        """The members whose recorded number is not a reading of held-out studies at all.

        Matched on the word in the decline's own prose, because that is where the fact lives: the file
        is the committed argument, and a second list of names would be the same memory problem the
        declines file exists to end.
        """
        return {name for name, reason in self.reasons().items() if VOID in reason.lower()}


@dataclass(frozen=True)
class MemberPool:
    """The readable holdout records in a directory, with the void ones taken out and counted."""

    directory: Path
    declines: MemberDeclines

    @classmethod
    def at(cls, directory: Path, declines: Path = DECLINES) -> "MemberPool":
        """The pool a `--holdouts` directory stands for."""
        return cls(directory, MemberDeclines.read(declines))

    def names(self, only: list[str] | None = None) -> list[str]:
        """The member names to read. `only` is taken as given — an explicit ask is never second-guessed.

        A named member is the caller pricing a specific record, void or not (`column_headroom --only
        <member>` is how a single arm's per-column table is taken). The filter guards the DEFAULT, which
        is the one nobody inspects.
        """
        if only:
            return only
        # By SUFFIX, never by the first dot. A member name may hold one — `merged_blend_cf_n12b0.1-0.9`
        # is a blend named for its band — and splitting there truncates it to `merged_blend_cf_n12b0`,
        # which matches no decline and no roster line. That member is VOID, and under the split it was
        # the one void record that walked back into the pool.
        found = sorted(path.name.removesuffix(HOLDOUT_SUFFIX) for path in self.directory.glob(f"*{HOLDOUT_SUFFIX}"))
        void = self.declines.void()
        kept = [name for name in found if name not in void]
        dropped = len(found) - len(kept)
        if dropped:
            logger.info(f"pool: {len(kept)} of {len(found)} records; {dropped} excluded as VOID by {DECLINES}")
        return kept
