"""The two objects that decide an entry's DECODE BILL, which is its number of distinct readings.

They live in their own module rather than beside `Serving` because `ensemble.py` had reached the
god-file ceiling, and because they are a pair a reader wants together: `SharedRead` is ONE reading
every member is served under, `ReadGroups` is that same move applied to NAMED members only. The
all-or-nothing form is the special case rather than the general one — but it is the one that shipped
first, so both are here and neither is a mode flag on the other.

`holdout_backfill` prices a regroup on a member's own holdout and the kernel serves through it, so a
second copy of a regroup would be a second definition of what shipped.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from kneemri.inference.members import RESIZED_FROM, Member


@dataclass(frozen=True)
class SharedRead:
    """Serving an already-fitted member under ANOTHER read group's depth and band.

    THE ENTRY'S DECODE BILL IS THE NUMBER OF READ GROUPS, NOT THE NUMBER OF SEATS. `Ensemble.reads`
    groups members by their reading, and a group is one decode of every study; sizes collapse inside a
    group because `SlotReader.cropped` stops before the resize, but the slice count and the band do not
    — they decide WHICH files are opened. `raptorweight_v1`'s six seats span three groups (20 slices at
    the 0.1-0.9 band, 12 at that band, 12 at the historic one), and its Kaggle log decodes the same
    three visible studies three times over: `decode 4s`, then `2s`, then `2s`. One group would decode
    12 slices per study instead of 44.

    THE COST OF THAT IS A BOARD NUMBER, NOT A RUNTIME ONE. This paragraph read "projecting 7.6 h of a
    9 h limit" until 2026-09-08. There is no projection in it: `Budget.report` prints
    `projecting {projected}h of {seconds}h` and `seconds` is the CEILING, so 7.6 h is the deadline the
    run was given and 0.0 h is what it projected against it. The public rerun scores THREE studies, which
    leaves every decode-and-forward in these entries inside 4-8 s of a run that spends 20-35 s importing;
    `experiments/README.md` documents that instrument as BACKWARDS rather than merely weak, and it had
    already refuted this exact sentence once. What IS measured is `onegroup_v1` — the champion's six seats
    served out of ONE group, 0.942 against 0.944. Price a regroup against that, never against a log.

    So the question this asks is not "how much accuracy does a shallower read cost" but "how much does
    it cost to serve every seat out of ONE decode". It lives beside `Serving` because the same object
    has to answer it in BOTH halves: the kernel serves through it, and `holdout_backfill` prices it on
    each member's own holdout. Two copies of a regroup would be two definitions of what shipped.

    The band is the reader's own `[low, high]` recorded form, so the value handed here is the value a
    manifest carries and nothing has to translate it.

    A re-grouped member carries the `RESIZED_FROM` mark, which is what `TrainerRecord.instead_of`
    refuses on. The mark's name says "size" and its RULE is "this is not the reading the trainer
    fitted", which is what a re-grouped member is; without it the trainer's own array would be handed
    back and an offline price would read zero.
    """

    slices: int
    band: tuple[float, float]

    def regrouped(self, fitted: Member) -> Member:
        """The same weights, read at this group's depth and band."""
        return replace(
            fitted,
            config={**fitted.config, "slices": self.slices, "band": list(self.band), RESIZED_FROM: fitted.read_size},
        )

    def beside(self, checkpoint: Path) -> Path:
        """Where a priced record goes: beside the checkpoint, named for the group that produced it."""
        low, high = self.band
        return checkpoint.with_name(f"{checkpoint.stem}_n{self.slices}b{low:g}-{high:g}{checkpoint.suffix}")


@dataclass(frozen=True)
class ReadGroups:
    """Moving NAMED members onto another reading, leaving the rest on their own.

    THE DECODE BILL IS THE NUMBER OF DISTINCT READINGS, AND THE STEPS BETWEEN THEM ARE NOT ALL THE SAME
    SIZE. `SharedRead` answers "serve everything out of one decode" and that is the largest step there
    is: the champion's six span three readings, and collapsing all six onto one cost a measured -0.002
    on the board (`onegroup_v1`, 0.942 against 0.944) for 7.6 h -> 2.15 h. But three readings do not
    have to become one to become affordable. This entry's six are three at 20 slices / band 0.1-0.9, ONE
    at 12 / 0.1-0.9, and two at 12 / 0.2-0.8 — so moving that ONE member joins it to the pair and the
    bill is two, at a fifth of the members disturbed rather than five sixths.

    `Serving.read` calls `regrouped` on whichever of the two it holds, so the SOME case is not an `if`
    on a mode flag anywhere: it is the same message to a different object.

    A member not named here is returned untouched, which is what makes this composable with a roster
    that grows — a new seat is native until somebody says otherwise, never silently swept into a group
    it was not priced in.
    """

    by_member: Mapping[str, SharedRead]

    def regrouped(self, fitted: Member) -> Member:
        """The named members under their new reading; every other member unchanged."""
        moved = self.by_member.get(fitted.identifier)
        return fitted if moved is None else moved.regrouped(fitted)

    @property
    def readings(self) -> int:
        """How many DISTINCT readings this mapping itself names — the part of the bill it decides."""
        return len(set(self.by_member.values()))
