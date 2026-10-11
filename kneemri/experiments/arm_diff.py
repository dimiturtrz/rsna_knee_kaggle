"""One arm's flag line read against ANOTHER arm's — the two checks whose subject is a pair.

Every other check in `arm_check` asks a question of one invocation: are its flags real, does its cap
match its slices, does it state its operating point. These two ask what an arm is relative to the rest
of the tree — whether it differs from the control it names in ways it never declared (`mismatched`), and
whether it differs from some other arm at all (`twinned`). Both cost card when they were missing, and
both need the same two things: a flag line parsed into values, and the pair's own declarations.

It is a separate module because the questions compose over the whole printed set rather than over one
arm, which is also why they are the only checks taking `against`.
"""

import shlex
from collections.abc import Collection, Mapping

from kneemri.experiments.arm_header import ArmHeader
from kneemri.training.recipe_layers import CONFIG, SET

REPEATABLE = (CONFIG, SET)

INCIDENTAL_FLAGS = frozenset({"--out", "--labels", "--paths"})
"""Flags every arm sets differently by construction, so a diff on them is not a difference of RECIPE."""


class ArmDiff:
    """What an arm's flag line says about it relative to the arms it is not."""

    def __init__(self, headers: ArmHeader) -> None:
        self.headers = headers

    @staticmethod
    def values(flags: str) -> dict[str, str]:
        """One arm's flag line as flag -> value, with a bare switch mapping to the empty string.

        Values are normalised through `float` where both sides are numbers, because `4e-5` and `4e-05`
        are the same learning rate and a diff that reports them is a diff nobody reads twice.

        A repeatable flag (`--config`, `--set`) keeps every value in line order: its layers stack, so two
        arms sharing only the LAST recipe file are two recipes, not a twin.
        """
        tokens = shlex.split(flags)
        parsed: dict[str, str] = {}
        for index, token in enumerate(tokens):
            if not token.startswith("--"):
                continue
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            value = "" if following.startswith("--") else following
            parsed[token] = f"{parsed[token]} {value}" if token in REPEATABLE and token in parsed else value
        return parsed

    @staticmethod
    def _same(one: str, other: str) -> bool:
        """Whether two flag values mean the same thing — numerically where both are numbers."""
        try:
            return float(one) == float(other)
        except ValueError:
            return one == other

    def mismatched(self, invocations: Mapping[str, str], against: Mapping[str, str]) -> dict[str, list[str]]:
        """Each arm's UNDECLARED flag differences from the control it names, as `flag: control -> arm`.

        THE HOLE THIS CLOSES, and it cost 75 minutes of card on 2026-08-24. `distill_pairs.sh` declared
        "the control is band_dense byte-for-byte" and "ONE VARIABLE: --distill-weight", then omitted
        `--backbone-lr` and trained at the trainer's 8e-6 default against a control at 4e-5. Every other
        check passed it: they ask whether the flags PRESENT are real, and have no notion of the flags
        that should have been. The number it produced could not be read against anything.

        The diff itself already existed — `kneemri.evaluation.run_comparison` prints
        `backbone_lr 4e-05->8e-06` — but AFTER the card is spent. This is that diff, before.

        AN ARM IS SUPPOSED TO DIFFER, so a difference is reported only where the arm has not DECLARED it
        with a `# deliberate: <flag>` line. The declaration is the check: an experiment moving one thing
        writes one line, and an author writing three lines is being shown that their arm moves three.

        `against` is the FULL printed set rather than the selection, because `--only margin_seat` is
        exactly the invocation that must still resolve `band_dense`.
        """
        differences: dict[str, list[str]] = {}
        for name, control in self.headers.controls(invocations).items():
            if control not in against or control == name:
                continue
            found = self.undeclared(
                self.values(invocations[name]), self.values(against[control]), self.headers.deliberate(name)
            )
            if found:
                differences[name] = found
        return differences

    def undeclared(self, mine: Mapping[str, str], theirs: Mapping[str, str], declared: Collection[str]) -> list[str]:
        """The flags two parsed lines disagree on, minus the incidental and the declared, as `flag: control -> arm`."""
        return [
            f"{flag}: {theirs.get(flag, 'absent')} -> {mine.get(flag, 'absent')}"
            for flag in sorted(set(mine) | set(theirs))
            if flag not in INCIDENTAL_FLAGS
            and flag not in declared
            and not (flag in mine and flag in theirs and self._same(mine[flag], theirs[flag]))
        ]

    def twinned(self, invocations: Mapping[str, str], against: Mapping[str, str]) -> dict[str, str]:
        """Each arm whose flag line is ALREADY another arm's, up to the file it writes.

        WHAT IT COST, on 2026-08-30 and within one session of writing it. `recal_dense` was on the card,
        twelve epochs of the v3 supervision table at the dense corner. I read `recal_holdannot.sh` and
        `band_dense_hold.sh` to build the ledger's named follow-up, wrote `recal_band_dense.sh`, passed it
        through every check in this module, committed it and queued it behind two other arms — and it was
        `recal_dense` again, flag for flag. Forty minutes of card, and worse than the card: two runs of one
        recipe under two names, either of which a later session could quote as a replication of the other.

        `--out` is the only difference a twin is allowed to have, which is why it is the only flag dropped
        here. `--labels` is NOT dropped, unlike in `mismatched`: a control diff treats the table as
        incidental because every arm names one, but a label-table swap is the entire variable of the label
        lane, and two arms differing only there are two experiments rather than one duplicated.

        `against` is the full printed set for the same reason `mismatched` takes it — `--only new_arm` is
        exactly the invocation that must still see the twin it is about to duplicate.

        AN ARM MAY DELIBERATELY BE ANOTHER'S TWIN, and `ArmHeader.paired` holds both ways an author says
        so — a `# twin-of:` line, or printing both names from one shell.

        Not covered, and deliberately: an arm that duplicates a RUN rather than another arm file. The
        running set is `*.running.sh` in the processed experiments directory, which is out of the repo and
        absent in CI. This catches the case where the duplicate is committed, which is where it was caught.
        """
        identity = {name: self._identity(flags) for name, flags in against.items()}
        twins: dict[str, str] = {}
        for name in invocations:
            mine = identity.get(name)
            if mine is None:
                continue
            others = sorted(
                other
                for other, theirs in identity.items()
                if other != name and theirs == mine and not self.headers.paired(name, other)
            )
            if others:
                twins[name] = (
                    f"is {', '.join(others)} flag for flag, up to --out — so it spends a card slot to "
                    f"produce a second name for one recipe, and no comparison between them means anything"
                )
        return twins

    @classmethod
    def _identity(cls, flags: str) -> tuple[tuple[str, str], ...]:
        """An arm's flag line as what makes it a distinct EXPERIMENT — everything but the file it writes."""
        return tuple(sorted((flag, value) for flag, value in cls.values(flags).items() if flag != "--out"))
