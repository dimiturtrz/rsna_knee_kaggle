"""Did the arm's pre-registered READING ever get taken? `python -m kneemri.experiments.reading_check`.

Every other check in this project guards a shipment or a number. None of them guards the QUESTION, and
that is where the hole was. `recal_dense.sh` pre-registers on the gold ruler and says so in its header —
its three branches are all defined on a delta that a SECOND command (`kneemri.evaluation.supervision_gap`)
has to produce. The arm ran, converged, entered the ledger, joined the roster and was packaged; the gold
reading was never taken, and for four days the arm's deciding number did not exist while every gate in the
repo read green. `ledger_check` verifies a run reached the ledger. Nothing verified the ledger row answers
the question the run was armed to ask (knee-v78o1).

A READING IS DECLARED IN THE ARM, in a line the machine can read, because the convention already exists:
237 arms carry `# deliberate:` and 149 carry `# bead:`. This adds one more:

    # reading: annotated_predictions/recal_dense*

The right-hand side is a glob, tried against the processed root first and the repo second, and it names
the ARTEFACT the reading leaves behind rather than the command that leaves it. A command name would only
record that somebody wrote one down; a file on disk is the reading actually having happened.

TWO RULES, AND ONLY ONE OF THEM REFUSES.

DECLARED-AND-MISSING refuses. An arm whose checkpoint is on disk has run, so each glob it declared must
match something. This has no false positives by construction — the arm named the artefact itself — and it
is exactly the recal_dense shape.

DARK-AND-UNDECLARED reports. An arm fitting `--fold -1` holds out nothing, so its solo AUC is `nan` on
every epoch and its ledger row can never carry one: the only rulers left to it are a board entry or the
gold set, and both need a declaration. `rampsix` is the live instance — it pre-registered a gate on
'+0.002 solo over their calibsix originals' that is not merely unrun but unreadable by construction, and
spent 4.2 h of card before anyone could have noticed (knee-s8kim). Thirteen arms were dark when this rule
was written and none of them declared, so it REPORTED rather than refused — a gate that fails thirteen-strong
on its first run is a gate people learn to skip. All thirteen were written on 2026-09-04, each with a
`# unmeasurable:` line naming what it knowingly gave up, so the rule has GRADUATED and now REFUSES: a dark
arm is a new one, and a new one is the defect (knee-h1k0y).
"""

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path

from core import cli

logger = logging.getLogger(__name__)

ARMS = Path("experiments/arms")
ARM_GLOB = "*.sh"
READING = "# reading:"
UNMEASURABLE = "# unmeasurable:"
DARK_FIT = "--fold -1"
COMMENT = "#"
CHECKPOINT_GLOB_SUFFIX = "*.pt"


@dataclass(frozen=True)
class ArmDeclaration:
    """One arm script, read for what it says about its own ruler."""

    path: Path
    text: str

    @classmethod
    def read(cls, path: Path) -> "ArmDeclaration":
        return cls(path, path.read_text(encoding="utf-8"))

    @property
    def name(self) -> str:
        """The arm's stem, which is also the prefix its checkpoints are written under."""
        return self.path.stem

    def readings(self) -> list[str]:
        """Every artefact glob this arm declared. Empty for an arm that declared none.

        A GLOB IS ONE TOKEN; PROSE HAS SPACES — the same discriminator `MemberDeclines` needed, for the
        same reason. These headers are wrapped paragraphs, and `cover_labelwise` writes a sentence that
        begins 'reading: a flat mean with the thin findings UP is the finding'. Taken as a path it is a
        finding on its first run, which would have made the check's own debut a false positive.
        """
        declared = [
            line.partition(READING)[2].strip() for line in self.text.splitlines() if line.strip().startswith(READING)
        ]
        return [glob for glob in declared if glob and " " not in glob]

    def unmeasurable(self) -> bool:
        """Whether the arm states in writing that it bought no ruler. A decision, not an omission."""
        return any(line.strip().startswith(UNMEASURABLE) for line in self.text.splitlines())

    def dark(self) -> bool:
        """Whether the arm fits on everything, so no holdout number exists for it to be read by.

        COMMENTS DO NOT MAKE AN ARM DARK. These headers argue about the flag as often as they set it —
        `ema224` opens by calling itself 'the fold-0 measurement' and names `--fold -1` in prose further
        down. A substring over the whole file reads that as a dark arm, which is the check accusing an
        arm of the opposite of what it does.
        """
        return any(DARK_FIT in line for line in self.text.splitlines() if not line.lstrip().startswith(COMMENT))


class ReadingDrift:
    """Every way an arm's pre-registered reading has failed to become an artefact."""

    def __init__(self, models: Path, processed: Path, arms: Path = ARMS) -> None:
        self.models = models
        self.processed = processed
        self.arms = arms

    def declarations(self) -> list[ArmDeclaration]:
        """Every arm script, in name order."""
        return [ArmDeclaration.read(path) for path in sorted(self.arms.glob(ARM_GLOB))]

    def arm_names(self) -> frozenset[str]:
        """Every arm's stem, so a checkpoint can be attributed to the LONGEST name that prefixes it."""
        return frozenset(path.stem for path in self.arms.glob(ARM_GLOB))

    def ran(self, arm: ArmDeclaration) -> bool:
        """Whether any checkpoint on disk carries this arm's name AND belongs to no longer-named sibling.

        The prefix rather than an exact stem, because a run appends its seed (`recal_dense_s2026`) and
        an arm that takes an argument writes several. An arm that has not run yet is not a finding: its
        reading cannot be missing before there is anything to read.

        A SIBLING WHOSE NAME EXTENDS THIS ONE OWNS ITS OWN CHECKPOINTS. Arms are named by extension here —
        `mergedk_labelwise_drop40_long` and `mergedk_labelwise_drop40_long_336` are two arms, one 224px and
        one 336px — so a bare prefix glob credits the 336 run to the arm that never ran and then refuses it
        for not leaving the 224 artefact it declared. That is the check accusing an arm of a run it did not
        make, and it stood as a standing false positive until 2026-09-16. A checkpoint belongs to the
        longest arm name that prefixes it.
        """
        return bool(self.checkpoints(arm))

    def checkpoints(self, arm: ArmDeclaration) -> list[Path]:
        """The checkpoints on disk that belong to THIS arm rather than to a longer-named sibling."""
        longer = [name for name in self.arm_names() if name != arm.name and name.startswith(arm.name)]
        return [
            checkpoint
            for checkpoint in sorted(self.models.glob(f"{arm.name}{CHECKPOINT_GLOB_SUFFIX}"))
            if not any(checkpoint.name.startswith(name) for name in longer)
        ]

    def found(self, glob: str) -> bool:
        """Whether a declared glob matches anything, under the models or processed root, or under the repo.

        THE MODELS ROOT IS WHERE A HOLDOUT ARRAY LIVES, and it was missing here until 2026-09-06: a run
        writes `<checkpoint>.holdout.npz` beside its `.pt`, so every arm whose ruler is its own holdout
        record read as DECLARED-AND-MISSING the moment it finished. `foreign_recipe` was refused with its
        `.holdout.npz` on disk, which is the failure mode that teaches a check to be ignored.

        AN ABSOLUTE DECLARATION IS ACCEPTED, NOT FORBIDDEN (D193 §1). `pathlib.Path.glob` raises
        `NotImplementedError: Non-relative patterns are unsupported` on an absolute pattern, so this
        whole check exited 1 with a traceback and NO report from 2026-09-07 `84660bf5` — dead for a
        day, and four of the five absolute spellings landed inside it. An absolute path is the MORE
        honest declaration: it names the artefact without depending on which of three roots the checker
        happens to try. Resolve it directly, and only a relative one falls through to the roots.
        """
        candidate = Path(glob)
        if candidate.is_absolute():
            return candidate.exists() or any(candidate.parent.glob(candidate.name))
        return any(self.models.glob(glob)) or any(self.processed.glob(glob)) or any(Path().glob(glob))

    def missing(self) -> list[str]:
        """Arms that ran and whose own declared artefact is not there — the recal_dense shape."""
        return [
            f"{arm.name} declared {glob!r} and it matches nothing"
            for arm in self.declarations()
            if self.ran(arm)
            for glob in arm.readings()
            if not self.found(glob)
        ]

    def taken(self, arm: ArmDeclaration) -> list[str]:
        """Declared readings that are ALREADY on disk for an arm that has not run — the complement rule.

        `missing` asserts ran => the artefact exists. This asserts the other direction: not-yet-run => the
        artefact must NOT exist, because an arm whose reading is already there is about to re-take a number
        the project owns. Both halves are the same two predicates, and only this one can be checked BEFORE
        the card is spent.

        A re-read is often legitimate — a member changed, a ruler was recut — so the rule refuses rather
        than forbids, and `--anyway` releases it. The constraint is that the re-reader has to say so: the
        cost is never a bad decision, it is a decision taken without knowing the number exists.

        It keys on the DECLARED reading only, never on what the arm `require`s. An input the arm consumes
        is supposed to be on disk; conflating the two would refuse every arm that reads a built cache.
        """
        return [] if self.ran(arm) else [glob for glob in arm.readings() if self.found(glob)]

    def already_taken(self, only: str) -> list[str]:
        """The refusal lines for ONE named arm, with the matching artefacts named so they can be read.

        AN ARM THAT HAS ALREADY RUN IS THE STRONGER CASE, AND IT REPORTED AS CLEAN. `taken` is the exact
        complement of `missing` and so short-circuits on `ran`, which is correct for the whole-tree pass:
        there, a ran arm's reading being on disk is the thing `missing` asserts. Routed through `--only`
        that short-circuit INVERTED the guard — the one arm whose checkpoint AND number are both on disk,
        the case where re-arming overwrites a result the project already owns, printed the same
        `no declared reading is on disk yet` as an arm that had never run. So the half of this check that
        can still save a card was silent on its worst case. `--only` therefore asks `ran` itself.
        """
        arms = [arm for arm in self.declarations() if arm.name == only]
        return [
            f"{arm.name} HAS ALREADY RUN — re-arming overwrites {self._stamps(self.checkpoints(arm))}"
            for arm in arms
            if self.ran(arm)
        ] + [
            f"{arm.name} declared {glob!r} and it ALREADY matches: {self.matches(glob)}"
            for arm in arms
            for glob in self.taken(arm)
        ]

    def matches(self, glob: str) -> str:
        """What a declared glob resolves to, with mtimes, so the refusal names the number that exists."""
        candidate = Path(glob)
        absolute = candidate.is_absolute()
        roots = [candidate.parent] if absolute else [self.models, self.processed, Path()]
        pattern = candidate.name if absolute else glob
        return self._stamps([hit for root in roots for hit in sorted(root.glob(pattern))])

    def _stamps(self, hits: list[Path]) -> str:
        """Artefacts with their UTC mtimes, because a refusal has to say HOW OLD what already exists is."""
        return ", ".join(f"{hit} ({self._stamped(hit)})" for hit in hits)

    @staticmethod
    def _stamped(hit: Path) -> str:
        """A match's mtime in UTC, because the refusal has to say HOW OLD the number that exists is."""
        return f"{dt.datetime.fromtimestamp(hit.stat().st_mtime, tz=dt.UTC):%Y-%m-%dT%H:%MZ}"

    def undeclared(self) -> list[str]:
        """Dark arms with no ruler written down. GRADUATED 2026-09-04: this refuses now (knee-h1k0y)."""
        return [arm.name for arm in self.declarations() if arm.dark() and not arm.readings() and not arm.unmeasurable()]

    def faults(self) -> list[str]:
        """The findings that refuse: a declared reading never taken, and a dark arm with no ruler at all.

        The dark rule was advisory while thirteen arms fitted `--fold -1` with nothing written down. All
        thirteen now carry a `# unmeasurable:` line saying what they knowingly gave up, so the tree measures
        zero and a NEW dark arm is the defect the rule was written for rather than a backlog it inherits.
        """
        return self.missing() + [
            f"{name} fits --fold -1 and declares neither a reading nor why it has none" for name in self.undeclared()
        ]

    def report(self) -> list[str]:
        """The survey, whether or not a reading went missing."""
        declarations = self.declarations()
        declared = [arm for arm in declarations if arm.readings()]
        dark = self.undeclared()
        lines = [
            *self.faults(),
            *[f"{name} fits --fold -1 and declares no reading — no ruler can price it" for name in dark],
        ]
        lines.append(
            f"{len(declarations)} arms: {len(declared)} declare a reading, "
            f"{len(dark)} are dark with none declared, {len(self.missing())} declared and missing"
        )
        return lines


def main() -> None:
    """Assert every reading an arm pre-registered has left an artefact on disk."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--arms", type=Path, default=ARMS, help="where the arm scripts live")
    command.parser.add_argument(
        "--only", help="check ONE arm before it is armed: refuse if its declared reading is already on disk"
    )
    command.parser.add_argument("--anyway", help="why this reading is being re-taken although it exists")
    arguments = command.parsed()

    root = command.rooted(arguments)
    drift = ReadingDrift(root.models(), root.processed(), arguments.arms)
    if arguments.only:
        taken = drift.already_taken(arguments.only)
        command.report(taken or [f"{arguments.only}: no declared reading is on disk yet"])
        if not arguments.anyway:
            command.conclude(taken, "refusal(s): the arm ran or its reading exists — re-take with --anyway '<why>'")
        return

    command.report(drift.report())
    command.conclude(drift.faults(), "pre-registered reading(s) never taken")


if __name__ == "__main__":
    main()
