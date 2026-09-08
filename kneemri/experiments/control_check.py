"""Whether an arm's CONTROL can carry the comparison: `python -m kneemri.experiments.control_check`.

Every check in this package guards a shipment, a queue, or a single number. None of them guards a
COMPARISON — and a comparison is what an arm actually reports. An arm's whole claim is `mine − control`,
so a control that is not comparable makes the delta a statement about the control instead of about the
change, and nothing anywhere goes red.

WHAT IT COST BEFORE IT EXISTED. `adapt_null` was built to separate the two things every SSL collapse arm
changed at once, and it scored 0.8240 against `band_dense` 0.8578. That −0.0338 is quoted in four
write-ups as the cost of the `--encoder-weights` load path. It is not one:

- **BUDGET.** `adapt_null` ran 4 epochs and `band_dense` ran 12. Best-of-4 against best-of-12. At the
  same epoch the gap is −0.0090, and `band_dense` gains +0.0248 between its own e3 and its peak — the
  whole of the difference between the two numbers, and a fact about the schedule.
- **UNSOURCED.** The ledger quotes the control's epoch 3 as 0.8232. `band_dense_s2026.metrics.jsonl` has
  e2 0.8197 and e3 0.8330 and no epoch anywhere reads 0.8232. A residual computed against it would have
  read +0.0008 and closed the axis on a number with no producer.
- **BUILD.** The two recipes differ by eight keys `band_dense` does not record at all — `accumulate`,
  `layer_decay`, `slice_regions`, `through_plane`, `grade_ladder`, `severity_weight`, `train_fraction`,
  `encoder_adapt_depth`. Absent means the flag did not EXIST in that build, not that it defaulted. The
  runs started 3.11 days apart. "Byte-identical except one flag" was never established, and `layer_decay`
  alone is a whole optimiser strategy that existed for one of them.

WHY A DETECTOR AND NOT A READING. Each of those is invisible in the only artefact a write-up carries: two
numbers and their difference. A truncated control looks like a converged one, a fabricated score looks
like a measured one, and a control from an older trainer looks like the same trainer. The arms already
declare what they were compared against — `# control  <name> = <score>`, 228 of 296 of them — so the
join is mechanical and the whole ledger can be asked at once instead of one write-up at a time.

WHAT IT CANNOT SAY. Whether a comparable control was the RIGHT control: an arm compared against a
same-build, same-budget run of an unrelated recipe passes here. It reads the comparison the arm declared,
not the one it should have declared. And an arm whose control line names a run with no metrics file on
this machine reads UNRESOLVED, which is a fact about this disk rather than about the arm — four members
have had their source caches deleted and cannot be re-scored at all.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from core import cli
from core.denominator import Denominator
from kneemri.runs.run_metrics import RunMetrics

logger = logging.getLogger(__name__)

DECLARATION = re.compile(r"^#\s+control\s+(?P<name>[A-Za-z0-9_.-]+)\s*=\s*(?P<score>[01]\.\d+)")
"""The convention an arm states its comparison in, at the foot of its header block.

Anchored at the line start and requiring the `#` so a control named inside prose is not harvested: an
arm's paragraphs discuss several runs and only the declaration line is a claim about which one the delta
is against."""

TRAINED = re.compile(r"^run_train\s+(?P<name>[A-Za-z0-9_.-]+)")
"""How an arm names its own run. `foreign_run` arms have no such line and are skipped — their trainer
writes no metrics file here, so there is nothing on this side of the comparison to read."""

SEED_SUFFIX = "_s2026"
"""Some runs are spelled with the seed in the checkpoint and some are not, and a control line quotes
whichever spelling the author had in mind. Resolving both is not laxity: refusing the other spelling
would report UNRESOLVED for a run sitting on disk, which teaches the reader to ignore the output."""

QUOTED_PLACES = 4
"""Scores are written to four decimals everywhere in this repo, so a quoted number is compared against
each epoch rounded the same way. Comparing at full precision would report every correctly-sourced score
as unsourced."""

EPOCHS = "epochs"
"""The recipe key holding the budget an arm asked for — the axis `adapt_null`'s −0.0338 was really on."""

AUC = "auc"


@dataclass(frozen=True)
class Comparison:
    """One arm's declared control, and everything that decides whether the delta means anything."""

    arm: str
    control: str
    quoted: float
    source: Path

    def line(self, findings: tuple[str, ...], measured: float | None) -> str:
        """One flagged comparison, with the control's OWN best beside the score the arm quoted for it.

        The offset is the whole reading. A quoted score 0.0002 off its control's curve is a
        transcription slip worth one correction; one 0.038 above it is a number from a DIFFERENT RULER,
        and it makes the arm's delta wrong by that much in the direction that reads as a loss. Printing
        the verdict alone made those two look identical, which is how three gold-ruler quotes sat in the
        ledger as controls for holdout-ruler runs.
        """
        against = "no curve" if measured is None else f"its best {measured:.4f} ({self.quoted - measured:+.4f})"
        stated = f"{self.arm} vs {self.control} = {self.quoted:.4f}, {against}"
        return f"{', '.join(findings):<30} {stated}  [{self.source.name}]"


@dataclass(frozen=True)
class Comparability:
    """What a control can and cannot carry, read off both runs' metrics rather than off the write-up."""

    unsourced: str = "UNSOURCED"
    budget: str = "BUDGET"
    build: str = "BUILD"
    unresolved: str = "UNRESOLVED"

    def sourced(self, control: RunMetrics, quoted: float) -> bool:
        """Whether the quoted score is any epoch of the control's own curve.

        Any epoch rather than the best one, because a write-up legitimately quotes a matched epoch — the
        defect this catches is a number that is on NO epoch, which is the one thing a measured score
        cannot be.
        """
        scored = {round(float(row[AUC]), QUOTED_PLACES) for row in control.rows() if row.get(AUC) is not None}
        return round(quoted, QUOTED_PLACES) in scored

    def budgeted(self, arm: RunMetrics, control: RunMetrics) -> bool:
        """Whether both runs asked for the same number of epochs.

        The recipe's ask rather than the rows delivered: a run killed at 4 of 12 is a truncation, which
        `convergence_check` owns, while a run that ASKED for 4 under `one_cycle` annealed over 4 and is a
        different recipe rather than a shorter one.
        """
        return self._asked(arm) == self._asked(control)

    def built(self, arm: RunMetrics, control: RunMetrics) -> tuple[str, ...]:
        """The recipe keys present in one run and absent from the other, which are code paths, not values.

        A key missing from a recipe block was not defaulted — it did not exist in the trainer that wrote
        it. So a non-empty answer here says the two runs are on different builds and the comparison
        carries every behavioural change between them, not just the flag the arm names.
        """
        mine, theirs = self._keys(arm), self._keys(control)
        return tuple(sorted(mine ^ theirs))

    def _asked(self, run: RunMetrics) -> str | None:
        recipe = run.recipe()
        return None if recipe is None else recipe.get(EPOCHS)

    def _keys(self, run: RunMetrics) -> frozenset[str]:
        recipe = run.recipe()
        return frozenset() if recipe is None else frozenset(recipe)


@dataclass(frozen=True)
class ControlCheck:
    """Reads every arm's declared control and says which comparisons their deltas cannot support."""

    models: Path
    comparability: Comparability = Comparability()

    def declared(self, arm: Path) -> Comparison | None:
        """The comparison one arm script states, or nothing if it declares no control or trains no run."""
        lines = arm.read_text(encoding="utf-8", errors="replace").splitlines()
        control = self._first(DECLARATION, lines)
        trained = self._first(TRAINED, lines)
        if control is None or trained is None:
            return None
        return Comparison(trained["name"], control["name"], float(control["score"]), arm)

    def resolve(self, name: str) -> RunMetrics | None:
        """The metrics file for a run named either with or without the seed suffix."""
        for spelling in (name, f"{name}{SEED_SUFFIX}"):
            record = RunMetrics(self.models / f"{spelling}.pt")
            if record.rows():
                return record
        return None

    def findings(self, comparison: Comparison) -> tuple[tuple[str, ...], float | None]:
        """Every reason this arm's delta says something other than its own change, and the control's best."""
        arm = self.resolve(comparison.arm)
        control = self.resolve(comparison.control)
        if arm is None or control is None:
            return (self.comparability.unresolved,), None
        best = control.best()
        found = [] if self.comparability.sourced(control, comparison.quoted) else [self.comparability.unsourced]
        found += [] if self.comparability.budgeted(arm, control) else [self.comparability.budget]
        drifted = self.comparability.built(arm, control)
        found += [f"{self.comparability.build}({len(drifted)})"] if drifted else []
        return tuple(found), None if best is None else best["auc"]

    def audit(self, arms: Path) -> list[tuple[Comparison, tuple[str, ...], float | None]]:
        """Every declared comparison that cannot carry its own delta, worst first."""
        declared = [found for arm in sorted(arms.glob("*.sh")) if (found := self.declared(arm)) is not None]
        seen = [(comparison, *self.findings(comparison)) for comparison in declared]
        return sorted((row for row in seen if row[1]), key=lambda row: (-len(row[1]), row[0].arm))

    def _first(self, pattern: re.Pattern[str], lines: list[str]) -> re.Match[str] | None:
        return next((found for line in lines if (found := pattern.match(line)) is not None), None)


def main() -> None:
    """Report every arm whose control cannot carry its delta, and exit non-zero on an unsourced score."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--arms", type=Path, default=Path("experiments/arms"), help="the arm scripts")
    arguments = command.parsed()

    check = ControlCheck(command.rooted(arguments).models())
    seen = check.audit(arguments.arms)
    Denominator(len(list(arguments.arms.glob("*.sh"))), f"no arm scripts under {arguments.arms}").refuse_empty()

    for comparison, found, measured in seen:
        logger.info("%s", comparison.line(found, measured))
    unsourced = [row for row in seen if Comparability().unsourced in row[1]]
    logger.info(
        "%d declared comparisons flagged: %d unsourced, %d budget, %d build",
        len(seen),
        len(unsourced),
        sum(Comparability().budget in found for _, found, _measured in seen),
        sum(any(item.startswith(Comparability().build) for item in found) for _, found, _measured in seen),
    )
    if unsourced:
        # Only UNSOURCED blocks. A budget or build mismatch is a real comparison that needs a caveat;
        # an unsourced score is a number no run produced, and there is no caveat that repairs one.
        logger.info("an unsourced score is on NO epoch of the control's curve - it has no producer")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
