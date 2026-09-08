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
from kneemri.evaluation.ledger75 import LedgerSeventyFive
from kneemri.runs.run_metrics import RunMetrics

logger = logging.getLogger(__name__)

DECLARATION = re.compile(r"^#\s+control\s+(?P<name>[A-Za-z0-9_.-]+)\s*=\s*(?P<score>[01]\.\d+)(?P<rest>.*)")
"""The convention an arm states its comparison in, at the foot of its header block.

Anchored at the line start and requiring the `#` so a control named inside prose is not harvested: an
arm's paragraphs discuss several runs and only the declaration line is a claim about which one the delta
is against."""

TRAINED = re.compile(r"^run_train\s+(?P<name>[A-Za-z0-9_.-]+)")
"""How an arm names its own run. `foreign_run` arms have no such line and are skipped — their trainer
writes no metrics file here, so there is nothing on this side of the comparison to read."""

OUTPUT = re.compile(r".*?--out\s+\"?(?P<path>\S+?)\.pt\"?")
"""Where an arm says its checkpoint goes, which is what the run is actually NAMED on disk.

A control token is not always a checkpoint stem — `whole_stack` and `rate_wholestack` are ARM names and
their checkpoints are `wholestack_s2026` and `lr4e-5_ws_s2026`. This is the same `--out`-is-the-truth
rule `_arm_log` names an arm's log by; the control resolver is its second consumer and never learned it,
which put three arms that ran into the same bucket as one that never did."""

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

GOLD = "gold"
"""The one ruler word this check cannot compute, and the reason it reads the rest of the declaration.

A gold macro is scored against the radiologist's own annotations, which are not in a holdout record, so
no arithmetic here can reproduce one. `rank966_hold` declares `band_dense_hold_s2026 = 0.8956 gold /
~0.857 proxy` — the arm is right, the number is real, and it is unreachable from this side. Reporting
that as a score with no producer is the same error this check exists to catch, so it gets its own
verdict and does not block. This is ONE keyword rather than a parse of the ruler's spelling: `L75mid`,
`own holdout`, `ledger75` and `896, L75mid` all appear on declaration lines, and a check that had to
recognise them would fail on the next one somebody invents.
"""


@dataclass(frozen=True)
class Comparison:
    """One arm's declared control, and everything that decides whether the delta means anything."""

    arm: str
    control: str
    quoted: float
    source: Path
    rest: str = ""
    """Whatever the declaration line says after the number — where the arm names its ruler, if it does."""

    def gold(self) -> bool:
        """Whether the arm declared a GOLD-ruler control, which nothing on this side can reproduce."""
        return GOLD in self.rest.lower()

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
    unruled: str = "UNRULED"
    unrun: str = "UNRUN"
    """The ARM has no metrics file, so there is no comparison to judge yet.

    Split out of UNRESOLVED because the two are opposite readings and the merged bucket said neither.
    UNRESOLVED is a claim about a CONTROL that cannot be found and is a defect in the write-up; UNRUN is
    an arm that has not spent the card yet — `mae_unitfix` is queued behind a pretrain as this is
    written — and is not a finding at all. Merged, the check reported 42 UNRESOLVED against a
    researcher's independently-swept 4, and the whole difference was arms waiting their turn."""

    def sourced(self, control: RunMetrics, quoted: float) -> bool:
        """Whether the quoted score is the control's, under ANY ruler this side can compute.

        Any epoch rather than the best one, because a write-up legitimately quotes a matched epoch — the
        defect this catches is a number that is on NO epoch, which is the one thing a measured score
        cannot be.

        AND UNDER BOTH CUTS, because the first version of this check compared a declared number against
        `.metrics.jsonl`'s `auc` alone — which is the MIDPOINT macro — and then reported three arms as
        quoting a fabricated control. All three were right: `recal_through`'s declared 0.8867 is
        `through_dense_s2026`'s LEDGER75 macro to four decimals, and its own line says so
        (`= 0.8867 (own holdout, L75mid, n=909)`). The recut is worth about +0.031 and every one of the
        three sat there. That is `statistic / population / cut` (knee-f9aao) — the rule this check
        enforces on arms — broken by the check itself, which is why the fix is to score under every
        ruler rather than to parse the one the arm named.
        """
        target = round(quoted, QUOTED_PLACES)
        midpoint = {round(float(row[AUC]), QUOTED_PLACES) for row in control.rows() if row.get(AUC) is not None}
        return target in midpoint or target in self._recomputed(control)

    def scoreable(self, control: RunMetrics) -> bool:
        """Whether the control kept predictions, which is a run this side can price with no metrics file.

        An epoch export carries the checkpoint and the holdout record and leaves the metrics file under
        the training run's own stem. It is a real control and the four arms citing one were right.
        """
        return bool(self._recomputed(control))

    def _recomputed(self, control: RunMetrics) -> frozenset[float]:
        """Both macro rulers off the control's kept predictions, or nothing when it kept none.

        BOTH, because the recut is not a correction to the midpoint — they are two rulers and an arm may
        declare against either. Taking only the recut here would have read `wsfold0_e6`'s four citing
        arms as quoting a fabricated 0.8457 when 0.8457 is its midpoint macro and 0.8662 its ledger75.

        A dark member (`--fold -1`) holds nothing back and can never have either, and a member whose
        source cache was deleted cannot be re-scored — both are silence rather than an error, because the
        question asked is "is the declared number reachable", and an unreachable ruler answers no.
        """
        ruler = LedgerSeventyFive()
        try:
            row = ruler.row(control.checkpoint())
        except (FileNotFoundError, OSError, ValueError):
            return frozenset()
        return frozenset({round(row.ledger, QUOTED_PLACES), round(row.ledger75, QUOTED_PLACES)})

    def recorded(self, arm: RunMetrics, control: RunMetrics) -> bool:
        """Whether both runs wrote a recipe, which is what BUDGET and BUILD are read out of.

        An epoch export has predictions and no metrics file, so its recipe is unreadable — and an absent
        recipe is not a differing one. Judging it anyway reports every key of the arm as drift and every
        budget as mismatched, which is this check inventing the defect it exists to find, on the exact
        control (`wsfold0_e6`) whose four citing arms it had already accused once.
        """
        return arm.recipe() is not None and control.recipe() is not None

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
    arms: Path
    comparability: Comparability = Comparability()

    def declared(self, arm: Path) -> Comparison | None:
        """The comparison one arm script states, or nothing if it declares no control or trains no run."""
        lines = arm.read_text(encoding="utf-8", errors="replace").splitlines()
        control = self._first(DECLARATION, lines)
        trained = self._first(TRAINED, lines)
        if control is None or trained is None:
            return None
        return Comparison(trained["name"], control["name"], float(control["score"]), arm, control["rest"])

    def resolve(self, name: str) -> RunMetrics | None:
        """The run a control token names, under every spelling this repository gives one run.

        Three, because a token is written by whoever remembered the run and they remembered a different
        artefact each time: the checkpoint stem, the stem without its seed, or the ARM whose `--out` wrote
        it. The last one is the `--out`-is-the-truth rule, and without it three arms that ran read as
        never-run — `whole_stack` is `wholestack_s2026` and `rate_wholestack` is `lr4e-5_ws_s2026`.

        A checkpoint with a holdout record and NO metrics file resolves too, with an empty curve: it is a
        real, scoreable run (`wsfold0_e6` is `wholestack`'s epoch-6 export under a second name), and
        `sourced` recomputes both cuts off the record. Its recipe is unreadable, so budget and build
        cannot be judged — that is silence about them, never a pass.
        """
        for spelling in (name, f"{name}{SEED_SUFFIX}"):
            record = RunMetrics(self.models / f"{spelling}.pt")
            if record.rows() or self.comparability.scoreable(record):
                return record
        stem = self._output_stem(name)
        return None if stem is None or stem == name else self.resolve(stem)

    def _output_stem(self, arm: str) -> str | None:
        """The checkpoint basename the arm of this name writes, or nothing if no such arm exists."""
        script = self.arms / f"{arm}.sh"
        if not script.is_file():
            return None
        found = self._first(OUTPUT, script.read_text(encoding="utf-8", errors="replace").splitlines())
        return None if found is None else Path(found["path"]).name

    def findings(self, comparison: Comparison) -> tuple[tuple[str, ...], float | None]:
        """Every reason this arm's delta says something other than its own change, and the control's best."""
        arm = self.resolve(comparison.arm)
        control = self.resolve(comparison.control)
        if arm is None:
            return (self.comparability.unrun,), None
        if control is None:
            return (self.comparability.unresolved,), None
        best = control.best()
        sourced = self.comparability.sourced(control, comparison.quoted)
        found: list[str] = [] if sourced else [self._unmatched(comparison)]
        if not self.comparability.recorded(arm, control):
            return tuple(found), None if best is None else best[AUC]
        found += [] if self.comparability.budgeted(arm, control) else [self.comparability.budget]
        drifted = self.comparability.built(arm, control)
        found += [f"{self.comparability.build}({len(drifted)})"] if drifted else []
        return tuple(found), None if best is None else best["auc"]

    def audit(self) -> list[tuple[Comparison, tuple[str, ...], float | None]]:
        """Every declared comparison that cannot carry its own delta, worst first.

        The arm directory is the check's, not this call's: resolving a control now READS those scripts,
        so a second path here could audit one tree while resolving names against another.
        """
        declared = [found for arm in sorted(self.arms.glob("*.sh")) if (found := self.declared(arm)) is not None]
        seen = [(comparison, *self.findings(comparison)) for comparison in declared]
        return sorted((row for row in seen if row[1]), key=lambda row: (-len(row[1]), row[0].arm))

    def _unmatched(self, comparison: Comparison) -> str:
        """What to call a declared score that matches no ruler this side can compute.

        A gold-ruler declaration is UNRULED and does not block: the number is real and unreachable here,
        and calling it unsourced would be this check inventing the defect it was written to find.
        """
        return self.comparability.unruled if comparison.gold() else self.comparability.unsourced

    def _first(self, pattern: re.Pattern[str], lines: list[str]) -> re.Match[str] | None:
        return next((found for line in lines if (found := pattern.match(line)) is not None), None)


def main() -> None:
    """Report every arm whose control cannot carry its delta, and exit non-zero on an unsourced score."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--arms", type=Path, default=Path("experiments/arms"), help="the arm scripts")
    arguments = command.parsed()

    check = ControlCheck(command.rooted(arguments).models(), arguments.arms)
    seen = check.audit()
    Denominator(len(list(arguments.arms.glob("*.sh"))), f"no arm scripts under {arguments.arms}").refuse_empty()

    for comparison, found, measured in seen:
        logger.info("%s", comparison.line(found, measured))
    unsourced = [row for row in seen if Comparability().unsourced in row[1]]
    logger.info(
        "%d declared comparisons flagged: %d unsourced, %d unruled, %d budget, %d build, %d unresolved, %d unrun",
        len(seen),
        len(unsourced),
        sum(Comparability().unruled in found for _, found, _measured in seen),
        sum(Comparability().budget in found for _, found, _measured in seen),
        sum(any(item.startswith(Comparability().build) for item in found) for _, found, _measured in seen),
        sum(Comparability().unresolved in found for _, found, _measured in seen),
        sum(Comparability().unrun in found for _, found, _measured in seen),
    )
    if unsourced:
        # Only UNSOURCED blocks. A budget or build mismatch is a real comparison that needs a caveat;
        # an unsourced score is a number no run produced, and there is no caveat that repairs one.
        logger.info("an unsourced score is on NO epoch of the control's curve - it has no producer")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
