"""What has this corpus actually PRICED: `python -m kneemri.experiments.recipe_pairs --holdouts <dir>`.

Every knob audit in this project has read `experiments/arms/*.sh` and looked for two arms differing in
one flag. That works for a flag an arm WRITES and is structurally blind to a leaf left at its default,
because a default is written in no arm script at all. The shipping corner's audit reported 40 of its 93
leaves as unpriceable for that reason, and `scratchpad/tri.py` reported a three-point weight-decay curve
as existing nowhere in 407 arms when its middle point was a default.

The resolved side is on disk and always has been: `<stem>.metrics.jsonl` line 1 carries the recipe with
every default filled in, and the checkpoint carries the same dict under `options`. Reading THAT side
finds roughly seven times as many isolating pairs as reading the arms.

FOUR THINGS A RESOLVED RECIPE STILL CANNOT BE READ NAIVELY FOR. Each is a guard below, and each was
added only after its absence produced a number that had to be retracted.

  * POPULATION. An AUC is a statistic OF a study set, so two rows read on different holdouts are two
    questions rather than a delta. Ignoring the dividers `ledger75` prints gave 162 pairs whose two
    largest keys were `fold` and `exclude_fold` — keys that cross holdouts BY CONSTRUCTION.
  * TRUNCATION. A killed run holds a ledger row like any other, and under `one_cycle` its score is a
    FLOOR rather than a reading. Pairs sort by |delta|, so dead processes land at the TOP of every key,
    which is precisely where a reader quotes from. This cost a -0.0953 `layer_decay` finding that paired
    against a run dead at epoch 3 of 24; the runs that finished say the opposite, at a twentieth the size.
  * SCHEMA DRIFT. A leaf absent from an older recipe resolves to None, which is not a value the knob ever
    took. Tolerating that asymmetry inflated the table to 1015 pairs, admitting recipes eighteen and
    thirty-six leaves apart as "isolating".
  * VOID. A member fitted on its own holdout reads HIGH and cannot be compared to one that was not.
    `comp/members.declines` is where that is recorded, and `Ledger75Row.void` carries it.

WHAT A PAIR IS AND IS NOT. A pair prices the STEP BETWEEN ITS OWN TWO VALUES, on the recipe those two
runs share — never the step from wherever the shipping corner happens to sit. Where the corner's value
lies BETWEEN two deliberately extreme endpoints, a large delta is a gradient across that whole span and
says nothing about moving one notch. Both of the corner's 0.010-sized keys are that shape.
"""

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from core import cli
from core.denominator import Denominator
from kneemri.evaluation.ledger75 import Ledger75Row, LedgerSeventyFive
from kneemri.evaluation.member_pool import DECLINES, MemberPool
from kneemri.experiments.convergence_check import ConvergenceCheck
from kneemri.runs.holdout_record import SUFFIX
from kneemri.runs.recorded_run import RecordedRun
from kneemri.runs.run_metrics import RunMetrics

logger = logging.getLogger(__name__)

NAMING_LEAVES = frozenset({"out", "name", "tag"})
"""Leaves that NAME a run rather than configure it.

Two runs always differ on these and never differ BECAUSE of them, so counting them makes every pair look
two-key and the table comes back empty."""

SHOWN = 8
"""How many pairs of one key are printed, largest |delta| first.

A key with thirty pairs is a distribution rather than a reading, and printing all of it buries the keys
that have one or two."""

PAIRED_FLOOR = 0.0062
"""The smallest difference this 909-study holdout resolves, legislated elsewhere and reused here.

Printed beside each key so a delta is read against the ruler rather than against zero."""


@dataclass(frozen=True)
class KnobPair:
    """Two completed runs on one population whose resolved recipes differ in EXACTLY one leaf."""

    key: str
    low_name: str
    high_name: str
    low_value: str
    high_value: str
    low_score: float
    high_score: float

    def delta(self) -> float:
        """What the corpus paid for moving this leaf from the first value to the second."""
        return self.high_score - self.low_score

    def resolved(self) -> bool:
        """Whether this single pair clears the holdout floor, which is the only thing it can claim alone."""
        return abs(self.delta()) >= PAIRED_FLOOR

    def line(self) -> str:
        """The pair as one line, carrying both runs so the reading can be re-taken by hand."""
        mark = " CLEARS" if self.resolved() else ""
        return (
            f"   {self.low_value:>12} {self.low_score:.4f}  ->  {self.high_value:>12} {self.high_score:.4f}"
            f"   {self.delta():+.4f}{mark}  {self.low_name} / {self.high_name}"
        )


class RecipePairs:
    """Single-key isolating pairs over resolved recipes, guarded on all four axes."""

    def __init__(self, models: Path) -> None:
        self.models = models
        self.convergence = ConvergenceCheck()

    def recipes(self) -> dict[str, dict[str, str]]:
        """Every run on disk that stamped a resolved recipe, keyed by stem.

        Reads through `RecordedRun`, which falls back to the checkpoint's `options` when the metrics file
        predates the stamp — three quarters of the corpus is only reachable that way.
        """
        found = {}
        for metrics in sorted(self.models.glob("*.metrics.jsonl")):
            run = RunMetrics(metrics)
            recipe = RecordedRun.of(run).recipe
            if recipe is not None:
                found[run.name()] = {key: value for key, value in recipe.items() if value is not None}
        return found

    def quotable(self, rows: list[Ledger75Row], population: str) -> dict[str, float]:
        """Scores that may be compared to each other: one population, not void, and finished.

        The three refusals are independent and all three are needed. A row on another population answers a
        different question; a void row was fitted on the studies it is priced on; and a truncated run's
        number is a floor. `settled()` is what separates a truncation from a `--patience` stop, which has
        the same shape on disk and the opposite meaning.
        """
        kept = {}
        for row in rows:
            if row.population != population or row.void:
                continue
            verdict = self.convergence.verdict(RunMetrics(self.models / f"{row.name}.metrics.jsonl"))
            if verdict.settled():
                kept[row.name] = row.ledger75
        return kept

    def isolating(self, first: dict[str, str], second: dict[str, str]) -> str | None:
        """The one leaf these two recipes differ in, or None if they differ in none or several.

        Schema drift BREAKS the pair rather than being tolerated: two recipes that are not even the same
        SHAPE are not isolating anything, whatever their shared keys say.
        """
        if (set(first) ^ set(second)) - NAMING_LEAVES:
            return None
        differing = [key for key in set(first) - NAMING_LEAVES if str(first[key]) != str(second[key])]
        return differing[0] if len(differing) == 1 else None

    def table(self, scores: dict[str, float], recipes: dict[str, dict[str, str]]) -> dict[str, list[KnobPair]]:
        """Every single-key pair among the quotable runs, grouped by the key it prices."""
        found: dict[str, list[KnobPair]] = {}
        for low, high in combinations(sorted(scores), 2):
            key = self.isolating(recipes[low], recipes[high])
            if key is None:
                continue
            found.setdefault(key, []).append(
                KnobPair(
                    key=key,
                    low_name=low,
                    high_name=high,
                    low_value=str(recipes[low][key]),
                    high_value=str(recipes[high][key]),
                    low_score=scores[low],
                    high_score=scores[high],
                )
            )
        return found

    def lines(self, table: dict[str, list[KnobPair]]) -> list[str]:
        """The table as text, keys with the most pairs first and each key's largest |delta| at its top."""
        rendered = []
        for key in sorted(table, key=lambda name: (-len(table[name]), name)):
            pairs = sorted(table[key], key=lambda pair: -abs(pair.delta()))
            clearing = sum(pair.resolved() for pair in pairs)
            rendered.append(f"== {key}  ({len(pairs)} pairs, {clearing} clearing {PAIRED_FLOOR})")
            rendered.extend(pair.line() for pair in pairs[:SHOWN])
        return rendered


def main() -> None:
    """Price every leaf of every resolved recipe against every other, on one holdout population."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True, help="directory of .holdout.npz records")
    command.parser.add_argument("--declines", type=Path, default=DECLINES, help="what deliberately does not ship")
    command.parser.add_argument("--key", nargs="+", default=[], help="report only these leaves")
    arguments = command.parsed()

    models = command.rooted(arguments).models()
    pool = MemberPool.at(arguments.holdouts, arguments.declines)
    rows = LedgerSeventyFive(declines=pool.declines).table(
        [arguments.holdouts / f"{name}{SUFFIX}" for name in pool.names([])]
    )
    Denominator(len(rows), f"no member priced under {arguments.holdouts}").refuse_empty()

    # `table()` already leads with the largest population, which is the one every shipping number in this
    # repo is quoted on. Taking it from the data rather than naming a digest keeps this readable on a
    # recut corpus, where a hardcoded one would silently select nothing.
    population = rows[0].population

    pairs = RecipePairs(models)
    recipes = pairs.recipes()
    scores = {name: score for name, score in pairs.quotable(rows, population).items() if name in recipes}
    Denominator(len(scores), f"no finished, non-void run on population {population} carries a recipe").refuse_empty()

    table = pairs.table(scores, recipes)
    if arguments.key:
        table = {key: found for key, found in table.items() if key in arguments.key}

    total = sum(len(found) for found in table.values())
    command.report(
        [
            f"{len(recipes)} resolved recipes; {len(scores)} finished, non-void and priced on population {population}",
            f"{total} single-key isolating pairs across {len(table)} leaves",
            *pairs.lines(table),
        ]
    )


if __name__ == "__main__":
    main()
