"""Every one-variable A/B this project has ALREADY run: `python -m kneemri.evaluation.matched_axes`.

WHAT IT DOES. Group every member carrying a recipe on ALL recipe keys except one. Any group holding two
distinct values of the held-out key is a matched A/B on that key alone — a controlled comparison that
cost a card arm at the time and costs nothing to read now. 397 members carry a recipe, and they resolve
into a few hundred such comparisons across the flags the project owns. Nothing here touches the GPU.

WHY IT EXISTS. An axis priced from ONE pair sits under the 0.0053 paired floor and cannot be told from
scatter, which is how a knob gets planned around. Pooling every matched group on an axis is the only
extra power available without spending wall-clock, and sign agreement across independent groups is
evidence a single delta cannot give.

POOL PER VALUE-PAIR, NEVER PER AXIS — the mistake this tool exists to stop making. Orienting groups by
an axis's most common value silently mixes contrasts that are different questions, and on first use it
manufactured two levers that a look at the pairs destroyed. `stochastic_depth` pooled to +0.0035
favouring 0.05 across two groups; those groups are 0.05-vs-0.1 (+0.0067, which says do not OVERDO it)
and 0.05-vs-0.0 (+0.0003, the add-it-or-not question, and it is nothing). `epochs` mixed 24-vs-12 with
12-vs-48 and came out flat, hiding that longer helps to 24 (+0.0035, n=5) and hurts by 48 (-0.0079).
`slice_pooling` signed a `moments_wide`-vs-`moments` group as though it were labelwise-vs-max. A
contrast is (key, one value, the other value); this module keys on exactly that.

THREE GUARDS, each for a way a group lies. POPULATION: an AUC is a statistic of a study set, so two rows
read on different sets are different questions and their difference is not a delta (`Ledger75Row.
population`). VOID: a member ruled in-sample by `comp/members.declines` has no held-out reading to
contribute. DEAD RUNS: a member far below the pool is a crashed or diverged arm, and pairing it against
a healthy twin reads as a colossal axis effect — the largest deltas on the first sweep were +0.17 and
+0.13, every one of them a death rather than a flag.

WHAT IT CANNOT SEE. An axis with no matched pair on disk is invisible here, not absent. Every reading is
a MAIN effect — two flags that only pay off together appear as two nulls. And a contrast is only as
controlled as the recipe record: two runs identical in every logged key still differ in the data they
happened to draw.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import pstdev

from core import cli
from kneemri.evaluation.ledger75 import Ledger75Row, LedgerSeventyFive
from kneemri.evaluation.member_pool import DECLINES, MemberPool
from kneemri.runs.holdout_record import SUFFIX

logger = logging.getLogger(__name__)

RECIPE_NOISE = frozenset({"out", "seed", "labels", "paths", "eval_labels", "config"})
"""Keys naming WHERE a run wrote or WHICH draw it took, never what it tried.

Leaving them in makes every member its own group and the sweep returns nothing. `seed` is here because a
seed difference is the noise floor itself, not an axis; a deliberate seed study is read with `--only`.
"""

DEAD_RUN_FLOOR = 0.86
"""Below this a member is a crashed or diverged arm rather than a reading of its recipe.

Legislated, not seeded from the pool: the project's shipping band is 0.88-0.91 and the paired floor is
0.0053, so 0.86 is more than three floors below anything that has ever been a candidate. Members under
it exist (0.67, 0.75, 0.77) and every one inspected was a death.
"""

MIN_GROUPS = 2
"""A contrast needs two independent matched groups before it is pooled at all — one is the floor problem
this module exists to get past, and it is already printed by the per-group listing."""

METRICS_SUFFIX = ".metrics.jsonl"


@dataclass(frozen=True)
class AxisContrast:
    """One recipe key at two specific values, pooled over every matched group that ran it."""

    key: str
    favoured: str
    against: str
    deltas: tuple[float, ...]
    """Signed toward `favoured`, one per matched group."""

    @property
    def mean(self) -> float:
        return sum(self.deltas) / len(self.deltas)

    @property
    def spread(self) -> float:
        return pstdev(self.deltas) if len(self.deltas) > 1 else 0.0

    @property
    def agreement(self) -> int:
        """How many of the groups moved the way the pooled mean does."""
        return sum(1 for delta in self.deltas if delta > 0)

    @classmethod
    def oriented(cls, key: str, first: str, second: str, deltas: list[float]) -> "AxisContrast":
        """Point the contrast at whichever value the pool favours, so a printed mean is never negative."""
        if sum(deltas) >= 0:
            return cls(key=key, favoured=first, against=second, deltas=tuple(deltas))
        return cls(key=key, favoured=second, against=first, deltas=tuple(-delta for delta in deltas))

    def line(self) -> str:
        contrast = f"{self.favoured[:14]} > {self.against[:14]}"
        return (
            f"{self.key:22s}{contrast:34s}{len(self.deltas):3d}{self.mean:+9.4f}"
            f"{self.spread:8.4f}{self.agreement:5d}/{len(self.deltas)}"
        )


@dataclass(frozen=True)
class MatchedAxes:
    """The pool's recipes and readings, indexed so a held-out key resolves to its matched groups."""

    recipes: dict[str, dict[str, str]]
    rows: dict[str, Ledger75Row]

    @classmethod
    def at(cls, holdouts: Path, declines: Path = DECLINES) -> "MatchedAxes":
        pool = MemberPool.at(holdouts, declines)
        ruler = LedgerSeventyFive(declines=pool.declines)
        recipes: dict[str, dict[str, str]] = {}
        rows: dict[str, Ledger75Row] = {}
        for name in pool.names():
            recipe = cls.recipe_at(holdouts / f"{name}{METRICS_SUFFIX}")
            if recipe is None:
                continue
            try:
                row = ruler.row(holdouts / f"{name}{SUFFIX}")
            except FileNotFoundError:
                logger.debug("%s has no holdout record", name)
                continue
            if row.void or row.ledger75 < DEAD_RUN_FLOOR:
                continue
            recipes[name] = recipe
            rows[name] = row
        return cls(recipes=recipes, rows=rows)

    @staticmethod
    def recipe_value(recipe: dict[str, str], key: str) -> str:
        """The recorded value as a string, because a stamp is text and `24` and `'24'` are one value."""
        return str(recipe[key])

    @staticmethod
    def recipe_at(metrics: Path) -> dict[str, str] | None:
        """The recipe stamp a run wrote on its first metrics row, or None where there is no stamp."""
        try:
            with metrics.open(encoding="utf-8") as handle:
                recorded = json.loads(handle.readline()).get("recipe")
        except (OSError, json.JSONDecodeError):
            logger.debug("%s is unreadable as a metrics record", metrics)
            return None
        return {str(key): str(value) for key, value in recorded.items()} if recorded else None

    def keys(self) -> list[str]:
        return sorted({key for recipe in self.recipes.values() for key in recipe} - RECIPE_NOISE)

    def groups(self, key: str) -> list[list[str]]:
        """Members sharing every other key, split into the sets that differ only on `key`."""
        buckets: dict[tuple[tuple[str, str], ...], list[str]] = defaultdict(list)
        for name, recipe in self.recipes.items():
            if key not in recipe:
                continue
            rest = tuple(sorted((k, v) for k, v in recipe.items() if k != key and k not in RECIPE_NOISE))
            buckets[rest].append(name)
        return [
            members
            for members in buckets.values()
            if len({MatchedAxes.recipe_value(self.recipes[m], key) for m in members}) > 1
            and len({self.rows[m].population for m in members}) == 1
        ]

    def readings(self, members: list[str], key: str) -> dict[str, float]:
        """One LEDGER75 per VALUE of `key`, averaging the members a group happens to hold at that value.

        A group of three values is three questions, and taking only its best and worst member would
        answer none of them: min-minus-max over a group is the widest delta it contains by construction,
        so it inflates every pooled mean that a third arm was ever run for. Every value-pair is emitted
        instead, and a value carrying two seeds contributes its mean rather than entering twice.
        """
        at_value: dict[str, list[float]] = defaultdict(list)
        for name in members:
            at_value[MatchedAxes.recipe_value(self.recipes[name], key)].append(self.rows[name].ledger75)
        return {value: sum(readings) / len(readings) for value, readings in at_value.items()}

    def contrasts(self) -> list[AxisContrast]:
        """Every (key, value, value) comparison with at least `MIN_GROUPS` matched groups, best first."""
        pooled: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        for key in self.keys():
            for members in self.groups(key):
                read = self.readings(members, key)
                for first, second in combinations(sorted(read), 2):
                    pooled[key, first, second].append(read[first] - read[second])
        contrasts = [
            AxisContrast.oriented(key, first, second, deltas)
            for (key, first, second), deltas in pooled.items()
            if len(deltas) >= MIN_GROUPS
        ]
        return sorted(contrasts, key=lambda contrast: -abs(contrast.mean))


def main() -> None:
    """Pool every matched one-variable comparison on disk, per value-pair, under LEDGER75."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True, help="directory of .holdout.npz records")
    command.parser.add_argument("--declines", type=Path, default=DECLINES, help="what deliberately does not ship")
    command.parser.add_argument("--key", default="", help="one recipe key, listing its groups member by member")
    arguments = command.parsed()

    axes = MatchedAxes.at(arguments.holdouts, arguments.declines)
    if arguments.key:
        lines = [f"matched groups on `{arguments.key}`, LEDGER75:"]
        for members in axes.groups(arguments.key):
            ranked = sorted(members, key=lambda name: axes.rows[name].ledger75, reverse=True)
            values = {name: MatchedAxes.recipe_value(axes.recipes[name], arguments.key) for name in ranked}
            lines += [f"  {axes.rows[name].ledger75:.4f}  {arguments.key}={values[name]}  {name}" for name in ranked]
            lines.append("  --")
        command.report(lines)
        return

    contrasts = axes.contrasts()
    header = f"{'axis':22s}{'contrast':34s}{'n':>3s}{'mean':>9s}{'sd':>8s}{'agree':>7s}"
    command.report(
        [f"{len(axes.recipes)} members carry a recipe and a held-out reading", header]
        + [contrast.line() for contrast in contrasts]
        + [
            f"{len(contrasts)} contrast(s) with >= {MIN_GROUPS} matched groups. Every reading is a MAIN "
            f"effect on LEDGER75; the paired floor is 0.0053, so a mean under it is DIRECTIONAL."
        ]
    )


if __name__ == "__main__":
    main()
