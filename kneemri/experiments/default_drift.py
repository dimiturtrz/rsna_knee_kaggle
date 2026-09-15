"""Which of a finished run's settings the trainer would no longer give it: `python -m
kneemri.experiments.default_drift --metrics <run>.metrics.jsonl`.

WHY IT EXISTS. A one-knob arm is written by copying a parent's flag line and adding one flag, and that
is only a one-knob arm if every setting NEITHER script states is still what the parent got. It is not.
`band_dense_hold -> relabel_hold` was written as one flag and its two recorded recipes differ on SEVEN
keys — `sample_alpha`, `warmup`, `weight_decay`, `init_scope`, `target_corpus`, `spacing` were DEFAULTS
whose values moved between the parent's run and the child's, and the eighth was the flag. A default that
changes is invisible in both scripts by construction: neither one mentions it, so a diff of the two
scripts is silent and the arm reads as clean right up until its own recorded row is compared to its
parent's, which happens after the card has been spent.

WHAT IT COMPARES, and the reason it is the recorded row and not the other script. `TrainConfig`
holds the defaults the trainer would apply TODAY; the parent's `metrics.jsonl` row 0 holds the values it
actually parsed. Every key where those two disagree is a key the child will silently receive at the new
value unless its arm passes it explicitly at the parent's. That is the pin list, and it is computable on
the CPU in a second — before the card, rather than fifteen minutes into it.

WHAT IT IS NOT. It says nothing about whether the child's flag line is correct; `arm_check` reads flags
and `arm_diff` reads one arm against another. It also cannot see a default that moved and moved BACK, or
a change in what a value MEANS, which no comparison of strings can reach. An empty report is the normal
answer for a parent that ran recently, and saying so out loud is the point — "no drift" is a measurement,
not an absence of one.

`out`, `labels` and `paths` are dropped: every arm sets them differently by construction, so their
disagreement with a default is not drift.
"""

import json
import logging
from collections.abc import Mapping
from pathlib import Path

from core import cli
from kneemri.training.config_tree import TrainConfig

logger = logging.getLogger(__name__)

INCIDENTAL = frozenset({"out", "labels", "paths"})
"""Keys every run states for itself, so a difference from the default carries no information."""


class DefaultDrift:
    """One finished run's recorded settings against the defaults the trainer would apply now."""

    def __init__(self, recipe: Mapping[str, str], incidental: frozenset[str] = INCIDENTAL) -> None:
        self.recipe = recipe
        self.incidental = incidental
        self.current = TrainConfig().recipe

    @classmethod
    def of_metrics(cls, metrics: Path) -> "DefaultDrift":
        """The drift of the run that wrote this `metrics.jsonl`, read from its first line.

        Row 0 is the recipe row — the trainer logs it before the corpus is mapped, so it exists for a
        run that died in epoch 1 and for one that never wrote a checkpoint at all.
        """
        first = metrics.read_text(encoding="utf-8").splitlines()[0]
        recipe = json.loads(first).get("recipe", {})
        return cls(recipe)

    def drifted(self) -> dict[str, tuple[str, str]]:
        """Key -> (what the run got, what it would get today), for every key that moved underneath it.

        A key the recorded row does not carry is SKIPPED rather than reported: the ledger spans renamed
        and added flags, and a field that did not exist when the parent ran did not drift, it arrived.
        Pinning an arrival at "absent" is not possible and reporting it as drift would bury the real
        rows — `exclude_fold` alone would appear in every run recorded before the key existed.
        """
        return {
            key: (recorded, self.current[key])
            for key, recorded in self.recipe.items()
            if key in self.current and key not in self.incidental and self.current[key] != recorded
        }


def main() -> None:
    """Print the keys a child of this run must pass explicitly to inherit its parent's recipe."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--metrics", type=Path, required=True, help="a run's <name>.metrics.jsonl")
    arguments = command.parsed()

    drifted = DefaultDrift.of_metrics(arguments.metrics).drifted()
    lines = [
        f"{key:24s} recorded {recorded!r:>20s}  default now {current!r}" for key, (recorded, current) in drifted.items()
    ]
    lines.append(f"{len(drifted)} key(s) drifted since {arguments.metrics.name} ran - pin each at its recorded value")
    command.report(lines)


if __name__ == "__main__":
    main()
