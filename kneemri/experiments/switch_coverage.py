"""Every training switch is armed, smoke-covered, or declared not a recipe: `arm_check --switches`.

`arm_check --smoke` proves that every committed recipe survives a real epoch — which is only true of
the paths the ARMS take. Measured 2026-08-18 by diffing `TrainConfig` against every `--flag` under
`experiments/**.sh`: five real switches were landed-checked, stamped, and never once RUN (`--pool`
focal, `--anatomical-prior`, `--normalise-by-count`, `--tie`, `--head-lr`). A break in one of those
surfaces at the queue, after the card is held, which is the failure the whole check family exists to
retire.

So the partition is made TOTAL and asserted: an option is exercised by an arm, or carries a recipe here
that the same smoke machinery runs, or is written down as not being a recipe at all with the reason. A
new knob then forces that decision at the unit gate rather than at a queue slot.

The recipes are `{name: flags}`, the same shape `ArmCheck.invocations()` returns, so the cache resolver,
the truncated decode and the smoke runner all apply unchanged — `--tie last` names a cache no full corpus
has, and `decode_smoke_caches` builds the sixty-study one for it without knowing this module exists.
"""

import re
import tomllib
from pathlib import Path

from kneemri.training.config_tree import TrainConfig

SCRIPTS = Path("experiments")

NOT_A_RECIPE = {
    "paths": "the data root, machine-specific and gitignored",
    "labels": "the label table every arm passes through lib.sh",
    "out": "where a run writes, set per invocation",
    "config": "a recipe file, whose keys are the fields it sets and are accounted for under their own names",
    "epochs": "run length, overridden to 1 by every smoke",
    "studies": "corpus truncation, which IS the smoke",
    "cache_studies": "the truncated cache the smoke decodes, set by the checker not by a recipe",
    "seed": "a seed is not a method; folds and seeds are banned as arms here",
    "split_seed": "same, and it moves the split rather than the recipe",
    "folds": "banned as an arm for the same reason",
    "fold": "which split half a run holds out, chosen per invocation",
    "band_low": "a cache axis the builder sets, read back by the run",
    "band_high": "the other end of that same axis",
    "spacing": "a cache axis: how the slices sit inside that band",
    "bf16": "the default precision every current arm trains under",
    "vram_gb": "a memory cap, set by the queue against the card it runs on",
    "target_corpus": "the companion to --corpus-oversample: it is exercised by the same coverage recipe",
}
"""Options that cannot be exercised as a recipe, each with WHY — a decision record, not an exemption list.

Written down rather than inferred, because 'no arm passes it' is true both of a knob nobody has tested
and of a knob that is not a method at all, and only one of those is a hole.
"""

COVERAGE = {
    "pool": "--pool cls_mean_focal",
    "anatomical_prior": "--anatomical-prior",
    "normalise_by_count": "--normalise-by-count",
    "tie": "--tie last",
    "head_lr": "--head-lr 3e-4",
    "side_weight": "--side-weight 0.5",
    "certainty_ramp": "--certainty-ramp 6",
    "hard_start": "--hard-start 0 --hard-ramp 0.5",
    "hard_mid": "--hard-mid 0.5",
    "hard_end": "--hard-end 0.5 --hard-tail 0.5",
    "hard_ramp": "--hard-start 0.5 --hard-ramp 1.0",
    "hard_tail": "--hard-end 0 --hard-tail 0.25",
    "corpus_oversample": "--corpus-oversample 3.0 --target-corpus canonical",
    "padding_mode": "--padding-mode reflection",
    "patience_from_progress": "--patience 2 --patience-from-progress 0.5",
    "hold_annotation_studies": "--hold-annotations --hold-annotation-studies",
    "pseudo_fold": "--pseudo-fold 3",
    "early_weight": "--early-weight 3.0",
    "early_beta": "--early-weight 3.0 --early-beta 0.5",
}
"""One smoke recipe per switch no arm selects, holding every other flag at its default.

Each was run on 2026-08-18 at sixty studies and one epoch: all five reach a written checkpoint. `--tie
last` needed its own sixty-study decode, which is the point of routing these through `arm_check` rather
than a script — the router is a CACHE axis, so the only arm that could exercise it is one nothing has
decoded for.
"""


class SwitchCoverage:
    """Which training options an arm exercises, which a recipe here covers, and which nothing does."""

    def __init__(self, scripts: Path = SCRIPTS) -> None:
        self.scripts = scripts

    def armed(self) -> set[str]:
        """The option names some committed arm or script actually passes on a command line.

        Scraped from the shell rather than declared, for the reason the config walk is scraped: a hand
        list of what the arms pass is a second copy of the arms, free to disagree with them.
        """
        text = "\n".join(path.read_text(encoding="utf-8") for path in self.scripts.rglob("*.sh"))
        passed = {flag.removeprefix("--").replace("-", "_") for flag in re.findall(r"--[a-z0-9-]+", text)}
        return {leaf.name for leaf, _ in TrainConfig.leaves() if leaf.name in passed | self.recipe_keys()}

    def recipe_keys(self) -> set[str]:
        """The leaf names some recipe layer sets, which is a flag passed by a file instead of a line."""
        layers = [tomllib.loads(path.read_text(encoding="utf-8")) for path in self.scripts.rglob("*.toml")]
        return {
            key
            for layer in layers
            for node, table in layer.items()
            if node != "meta" and isinstance(table, dict)
            for key in table
        }

    def unaccounted(self) -> set[str]:
        """Options no arm passes, no recipe here covers, and nobody has written off. Must stay empty."""
        return {leaf.name for leaf, _ in TrainConfig.leaves()} - self.armed() - set(COVERAGE) - set(NOT_A_RECIPE)

    @staticmethod
    def labels(printed: dict[str, str]) -> str:
        """The `--labels` an ARM prints, taken from the arms rather than spelled again here.

        A coverage recipe is a train invocation and needs the label table like any other, and that path
        lives in `lib.sh`. Copying it would put a fourth spelling of the same file in the tree, free to
        drift from the one the arms actually run — and the stub fabricates a different root under CI, so
        a hardcoded one would be wrong there by construction. Empty when no arm printed one, which is
        the caller's signal that there is nothing to smoke against.
        """
        for flags in printed.values():
            words = flags.split()
            if "--labels" in words:
                return " ".join(words[words.index("--labels") : words.index("--labels") + 2])
        return ""

    def invocations(self, printed: dict[str, str]) -> dict[str, str]:
        """The coverage recipes, named as the smoke reports them, on the arms' own label table."""
        labels = self.labels(printed)
        return {f"switch_{name}": f"{labels} {flags}".strip() for name, flags in COVERAGE.items()}
