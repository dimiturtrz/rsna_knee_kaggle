"""Does a deployed `_ff` member differ from its validated twin ONLY on fold? `python -m
kneemri.experiments.twin_check`.

WHAT IS VALIDATED IS NOT WHAT IS SHIPPED. A fold-0 run holds out a fifth of the corpus and earns the
number this project quotes; its `_ff` twin refits the same recipe on everything (`--fold -1`) and is
what a kernel actually loads. That trade is deliberate and priced. What is NOT priced is anything ELSE
that differs between the two, because no gate compared them: the arms are separate shell scripts, and a
flag added to one and not the other is invisible everywhere a human looks.

THE RULE. Across every axis `member_record` stamps — the `options` block and the `config` block — an
`_ff` checkpoint and its fold-0 twin may differ on the axes named in `BY_DESIGN` and on nothing else.
Anything further is a second, unread variable at a site the arm author never reads.

AN ABSENT KEY IS NOT A DIFFERENCE WHEN THE OTHER SIDE HOLDS THAT FLAG'S DECLARED DEFAULT. The two
halves of a pair are usually months apart, so the later one carries flags the earlier one predates. On
first run that read as 28 distinct drifting axes over 11 pairs — `layer_decay`, `through_plane`,
`hold_annotations`, `accumulate` and twenty-five more — and every one of them was `None` against that
flag's own default. The defaults are READ, from `DefaultDrift.declared()`, never listed here:
`OFF_VALUES` in `run_comparison` covers `False/0/0.0/None/""` and cannot express `layer_decay=1.0`,
`accumulate=1` or `slice_regions=1`, and a hand-written identity list is exactly the heuristic that
would one day absorb a real difference.

WHAT IT FOUND. All 11 pairs on disk differ on fold, ema and nothing else, so the axis this was filed
against is clean and the remaining exposure is `ema` alone — which is in `BY_DESIGN` because `FIT`
sets it on every `--fold -1` arm, and is the one entry there that buys nothing. Its price is measured,
not assumed: `ema_dense` against `band_dense` is a one-variable pair reading +0.0010 midpoint and
-0.0020 on ledger75 — inside the 0.0062 paired floor, with the two rulers disagreeing on sign. `--strict`
drops it from the allowance, which is how this check graduates the day `FIT` stops setting it.
"""

import logging
from pathlib import Path

from core import cli
from kneemri.experiments.default_drift import DefaultDrift
from kneemri.runs.member_record import StampedRun

logger = logging.getLogger(__name__)

FF_SUFFIX = "_ff.pt"
TWIN_SUFFIX = "_s2026.pt"

BY_DESIGN = {
    "fold": "the axis the twin exists to change: -1 fits the whole corpus, 0 holds a fifth out",
    "out": "where the member is written, which differs for any two runs (run_comparison.NOT_AN_AXIS)",
    "vram_gb": "how much of the card the run was allowed (run_comparison.NOT_AN_AXIS)",
    "witness": "a per-checkpoint fingerprint; two checkpoints sharing one would be the defect",
    "ema": "FIT sets it on every --fold -1 arm. Measured at +0.0010 midpoint / -0.0020 ledger75 "
    "against band_dense, inside the 0.0062 floor and with the rulers disagreeing on sign. "
    "Dropped by --strict",
}
"""Each axis a pair may differ on, with why. Everything NOT named here is a difference somebody has to
justify, which is the whole value of spelling the allowance out rather than inferring it."""

STRICT_DROPS = "ema"
"""The one entry `--strict` removes: it is the rider, not the trade, and it is what this was filed on."""


class TwinDrift:
    """Every `_ff` member on disk beside the fold-0 twin it was refitted from."""

    def __init__(self, models: Path, *, strict: bool = False) -> None:
        self.models = models
        self.allowed = {axis for axis in BY_DESIGN if not (strict and axis == STRICT_DROPS)}

    def pairs(self) -> list[tuple[Path, Path]]:
        """Each deployed checkpoint with its validated twin, matched by name."""
        found = []
        for deployed in sorted(self.models.glob(f"*{FF_SUFFIX}")):
            twin = self.models / f"{deployed.name[: -len(FF_SUFFIX)]}{TWIN_SUFFIX}"
            if twin.exists():
                found.append((deployed, twin))
        return found

    def drift(self, deployed: Path, twin: Path) -> list[str]:
        """Every axis this pair disagrees on beyond the allowance, each as `axis=twin->deployed`."""
        shipped, validated = StampedRun.of(deployed), StampedRun.of(twin)
        if shipped is None or validated is None:
            return [f"{deployed.stem} or its twin carries no stamp, so nothing here can compare them"]
        defaults = DefaultDrift.declared()
        # KEYED, BECAUSE AN AXIS IS STAMPED TWICE. `ema` and `through_plane` sit in both the options and
        # the config block, and a list reports each of them as two findings on one variable.
        found: dict[str, str] = {}
        for theirs, ours in ((shipped.options, validated.options), (shipped.config, validated.config)):
            for axis in sorted(set(theirs) | set(ours)):
                here, there = theirs.get(axis), ours.get(axis)
                if axis in self.allowed or here == there:
                    continue
                # ONE SIDE PREDATES THE FLAG. A key absent from one stamp while the other holds that
                # flag's declared default is one era, not two models.
                if (here is None or there is None) and (here or there) == defaults.get(axis):
                    continue
                found[axis] = f"{axis}={there!r}->{here!r}"
        return list(found.values())

    def faults(self) -> list[str]:
        """Each pair carrying an unread second variable, named with the axes that carry it."""
        return [
            f"{deployed.stem} differs from its validated twin on {', '.join(drifted)} — "
            f"a fold swap may not carry another axis"
            for deployed, twin in self.pairs()
            for drifted in [self.drift(deployed, twin)]
            if drifted
        ]

    def report(self) -> list[str]:
        """The survey, whether or not anything drifted."""
        allowance = ", ".join(sorted(self.allowed))
        surveyed = f"{len(self.pairs())} deployed/validated pair(s) compared; may differ on {allowance}"
        return [*self.faults(), surveyed]


def main() -> None:
    """Assert every deployed member differs from what was validated on fold alone."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument(
        "--strict",
        action="store_true",
        help=f"drop {STRICT_DROPS!r} from the allowance — how this graduates once FIT stops setting it",
    )
    arguments = command.parsed()

    drift = TwinDrift(command.rooted(arguments).models(), strict=arguments.strict)
    command.report(drift.report())
    command.conclude(drift.faults(), "deployed member(s) carry an axis nobody validated")


if __name__ == "__main__":
    main()
