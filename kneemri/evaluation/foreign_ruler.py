"""A foreign model and ours on ONE ruler: `python -m kneemri.evaluation.foreign_ruler`.

WHY IT EXISTS. `knee-rh4t` priced a ~0.017 member gap by putting our BOARD single (`soloseat` 0.925)
beside rivals' FORUM claims (0.938-0.942). Those are two rulers over two study sets, and nothing joined
them; `knee-urvf` then built on the gap, and both are P0s that have steered what got queued. A published
foreign checkpoint that self-reports per-finding AUC **on the 58 radiologist-annotated studies** can be
joined to our own `annotated_predictions` cache, because our `*_hold` members withheld exactly those 58
and the foreign corpus (4,349 = 4,407 - 58) excludes them too. Both sides held out the same studies and
answered the same radiologists. That is one ruler, and it costs no card, no slot and no inference.

WHAT IT REFUSES TO DO, WHICH IS THE POINT OF THE MODULE. A twelve-row per-finding table over 58 studies
invites exactly one mistake: reading down the delta column, keeping the rows where the foreign model wins,
and quoting the resulting per-label ORACLE as the prize. `oracle` computes that number AND simulates what
the SAME arithmetic returns when both models are equally good and differ only by sampling noise, so the
oracle is never printed without its own null beside it.

The null is compared on its QUANTILE, not its mean, and that distinction is the whole gate. Routing's
baseline is the BETTER single model, which is selected on the same noise, so a hand estimate of the
inflation as `SE/sqrt(pi)` over-states it -- part of the lift cancels. Simulated properly the null came
out BELOW the gain observed on 2026-08-27 (+0.0163 against +0.0188), which on a mean threshold reads as
a finding; 33.9% of the null's own draws reach that gain, which is what it actually is. Testing against
the mean called four of the six foreign checkpoints winners. Testing against the quantile calls all six
noise.

WHAT IT CANNOT DO. A foreign checkpoint publishes AUCs, not per-study probabilities, so there is no paired
bootstrap against it -- `table_separation`'s interval needs both arms' cells and only one arm has them.
Per-finding deltas are therefore bracketed by the Hanley-McNeil standard error of a SINGLE arm, which is
the conservative direction (it ignores the pairing that would tighten it) and is quoted as such. Our own
members ARE mutually pairable and that comparison belongs in `table_separation`, not here.

AND IT IS A HYPOTHESIS GENERATOR, NOT A VERDICT. 58 studies resolve a macro difference (twelve findings
average the per-label error down to ~0.012) and resolve nothing per finding (0.031-0.072). Use the macro
to answer "is their member better"; use the per-finding column only to CHOOSE what to measure next on the
909-study holdout, where the same question can actually be settled.
"""

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.protocol_prior import POSITIVE, Ordering
from kneemri.evaluation.supervision_gap import SupervisionGap

logger = logging.getLogger(__name__)

ORACLE_DRAWS = 20000
"""Simulated corpora behind the oracle's null. Large enough that its own standard error is invisible
against the 0.001 the report prints, and cheap because each draw is twelve normals rather than a corpus."""

ORACLE_SEED = 2026
"""Seeded so the null a decision is taken against is the null a reader reproduces."""

CHANCE_FLOOR = 0.05
"""Above this share of null draws reaching the gain, noise selection is not ruled out and the oracle
is printed as noise. One-sided and conventional; the report prints the share itself, so a reader who
wants a different floor has the number rather than the verdict."""


class StudyOrderError(Exception):
    """A cached prediction block whose rows are not the study list it is being scored against."""


@dataclass(frozen=True)
class ForeignReading:
    """One published foreign checkpoint's self-reported per-finding AUC on the annotated studies.

    BESIDE ITS PRODUCER: the two facts that make a foreign row joinable are the study set it was scored
    on and the finding order its values are keyed by, and both are properties of the checkpoint file.
    """

    name: str
    macro: float
    per_finding: Float[np.ndarray, " targets"]

    @staticmethod
    def load(checkpoint: Path) -> "ForeignReading | None":
        """Read a checkpoint's `aucs` without executing its pickle; `None` if it publishes none.

        `weights_only=True` is not politeness. A foreign `.pt` is an untrusted archive, and the whole
        argument for using one at all is that reading its numbers never runs its code.
        """
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        published = payload.get("aucs")
        if published is None:
            return None
        values = np.array([float(published[target]) for target in TARGETS], dtype=np.float64)
        return ForeignReading(checkpoint.stem, float(values.mean()), values)


@dataclass(frozen=True)
class OracleNull:
    """The per-finding oracle beside what the same arithmetic returns on two equally-good models.

    Beside its producer rather than a bare tuple: a gain and the gain chance manufactures are the same
    kind of number and are only ever read together, and separating them is how the prize gets quoted.
    """

    gain: float
    expected: float
    chance: float
    """Share of null draws reaching the observed gain -- the number that keeps a 0.002 margin honest."""

    @property
    def explained(self) -> bool:
        """Whether noise selection alone accounts for the oracle, which is the verdict the caller wants.

        Tested on the null's QUANTILE, not against its mean. A gain a hair above the mean of a null whose
        own scatter is several times that hair is not a finding, and comparing to the mean would call it
        one -- which is the single mistake this module exists to refuse.
        """
        return self.chance >= CHANCE_FLOOR


@dataclass(frozen=True)
class ForeignRuler:
    """Our cached annotated predictions and a foreign checkpoint's published AUCs, per finding."""

    ordering: Ordering = field(default_factory=Ordering)

    def foreign(self, directory: Path) -> dict[str, ForeignReading]:
        """Every checkpoint under `directory` that publishes per-finding AUCs, keyed by file stem."""
        found = (ForeignReading.load(path) for path in sorted(directory.rglob("*.pt")))
        return {reading.name: reading for reading in found if reading is not None}

    def ours(
        self, cache: Path, identifiers: list[str], annotations: Float[np.ndarray, "studies targets"]
    ) -> dict[str, Float[np.ndarray, " targets"]]:
        """Each cached member's per-finding AUC against the annotations, keyed by checkpoint stem.

        The cache stores one `.npy` of probabilities beside a `.json` naming the checkpoint that wrote
        it, and the rows are in the study order that manifest's `studies` digest was taken over. That
        digest is CHECKED rather than assumed: predictions read against a different row order score at
        chance, and chance is a number this module would print rather than raise on.
        """
        digest = hashlib.sha256("\n".join(identifiers).encode()).hexdigest()
        readings: dict[str, Float[np.ndarray, " targets"]] = {}
        for manifest in sorted(cache.glob("*.json")):
            written = json.loads(manifest.read_text(encoding="utf-8"))
            if written["studies"] != digest:
                raise StudyOrderError(f"{manifest.name} was scored over a different study list")
            values = np.load(manifest.with_suffix(".npy"))
            readings[Path(written["checkpoint"]).stem] = self.against(values, annotations)
        return readings

    def against(
        self, values: Float[np.ndarray, "studies targets"], annotations: Float[np.ndarray, "studies targets"]
    ) -> Float[np.ndarray, " targets"]:
        """One member's twelve AUCs, in `TARGETS` order."""
        columns = values.reshape(values.shape[0], -1)[:, : len(TARGETS)]
        return np.array([self.ordering.auc(columns[:, index], annotations[:, index]) for index in range(len(TARGETS))])

    def standard_error(self, auc: float, positives: int, negatives: int) -> float:
        """Hanley-McNeil for ONE arm, the conservative bracket a foreign row can carry.

        Unpaired on purpose: the pairing that would tighten this needs the foreign model's per-study
        probabilities, and a published checkpoint ships AUCs instead.
        """
        first = auc / (2.0 - auc)
        second = 2.0 * auc * auc / (1.0 + auc)
        variance = auc * (1.0 - auc) + (positives - 1) * (first - auc * auc) + (negatives - 1) * (second - auc * auc)
        return math.sqrt(variance / (positives * negatives))

    def oracle(
        self,
        ours: Float[np.ndarray, " targets"],
        theirs: Float[np.ndarray, " targets"],
        errors: Float[np.ndarray, " targets"],
        draws: int = ORACLE_DRAWS,
    ) -> OracleNull:
        """Per-finding routing's gain over the better single model, beside chance's own version of it.

        The null draws two equally-good models -- both findings' true AUC set to the pair's own mean, so
        neither is favoured -- perturbs each by its finding's standard error, and routes on the result.
        Whatever that returns is what routing buys when there is nothing to route.
        """
        gain = float(np.maximum(ours, theirs).mean() - max(ours.mean(), theirs.mean()))
        truth = (ours + theirs) / 2.0
        generator = np.random.default_rng(ORACLE_SEED)
        noise = generator.normal(scale=errors, size=(draws, 2, len(errors)))
        drawn = truth + noise
        routed = drawn.max(axis=1).mean(axis=1) - drawn.mean(axis=2).max(axis=1)
        return OracleNull(gain, float(routed.mean()), float((routed >= gain).mean()))

    def report(
        self,
        ours: Float[np.ndarray, " targets"],
        theirs: ForeignReading,
        annotations: Float[np.ndarray, "studies targets"],
        *,
        name: str,
    ) -> list[str]:
        """The per-finding table, the macro that 58 studies can actually resolve, and the oracle's null."""
        positive = (annotations > POSITIVE).sum(axis=0).astype(int)
        studies = annotations.shape[0]
        errors = np.array(
            [
                self.standard_error(theirs.per_finding[i], int(positive[i]), studies - int(positive[i]))
                for i in range(len(TARGETS))
            ]
        )

        lines = [f"{studies} annotated studies -- {name} against {theirs.name}", ""]
        lines.append(f"{'finding':22s} {'n+':>3s} {'ours':>7s} {'theirs':>7s} {'delta':>8s} {'HM se':>7s}")
        for index, target in enumerate(TARGETS):
            delta = theirs.per_finding[index] - ours[index]
            mark = " " if abs(delta) < errors[index] else "*"
            lines.append(
                f"{target:22s} {positive[index]:3d} {ours[index]:7.4f} {theirs.per_finding[index]:7.4f} "
                f"{delta:+8.4f} {errors[index]:7.4f}{mark}"
            )
        macro = float(errors.mean() / math.sqrt(len(TARGETS)))
        lines.append("")
        lines.append(
            f"{'MACRO':22s} {'':3s} {ours.mean():7.4f} {theirs.macro:7.4f} "
            f"{theirs.macro - ours.mean():+8.4f} {macro:7.4f}"
        )

        null = self.oracle(ours, theirs.per_finding, errors)
        verdict = "NOISE SELECTION -- not a prize" if null.explained else "survives its own null"
        lines.append(
            f"per-finding oracle {null.gain:+.4f} against a null of {null.expected:+.4f} "
            f"(chance reaches it {null.chance:.1%} of the time): {verdict}"
        )
        return lines


def main() -> None:
    """Rank our cached annotated members, then read the best one against each foreign checkpoint."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--cache", type=Path, required=True, help="the annotated_predictions directory")
    command.parser.add_argument("--foreign", type=Path, required=True, help="a directory of foreign .pt files")
    command.parser.add_argument("--train", type=Path, required=True, help="train.csv, which carries the gold rows")
    command.parser.add_argument(
        "--member",
        default=None,
        help="read THIS member instead of the best one on these 58 studies. knee-3uje: the maximum over "
        "our cached members is chosen on the same 58 it is then scored on, so the macro it reports "
        "carries a selection term that the foreign checkpoint's single self-reported number does not. "
        "Naming a member fixed in advance -- by the 909-study holdout, or by anything but this set -- "
        "is the only way to tell a selection effect from a ruler effect.",
    )
    arguments = command.parsed()

    ruler = ForeignRuler()
    identifiers, annotations = SupervisionGap().annotated(arguments.train)
    ours = ruler.ours(arguments.cache, identifiers, annotations)
    lines = ["our cached members, macro on the annotated studies:"]
    lines.extend(
        f"  {member:28s} {values.mean():.4f}"
        for member, values in sorted(ours.items(), key=lambda item: -item[1].mean())
    )

    chosen = arguments.member or max(ours, key=lambda member: ours[member].mean())
    if chosen not in ours:
        message = f"{chosen} is not in the cache; it holds {sorted(ours)}"
        raise SystemExit(message)
    lines.append(
        f"reading {chosen}, "
        + ("named in advance" if arguments.member else "SELECTED ON THESE 58 -- the macro below is a maximum")
    )
    for reading in ruler.foreign(arguments.foreign).values():
        lines.extend(["", *ruler.report(ours[chosen], reading, annotations, name=chosen)])
    command.report(lines)


if __name__ == "__main__":
    main()
