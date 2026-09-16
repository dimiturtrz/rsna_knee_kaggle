"""The measurement that judges a learned combiner against the fixed rules, on studies none of them fitted.

It lives in `evaluation` and not beside `StackedLogit` because a rule may not import its own ruler:
`kneemri.inference -> kneemri.evaluation` is a forbidden arrow, and the arrow is right — a combiner that
could reach the scoring code is a combiner that can be tuned on it.

THE ONE THING THIS FILE IS FOR. A fitted combiner scored on the studies it fitted always wins, so every
number here comes from a half of the holdout the weights never saw, and the ridge penalty is chosen by a
NESTED split inside the fitting half. An earlier reading picked the penalty by looking at the scoring half
and reported that the gain doubled with pool size; under honest selection that claim vanished.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int
from numpy.typing import NDArray

from kneemri.evaluation.scores import MIDPOINT, LabelScores
from kneemri.inference.stacked import PENALTIES, STACK_PENALTY, StackedLogit
from kneemri.inference.vote import LogitAverage, RankAverage, Vote
from kneemri.runs.holdout_record import SUFFIX

FIT_SHARE = 0.5
"""Half the holdout fits, half scores. Half rather than a larger fitting share because the scoring half is
what carries the error bar, and at 909 studies a 0.0062 floor is already wider than every effect here."""

SPLIT_SEED = 0
"""The split is seeded so two readings of the same pool are comparable. This is not a seed EXPERIMENT — the
ban is on averaging over seeds to manufacture a result, not on making one measurement reproducible."""


@dataclass(frozen=True)
class MemberHoldout:
    """One member's predictions over the holdout studies, beside the truth they are judged against."""

    name: str
    studies: NDArray[np.str_]
    predictions: Float[NDArray[np.float64], "studies findings"]
    truth: Int[NDArray[np.int64], "studies findings"]

    @classmethod
    def loaded(cls, path: Path) -> "MemberHoldout":
        """Read a `<name>.holdout.npz`, binarising truth at the project's midpoint."""
        held = np.load(path, allow_pickle=False)
        return cls(
            # The suffix, not the first dot: a member named for a band carries one (`n12b0.1-0.9`).
            name=path.name.removesuffix(SUFFIX),
            studies=held["studies"],
            predictions=held["predictions"].astype(np.float64),
            truth=(held["truth"] >= MIDPOINT).astype(np.int64),
        )

    @staticmethod
    def aligned(members: list["MemberHoldout"]) -> list["MemberHoldout"]:
        """Cut every member to the studies they all hold, in one shared order.

        By study id, never by position: two members recorded over different holdouts line up perfectly
        under a positional read and the combination is then nonsense with no symptom.
        """
        shared = sorted(set.intersection(*[set(member.studies.tolist()) for member in members]))
        order = np.asarray(shared)
        cut = []
        for member in members:
            where = {study: index for index, study in enumerate(member.studies.tolist())}
            taken = np.asarray([where[study] for study in shared])
            cut.append(
                MemberHoldout(member.name, order, member.predictions[taken], member.truth[taken]),
            )
        return cut


@dataclass(frozen=True)
class StackResult:
    """One rule's number, carrying the study counts that make it readable."""

    rule: str
    fitted_on: int
    scored_on: int
    mean: float

    def line(self) -> str:
        """One printable row."""
        return f"{self.rule:<16} fit={self.fitted_on:>4} score={self.scored_on:>4} mean_auc={self.mean:.4f}"


class StackFit:
    """Fit on one half, score every rule on the other, and choose the penalty without touching either."""

    @staticmethod
    def split(studies: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Disjoint fitting and scoring positions covering every study."""
        order = np.random.default_rng(SPLIT_SEED).permutation(studies)
        cut = int(studies * FIT_SHARE)
        return order[:cut], order[cut:]

    @staticmethod
    def mean_area(
        predictions: Float[NDArray[np.float64], "studies findings"],
        truth: Int[NDArray[np.int64], "studies findings"],
    ) -> float:
        """Mean per-label area, skipping labels the scoring half left with one class."""
        areas = [
            LabelScores.area_under_curve(predictions[:, label], truth[:, label])
            for label in range(truth.shape[1])
            if 0 < truth[:, label].sum() < len(truth)
        ]
        return float(np.mean(areas)) if areas else 0.0

    def chosen_penalty(
        self,
        per_member: list[Float[NDArray[np.float64], "studies findings"]],
        truth: Int[NDArray[np.int64], "studies findings"],
    ) -> float:
        """Pick the ridge penalty on a nested split INSIDE the fitting half.

        The caller's scoring studies never enter here, which is the whole point: a penalty read off the
        scoring half is a parameter fitted on the number being reported.
        """
        inner_fit, inner_score = self.split(len(truth))
        held = [member[inner_score] for member in per_member]
        areas = {
            penalty: self.mean_area(
                StackedLogit.fitted([member[inner_fit] for member in per_member], truth[inner_fit], penalty).combine(
                    held
                ),
                truth[inner_score],
            )
            for penalty in PENALTIES
        }
        return max(areas, key=lambda penalty: areas[penalty])

    def shipped_weights(self, members: list[MemberHoldout]) -> Float[NDArray[np.float64], "members findings"]:
        """Weights fitted on ALL holdout studies, for a kernel that will never see them again.

        Different from `scored` on purpose. `scored` holds a half back because it is REPORTING a number;
        this is producing a constant to ship, where holding studies back only makes the weights noisier
        and buys nothing — the kernel is judged on the competition test set, which no fit here can touch.
        The penalty still comes from a nested split, so it is chosen rather than assumed.
        """
        shared = MemberHoldout.aligned(members)
        truth = shared[0].truth
        per_member = [member.predictions for member in shared]
        penalty = self.chosen_penalty(per_member, truth)
        return StackedLogit.fitted(per_member, truth, penalty).weights

    def scored(self, members: list[MemberHoldout]) -> list[StackResult]:
        """Every rule on the same held-out studies, the stacker fitted only on the other half."""
        shared = MemberHoldout.aligned(members)
        truth = shared[0].truth
        per_member = [member.predictions for member in shared]
        fitting, scoring = self.split(len(truth))
        held = [member[scoring] for member in per_member]
        penalty = self.chosen_penalty([member[fitting] for member in per_member], truth[fitting])
        stacker = StackedLogit.fitted([member[fitting] for member in per_member], truth[fitting], penalty)
        rules: list[tuple[str, Vote]] = [
            ("StackedLogit", stacker),
            ("LogitAverage", LogitAverage()),
            ("RankAverage", RankAverage()),
        ]
        return [
            StackResult(
                rule=name,
                fitted_on=len(fitting) if name == "StackedLogit" else 0,
                scored_on=len(scoring),
                mean=self.mean_area(rule.combine(held), truth[scoring]),
            )
            for name, rule in rules
        ]


def main() -> None:
    """Score the learned combiner against the fixed rules over a directory of holdout records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdouts", type=Path, required=True)
    parser.add_argument("--members", type=int, default=0)
    parser.add_argument("--penalty", type=float, default=STACK_PENALTY)
    parser.add_argument("--only", nargs="+", default=None, help="member names, in the order a kernel seats them")
    parser.add_argument("--emit-weights", action="store_true", help="print shippable weights instead of a score")
    parsed = parser.parse_args()
    if parsed.only:
        loaded = [MemberHoldout.loaded(parsed.holdouts / f"{name}.holdout.npz") for name in parsed.only]
    else:
        found = sorted(parsed.holdouts.glob("*.holdout.npz"))
        loaded = [MemberHoldout.loaded(path) for path in found[: parsed.members or len(found)]]
    if parsed.emit_weights:
        for name, row in zip([member.name for member in loaded], StackFit().shipped_weights(loaded), strict=True):
            print(f"    ({', '.join(f'{value:.6f}' for value in row)}),  # {name}")  # noqa: T201 - the output
        return
    for result in StackFit().scored(loaded):
        print(result.line())  # noqa: T201 - a CLI report is the output of this module


if __name__ == "__main__":
    main()
