"""Does ONE slice-pooling rule for twelve findings cost anything? `python -m kneemri.evaluation.pooling_by_finding`

THE QUESTION. Every member this project owns aggregates its slices with a single rule chosen per MEMBER
and never per FINDING — `max` for 284 of the runs on disk, `labelwise` for 23, `attend` for 5. But the
twelve targets are not one kind of thing. A cruciate tear, a meniscal tear and a fracture are FOCAL: the
evidence is a few slices and the rest of the stack is a distraction, which is what `max` is for. OA,
effusion and synovitis are DIFFUSE: the evidence is spread across the stack and `max` keeps one slice of
it, which is what an averaging rule is for. If that split is real, one rule is necessarily wrong for half
the board.

WHY THIS IS WORTH A READ BEFORE A CARD. `column_headroom` established the board is a plain unweighted
mean of twelve per-column AUCs, so a column gaining `d` moves the board by `d/12` and six columns gaining
0.01 is +0.005 — above `BOARD_RESOLVABLE` and inside the range that has actually moved the board. And the
read costs nothing: both pooling families already exist on disk with holdout records, so the contrast can
be taken without fitting anything.

WHAT IT CAN AND CANNOT SAY. It compares FAMILIES of members, not a controlled pair — the `labelwise` runs
are not the `max` runs with one flag changed, and they differ in backbone, depth and budget as well. So a
per-column difference here is an ASSOCIATION between a pooling rule and a column, and the only claim it
licenses is about the SHAPE: whether the sign of that difference sorts with the focal/diffuse split at a
rate a coin would not produce. A yes buys a controlled arm. A no closes the axis for the cost of a read,
which is the whole point of taking it first.

THE NULL WORTH NAMING. `labelwise` pooling already learns a per-column weighting over slices, so the
mechanism this bead proposes may ALREADY be in the pool. If it is, `labelwise` wins the diffuse columns
and loses nothing on the focal ones, and a per-finding head is a reimplementation rather than a lever.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core import cli
from core.findings import TARGETS
from kneemri.evaluation.ledger75 import LedgerSeventyFive
from kneemri.evaluation.seat_swap import SeatFrame
from kneemri.runs.holdout_record import SUFFIX, HoldoutReading
from kneemri.runs.run_metrics import RunMetrics

logger = logging.getLogger(__name__)

type Ranks = Float[NDArray[np.float64], "studies targets"]
"""One member's per-column midranks on a shared frame — the space every block mean here is taken in."""

POOLING = "slice_pooling"
"""The recipe key the rule is stamped under. `pool` and `slot_pooling` are different axes on the same runs."""

FOCAL: tuple[str, ...] = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Fracture", "Baker's")
"""Findings whose evidence is a discrete structure on a few slices — a tear, a break, a discrete collection.

Baker's sits here and not with effusion because a popliteal cyst is a BOUNDED mass with a neck, visible on
the slices that cross it and absent from the rest; joint fluid is not. The split is argued from what the
finding IS, never fitted to which side wins, because a split chosen on the outcome cannot then test it.
"""

DIFFUSE: tuple[str, ...] = tuple(target for target in TARGETS if target not in FOCAL)
"""The other six — graded severities and distributed fluid, where evidence accumulates across the stack."""

DEFAULT_TOP = 6
"""Members per family. Enough to average a family's scatter, few enough that all of them are real members."""


@dataclass(frozen=True)
class PoolingContrast:
    """One finding, under two pooling families, and the split it was assigned to before either was read."""

    finding: str
    focal: bool
    left: float
    right: float

    def gap(self) -> float:
        """Right family minus left, on this finding's own LEDGER75 column."""
        return self.right - self.left


class PoolingByFinding:
    """Per-column LEDGER75, grouped by the slice-pooling rule its members were fitted under."""

    def __init__(self, holdouts: Path, ruler: LedgerSeventyFive | None = None) -> None:
        self.holdouts = holdouts
        self.ruler = ruler or LedgerSeventyFive()

    def rule(self, stem: str) -> str | None:
        """The pooling rule this run recorded, or None when its metrics file never stated one."""
        metrics = RunMetrics(self.holdouts / f"{stem}.metrics.jsonl")
        recipe = metrics.recipe()
        return None if recipe is None else recipe.get(POOLING)

    def families(self) -> dict[str, list[str]]:
        """Every run with BOTH a holdout record and a stated pooling rule, grouped by that rule."""
        grouped: dict[str, list[str]] = {}
        for path in sorted(self.holdouts.glob(f"*{SUFFIX}")):
            stem = path.name.removesuffix(SUFFIX)
            rule = self.rule(stem)
            if rule is not None:
                grouped.setdefault(rule, []).append(stem)
        return grouped

    def read(self, stem: str) -> HoldoutReading:
        """One member's holdout record, under the mixed 0.75 cut."""
        return self.ruler.reading(self.holdouts / f"{stem}{SUFFIX}")

    def columns(self, stem: str) -> dict[str, float]:
        """One member's twelve LEDGER75 columns, each read where that column's own coin-flip rung sits."""
        return self.ruler.per_label(self.read(stem))

    def strongest(self, stems: list[str], top: int) -> list[str]:
        """The `top` members of a family by LEDGER75 macro — a family is judged on its best, not its mean.

        A rule's twenty-third-best member says more about what was tried under it than about the rule, and
        the two families differ by an order of magnitude in size, so a mean over all of each would compare
        `max`'s whole history against `labelwise`'s short one.
        """
        scored = [(self.ruler.macro(self.read(stem)), stem) for stem in stems]
        return [stem for _, stem in sorted(scored, reverse=True)[:top]]

    def family_columns(self, stems: list[str]) -> dict[str, float]:
        """A family's per-column mean over its chosen members."""
        readings = [self.columns(stem) for stem in stems]
        return {target: float(np.nanmean([reading[target] for reading in readings])) for target in TARGETS}

    def aligned(self, stems: list[str]) -> tuple[Float[NDArray[np.float32], "studies targets"], list[Ranks]]:
        """The truth and every member's per-column midranks, on the studies all of them hold out.

        Intersected rather than assumed equal: a member fitted on a different fold holds out different
        studies, and stacking those arrays by position scores one member's rows against another's truth
        without failing anywhere a reader would look.
        """
        readings = [self.read(stem) for stem in stems]
        common = sorted(set.intersection(*(set(reading.studies) for reading in readings)))
        if not common:
            message = f"no study is held out by all {len(stems)} members - these folds share no frame to vote on"
            raise ValueError(message)
        index = {study: position for position, study in enumerate(common)}
        ranks: list[Ranks] = []
        truth = np.zeros((len(common), len(TARGETS)), dtype=np.float32)
        for reading in readings:
            keep = [position for position, study in enumerate(reading.studies) if study in index]
            order = np.argsort([index[reading.studies[position]] for position in keep])
            rows = np.asarray(keep, dtype=np.int64)[order]
            ranks.append(SeatFrame.ranked(np.asarray(reading.predictions, dtype=np.float64)[rows]))
            truth = np.asarray(reading.truth, dtype=np.float32)[rows]
        return truth, ranks

    def blended_columns(self, stems: list[str]) -> dict[str, float]:
        """The family voted as ONE member — a rank average, the rule the kernel's outer vote uses.

        This is the question `family_columns` cannot answer. A mean of twelve per-member AUCs says what
        a family's members score; an entry is a BLEND, and a family whose members agree gains nothing
        from being averaged. Routing is only worth a slot if the per-column difference survives here.
        """
        truth, ranks = self.aligned(stems)
        voted = np.mean(np.stack(ranks), axis=0)
        return self.ruler.scorer().per_label(voted, truth)

    def contrast(self, left: str, right: str, top: int = DEFAULT_TOP, *, blend: bool = False) -> list[PoolingContrast]:
        """Both families' per-column scores, paired by finding, with each finding's pre-assigned split."""
        grouped = self.families()
        missing = [name for name in (left, right) if name not in grouped]
        if missing:
            message = f"no runs on disk pooled by {missing} - families present: {sorted(grouped)}"
            raise KeyError(message)
        read = self.blended_columns if blend else self.family_columns
        scores = {name: read(self.strongest(grouped[name], top)) for name in (left, right)}
        return [
            PoolingContrast(target, target in FOCAL, scores[left][target], scores[right][target]) for target in TARGETS
        ]

    def sorts(self, contrasts: list[PoolingContrast]) -> tuple[float, float]:
        """Mean gap on the focal findings and on the diffuse ones — the two numbers the split predicts differ."""
        focal = [row.gap() for row in contrasts if row.focal]
        diffuse = [row.gap() for row in contrasts if not row.focal]
        return float(np.mean(focal)), float(np.mean(diffuse))

    def routed(self, contrasts: list[PoolingContrast]) -> dict[str, float]:
        """What the split is WORTH: each family's macro, the BLIND routing, and the oracle's ceiling.

        `blind` takes the left family on every focal finding and the right family on every diffuse one —
        the assignment argued from anatomy before either column was read, so it is a real out-of-sample
        rule rather than a per-column argmax. `oracle` takes whichever family actually won each column
        and is therefore fitted on this very table: it is the CEILING the blind rule is measured against,
        never a result. The distance between them is how much of the routing gain is the split and how
        much is hindsight.
        """
        blind = [row.left if row.focal else row.right for row in contrasts]
        oracle = [max(row.left, row.right) for row in contrasts]
        return {
            "left": float(np.mean([row.left for row in contrasts])),
            "right": float(np.mean([row.right for row in contrasts])),
            "blind": float(np.mean(blind)),
            "oracle": float(np.mean(oracle)),
        }


def main() -> None:
    """Print two pooling families column by column, and whether the focal/diffuse split sorts their gaps."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True, help="directory of .holdout.npz records")
    command.parser.add_argument("--left", default="max", help="the baseline pooling rule")
    command.parser.add_argument("--right", default="labelwise", help="the pooling rule compared against it")
    command.parser.add_argument("--top", type=int, default=DEFAULT_TOP, help="members per family")
    command.parser.add_argument(
        "--blend", action="store_true", help="vote each family as one member instead of averaging their AUCs"
    )
    arguments = command.parsed()

    reader = PoolingByFinding(arguments.holdouts)
    contrasts = reader.contrast(arguments.left, arguments.right, arguments.top, blend=arguments.blend)
    focal, diffuse = reader.sorts(contrasts)

    lines = [f"{'finding':20s} {'split':>8s} {arguments.left[:9]:>9s} {arguments.right[:9]:>9s} {'gap':>8s}"]
    lines += [
        f"{row.finding[:20]:20s} {'focal' if row.focal else 'diffuse':>8s} "
        f"{row.left:9.4f} {row.right:9.4f} {row.gap():+8.4f}"
        for row in sorted(contrasts, key=lambda row: -row.gap())
    ]
    lines.append(f"mean gap on the {len(FOCAL)} focal findings:   {focal:+.4f}")
    lines.append(f"mean gap on the {len(DIFFUSE)} diffuse findings: {diffuse:+.4f}")
    lines.append(f"the split separates the two families by {diffuse - focal:+.4f}")

    worth = reader.routed(contrasts)
    best = max(worth["left"], worth["right"])
    lines.append(f"macro {arguments.left} {worth['left']:.4f} · {arguments.right} {worth['right']:.4f}")
    lines.append(f"blind routing by the split {worth['blind']:.4f}  {worth['blind'] - best:+.4f} on the better family")
    lines.append(f"oracle per-column argmax  {worth['oracle']:.4f}  {worth['oracle'] - best:+.4f} — a fitted CEILING")
    command.report(lines)


if __name__ == "__main__":
    main()
