"""The D43 swap gate: does ONE seat move the champion block, and does it keep the block's diversity.

`python -m kneemri.evaluation.seat_swap --holdouts <dir> --pair <child>:<parent> [...]`

WHY THIS AND NOT A SOLO DELTA. D43 swapped five champion parents for distilled children of their own
family across all 31 subsets: children carrying +0.0052 of solo read -0.0014 on the block, and
undistilled siblings at +0.0036 solo read -0.0002. Same-family solo lift reaches the block at ~0.08 of
itself, so a member's solo macro says almost nothing about whether it belongs in the entry. The gate an
arm pre-registers against is therefore `swap_gain = proxy(child in place of parent) - proxy(parents)`.

WHY IT IS NOT `swap_proxy.py`. That module answers "which SUBSET of original/distilled members scores
best", equal-weight, no anchor seat. This one answers "does this ONE seat move the block", under the
champion's own weights (five ours-members rank-mean at 2/3, the foreign anchor at 1/3). Different
question, different weighting, different space; the shared part is the holdout plumbing, not the number.

THE CORR GUARD IS D43'S SENTENCE, AND IT HAD NEVER BEEN CODED (researcher D49). Three different corr
numbers have been quoted as "the" guard: child-versus-parent (what the scratch script printed), the
block's mean pairwise (what D46 read), and child-versus-all-five (the D45 sixth-seat gate). None is
what D43 wrote. This module computes the sentence as written -- the child against the rank-mean of the
FOUR REMAINING parents, compared with the parent it replaces against those same four -- and ALSO emits
the mean-pairwise number, so the two readings already in the ledger can be reconciled without a second
definition of either. The child-versus-parent corr is deliberately NOT the guard: a child that copies
its parent passes it while stealing the block's diversity, which is the whole failure the gate exists
to catch.

TWO RANK TRANSFORMS, ON PURPOSE. The parents and the anchor are ranked over the FULL frame (909) and
the gold rows are dropped afterwards; the child is ranked after alignment to the kept rows (896). That
is what produced every published reading, so reproducing them means reproducing the asymmetry -- it is
at most one rank step per cell plus the 13 gold rows' displacement, and the deltas are AUCs of ranks so
the scaling itself cancels. `--ranked kept` ranks everything over the kept rows instead, which is the
cleaner definition and is offered so the difference can be MEASURED rather than argued about.

THE RULER IS `LedgerSeventyFive`, not a local recut. The scratch script cut truth at `>= 0.75` with
Medial Meniscus and Baker's at `> 0.5`; `LabelScores` cuts strictly (`> 0.75`), which is the same
partition because no label level sits at exactly 0.75.
"""

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray
from scipy.stats import rankdata

from core import cli
from kneemri.evaluation.ledger75 import LedgerSeventyFive
from kneemri.runs.holdout_record import UNKNOWN_EPOCH, HoldoutReading, HoldoutRecord

logger = logging.getLogger(__name__)

PARENTS: tuple[str, ...] = (
    "calib_span_s2026",
    "grade_ladder_s2026",
    "merged_blend_cf",
    "lr4e-5_ws_s2026",
    "through_dense_s2026",
)
"""The five fold-0 members the champion's ours-side is built from, in D16's order (the mean is
order-free; the order is kept so a printed block reads the same as every earlier table)."""

ANCHOR = "raptor_v4_swa"
"""The foreign member the proxy seats beside them. In the block for the SCORE, never for the corr: the
diversity the guard protects is the ours-side's, and the anchor is one member of a family we do not
control."""

ANCHOR_WEIGHT = 1.0 / 3.0
"""What that seat carries in the champion proxy -- the D35 frame every swap reading has been taken in."""

GATE = 0.0010
"""The pre-registered GO. Below it a seat is directional at best; D43's five readings span -0.0008 to
+0.0013, so the gate is at the top of the band the axis has ever produced."""

GUARD_SLACK = 0.01
"""How much MORE correlated with the rest of the block a child may be than the parent it replaces. A
child is allowed to be slightly closer than its parent -- families overlap -- but not to collapse the
block's spread while buying a solo point."""

GOLD_LEVELS = (0.0, 1.0)
"""A row whose twelve truth values are all exactly these is a radiologist override, not a report read.
Thirteen of the 909, dropped: they are the studies the two rulers disagree about most (D48), and no
published swap reading has ever included them."""


@dataclass(frozen=True)
class SeatFrame:
    """The studies, truth and member ranks a swap reading is taken over -- built once, read many times.

    A type rather than five arrays threaded through four signatures: the frame, the truth aligned to it
    and the rank matrices are only meaningful together, and a child aligned to a different order scores
    the wrong studies without failing.
    """

    studies: list[str]
    truth: Float[NDArray[np.float32], "studies targets"]
    ranks: dict[str, Float[NDArray[np.float64], "studies targets"]]
    holdouts: Path
    anchor: str = ANCHOR
    """The member seated at 1/3. A field rather than the module constant because `--anchor` was accepted
    and then ignored by `proxy`, which read ANCHOR directly -- so every reading with a substituted anchor
    died on a KeyError, and one that happened to name a seated member would have scored the default."""
    rank_kept: bool = False
    allow_ruler_move: bool = False
    """Whether a child scored by a DIFFERENT label table may still be seated. Off, because that is how a
    ruler change reads as a seat gain; on only to reproduce a historical number that was taken without
    the check."""

    @staticmethod
    def ranked(
        predictions: Float[NDArray[np.float64], "studies targets"],
    ) -> Float[NDArray[np.float64], "studies targets"]:
        """Per-column midranks scaled by the row count -- the space every block mean is taken in.

        Ties take midranks (`rankdata`'s default), which is not cosmetic: ordinal ranks would order tied
        predictions by array position, and a member serving a constant column would then carry a fake
        ordering into the block mean.
        """
        return np.asarray(rankdata(predictions, axis=0) / len(predictions), dtype=np.float64)

    @staticmethod
    def flat_correlation(
        left: Float[NDArray[np.float64], "studies targets"],
        right: Float[NDArray[np.float64], "studies targets"],
    ) -> float:
        """Pearson over every (study, finding) cell of two rank matrices, as one number.

        Flattened rather than averaged per label because that is what every corr in the ledger is, and a
        per-label mean of the same cells is a different number that would silently reprice D43 and D46.
        """
        return float(np.corrcoef(left.ravel(), right.ravel())[0, 1])

    @classmethod
    def over(
        cls,
        holdouts: Path,
        parents: tuple[str, ...] = PARENTS,
        anchor: str = ANCHOR,
        *,
        rank_kept: bool = False,
        allow_ruler_move: bool = False,
    ) -> "SeatFrame":
        """The frame the five parents and the anchor all hold out, minus the gold rows.

        Truth comes from the ANCHOR's record and every other member is asserted against it, because a
        member scored by a different label table reads as a seat improvement rather than as the ruler
        change it is -- which is exactly how `recal_holdannot`'s +0.0477 happened.
        """
        readings = {name: cls.record(holdouts, name) for name in (*parents, anchor)}
        frame = sorted(set.intersection(*[set(reading.studies) for reading in readings.values()]))
        truth = readings[anchor].at(frame).truth
        keep = ~np.all(np.isin(truth, GOLD_LEVELS), axis=1)
        studies = [study for study, kept in zip(frame, keep, strict=True) if kept]
        rows = studies if rank_kept else frame
        ranks = {
            name: cls.ranked(reading.at(rows).predictions.astype(np.float64)) for name, reading in readings.items()
        }
        return cls(
            studies=studies,
            truth=truth[keep],
            ranks={name: matrix if rank_kept else matrix[keep] for name, matrix in ranks.items()},
            holdouts=holdouts,
            anchor=anchor,
            rank_kept=rank_kept,
            allow_ruler_move=allow_ruler_move,
        )

    @staticmethod
    def record(holdouts: Path, name: str) -> HoldoutReading:
        """One member's held-out reading, refusing a member that never wrote one."""
        return HoldoutRecord(holdouts / f"{name}.pt").required()

    def child(self, name: str) -> Float[NDArray[np.float64], "studies targets"]:
        """A candidate member's ranks over this frame, asserted to be scored by the same truth.

        Ranked over the KEPT rows unless the frame was built `rank_kept`, in which case everything is:
        the parents were ranked over the full frame in every published reading and reproducing them
        means reproducing that, but the two definitions must never be mixed inside one call.
        """
        reading = self.record(self.holdouts, name)
        if not set(self.studies) <= set(reading.studies):
            message = f"{name} does not cover the frame ({len(self.studies)} studies); a partial member is refused"
            raise ValueError(message)
        kept = reading.at(self.studies)
        moved = float(np.mean(~np.isclose(kept.truth, self.truth)))
        if moved > 0.0:
            message = (
                f"{name} was scored against a different label table than {self.anchor}: {moved:.1%} of cells differ. "
                "Its swap gain would be a ruler move wearing a seat's clothes (the recal_holdannot incident). "
                "Pass --allow-ruler-move to score it against the frame's truth anyway, as D43 did."
            )
            if not self.allow_ruler_move:
                raise ValueError(message)
            logger.warning("%s", message)
        return self.ranked(kept.predictions.astype(np.float64))

    def proxy(
        self, block: list[Float[NDArray[np.float64], "studies targets"]], *, with_anchor: bool = True
    ) -> Float[NDArray[np.float64], "studies targets"]:
        """The champion proxy over this ours-side block: rank-mean at 2/3, the anchor at 1/3."""
        mean = np.mean(block, axis=0)
        return (1 - ANCHOR_WEIGHT) * mean + ANCHOR_WEIGHT * self.ranks[self.anchor] if with_anchor else mean

    def macro(self, predictions: Float[NDArray[np.float64], "studies targets"]) -> float:
        """LEDGER75 over this frame -- ten findings at 0.75, Medial Meniscus and Baker's at the midpoint."""
        reading = HoldoutReading(studies=self.studies, predictions=predictions, truth=self.truth, epoch=UNKNOWN_EPOCH)
        return LedgerSeventyFive().macro(reading)

    def block(
        self, parents: tuple[str, ...], child: Float[NDArray[np.float64], "studies targets"] | None, replaced: str
    ) -> list[Float[NDArray[np.float64], "studies targets"]]:
        """The five ours-side seats, with `child` in `replaced`'s chair when one is given."""
        return [child if name == replaced and child is not None else self.ranks[name] for name in parents]

    def mean_pairwise(self, block: list[Float[NDArray[np.float64], "studies targets"]]) -> float:
        """Mean upper-triangle corr between block members -- the number D46's 0.871 -> 0.889 was read as."""
        stacked = np.asarray(np.corrcoef(np.stack([member.ravel() for member in block])))
        return float(stacked[np.triu_indices(len(block), 1)].mean())


@dataclass(frozen=True)
class SwapReading:
    """One child in one parent's chair: what it moved, and what it cost the block's spread."""

    child: str
    parent: str
    solo_child: float
    solo_parent: float
    swap_gain: float
    swap_five: float
    child_rest_corr: float
    parent_rest_corr: float
    block_pairwise: float
    parents_pairwise: float
    gate: float = GATE
    slack: float = GUARD_SLACK

    def guarded(self) -> bool:
        """Whether the child keeps the block's diversity -- D43's sentence, computed as written."""
        return self.child_rest_corr <= self.parent_rest_corr + self.slack

    def passes(self) -> bool:
        """GO: the seat moves the block AND does not collapse it. Both, never either alone."""
        return self.swap_gain >= self.gate and self.guarded()

    def verdict(self) -> str:
        """GO, or the half of the gate that refused it -- so a reading is never quoted without its reason."""
        if self.passes():
            return "GO"
        return "NO-GO gain" if self.swap_gain < self.gate else "NO-GO corr"

    def line(self) -> str:
        """The reading as one table row."""
        return (
            f"{self.child[:26]:26s} {self.parent[:22]:22s} {self.solo_child:10.4f} {self.solo_parent:11.4f} "
            f"{self.solo_child - self.solo_parent:+7.4f} {self.swap_gain:+12.4f} {self.swap_five:+11.4f} "
            f"{self.child_rest_corr:9.3f} {self.parent_rest_corr:10.3f} {self.block_pairwise:9.3f} "
            f"{self.verdict():>11s}"
        )


class SeatSwap:
    """The gate itself: price a child in a parent's chair, over a frame built once."""

    HEADER = (
        f"{'child':26s} {'for parent':22s} {'solo child':>10s} {'solo parent':>11s} {'dsolo':>7s} "
        f"{'swap d proxy':>12s} {'swap d five':>11s} {'corr rest':>9s} {'parent rest':>10s} "
        f"{'pairwise':>9s} {'verdict':>11s}"
    )

    def __init__(self, frame: SeatFrame, parents: tuple[str, ...] = PARENTS, gate: float = GATE) -> None:
        self.frame = frame
        self.parents = parents
        self.gate = gate

    def baseline(self) -> tuple[float, float]:
        """The untreated proxy and the five ours-members alone -- recomputed, never cached."""
        block = self.frame.block(self.parents, None, "")
        return self.frame.macro(self.frame.proxy(block)), self.frame.macro(self.frame.proxy(block, with_anchor=False))

    def rest(self, parent: str) -> Float[NDArray[np.float64], "studies targets"]:
        """The rank-mean of the four parents a swap does NOT touch -- what the guard measures against."""
        return np.mean([self.frame.ranks[name] for name in self.parents if name != parent], axis=0)

    def reading(self, child: str, parent: str) -> SwapReading:
        """One swap, priced on both halves of the gate."""
        if parent not in self.parents:
            message = f"{parent} is not one of the block's seats: {', '.join(self.parents)}"
            raise ValueError(message)
        proxy, five = self.baseline()
        seated = self.frame.child(child)
        block = self.frame.block(self.parents, seated, parent)
        rest = self.rest(parent)
        return SwapReading(
            child=child,
            parent=parent,
            solo_child=self.frame.macro(seated),
            solo_parent=self.frame.macro(self.frame.ranks[parent]),
            swap_gain=self.frame.macro(self.frame.proxy(block)) - proxy,
            swap_five=self.frame.macro(self.frame.proxy(block, with_anchor=False)) - five,
            child_rest_corr=self.frame.flat_correlation(seated, rest),
            parent_rest_corr=self.frame.flat_correlation(self.frame.ranks[parent], rest),
            block_pairwise=self.frame.mean_pairwise(block),
            parents_pairwise=self.frame.mean_pairwise(self.frame.block(self.parents, None, "")),
            gate=self.gate,
        )

    def table(self, pairs: list[tuple[str, str]]) -> list[SwapReading]:
        """Every requested swap, in the order asked. A missing member raises rather than being skipped."""
        return [self.reading(child, parent) for child, parent in pairs]

    def lines(self, readings: list[SwapReading]) -> list[str]:
        """The table a decision is read off, with the baseline it is all relative to."""
        proxy, five = self.baseline()
        parents_pairwise = self.frame.mean_pairwise(self.frame.block(self.parents, None, ""))
        head = [
            f"frame {len(self.frame.studies)} | champion proxy {proxy:.4f} | five alone {five:.4f} | "
            f"parents mean pairwise corr {parents_pairwise:.3f} | ranked over "
            f"{'kept rows' if self.frame.rank_kept else 'the full frame (published transform)'}",
            "parents solo: "
            + ", ".join(f"{name} {self.frame.macro(self.frame.ranks[name]):.4f}" for name in self.parents),
            "",
            self.HEADER,
        ]
        return head + [reading.line() for reading in readings]

    @staticmethod
    def parsed_pairs(given: list[str]) -> list[tuple[str, str]]:
        """`child:parent` arguments as pairs -- the parent MATTERS and is never inferred.

        D43's readings were each taken against a named parent (`labelwise_dense` against `through_dense`
        because both are `_dense` trunks); picking a different chair changes the number, so the caller
        states it.
        """
        pairs = []
        for one in given:
            child, _, parent = one.partition(":")
            if not parent:
                message = f"--pair takes <child>:<parent>, not {one!r}"
                raise ValueError(message)
            pairs.append((child, parent))
        return pairs

    def compositions(self, pairs: list[tuple[str, str]]) -> list[str]:
        """Every subset of the given swaps applied at once, against the sum of its singles.

        D42's question, kept because it is the one that showed the axis SATURATES: the all-k delta ran below
        the sum of the singles at every k, which is why a second treated seat is priced from the treated
        block and never from the untreated one.
        """
        singles = {child: self.reading(child, parent).swap_gain for child, parent in pairs}
        proxy, _ = self.baseline()
        lines = [f"{'k':>2s} {'n':>3s} {'mean all-k':>10s} {'mean sum':>9s} {'mean interaction':>16s} {'pairwise':>9s}"]
        for size in range(1, len(pairs) + 1):
            alls: list[float] = []
            sums: list[float] = []
            corrs: list[float] = []
            for subset in itertools.combinations(pairs, size):
                seated = {parent: self.frame.child(child) for child, parent in subset}
                block = [seated.get(name, self.frame.ranks[name]) for name in self.parents]
                alls.append(self.frame.macro(self.frame.proxy(block)) - proxy)
                sums.append(sum(singles[child] for child, _ in subset))
                corrs.append(self.frame.mean_pairwise(block))
            lines.append(
                f"{size:2d} {len(alls):3d} {np.mean(alls):+10.4f} {np.mean(sums):+9.4f} "
                f"{np.mean(alls) - np.mean(sums):+16.4f} {np.mean(corrs):9.3f}"
            )
        return lines


def main() -> None:
    """Price one or more children in the champion block's chairs, under the D43 gate."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--holdouts", type=Path, required=True, help="directory of .holdout.npz records")
    command.parser.add_argument(
        "--pair",
        nargs="+",
        action="extend",
        required=True,
        help="child:parent, the chair it stands in; repeatable, and repeats ACCUMULATE",
    )
    command.parser.add_argument("--parents", nargs="+", default=list(PARENTS), help="the block's five seats")
    command.parser.add_argument("--anchor", default=ANCHOR, help="the foreign member seated at 1/3")
    command.parser.add_argument("--gate", type=float, default=GATE, help="the swap gain a GO needs")
    command.parser.add_argument(
        "--ranked",
        choices=("frame", "kept"),
        default="frame",
        help="rank the parents over the full frame (published) or over the kept rows (cleaner)",
    )
    command.parser.add_argument(
        "--compose", action="store_true", help="also apply every subset of the swaps at once (D42's saturation read)"
    )
    command.parser.add_argument(
        "--allow-ruler-move", action="store_true", help="seat a child scored by a different label table (D43 did)"
    )
    arguments = command.parsed()

    parents = tuple(arguments.parents)
    frame = SeatFrame.over(
        arguments.holdouts,
        parents,
        arguments.anchor,
        rank_kept=arguments.ranked == "kept",
        allow_ruler_move=arguments.allow_ruler_move,
    )
    gate = SeatSwap(frame, parents, arguments.gate)
    pairs = SeatSwap.parsed_pairs(arguments.pair)
    readings = gate.table(pairs)
    lines = [*gate.lines(readings), f"{sum(one.passes() for one in readings)} of {len(readings)} pass the gate"]
    if arguments.compose:
        lines += ["", "k-swap compositions, all-k delta vs the sum of its singles:", *gate.compositions(pairs)]
    command.report(lines)


if __name__ == "__main__":
    main()
