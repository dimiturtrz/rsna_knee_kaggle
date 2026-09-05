"""The best blend this project can PRICE, greedily, with no card and no slot.

`python -m kneemri.kaggle.priceable_blend [--width N] [--vote rank]`

WHY A SEARCH OVER PRICEABLE MEMBERS IS A DIFFERENT QUESTION FROM `seat_value`. That tool asks what a
candidate adds to the CHAMPION roster, and the champion is built from `_ff` members fitted at
`--fold -1`, so its own offline number does not exist. Every answer it gives is therefore a delta over a
pool whose level is unmeasurable. This asks the question the other way round: over the members that DID
hold out a split, which subset scores best, and what does the board-transfer compression predict its
board score would be?

WHAT MAKES THE PREDICTION POSSIBLE AT ALL, and it was not possible before 2026-09-05. `distinct14`
(offline 0.8749, board 0.937) is the first entry near the top of the board that carries an offline
number, so the compression finally has an anchor in the regime where entries actually get shipped. The
prediction below is `distinct14`'s board plus the compression slope times the offline gain over it. It
is a PREDICTION and it is quoted with the slope's own interval, because the top regime rests on four
resolved pairs and `board_transfer` calls that direction-only.

GREEDY, AND WHY NOT AN EXHAUSTIVE SEARCH. Forward selection over ~200 members is a few thousand pool
evaluations; the exhaustive version is not finite. Greedy is also the rule the roster was built under, so
a greedy answer is comparable to the entries already on the board rather than being a different kind of
object. The stopping rule is a gain under `MIN_GAIN`, not a fixed width, because a blend that stops
improving has found its own size.

THE NUMBER THIS PRINTS IS NOT A SEAT ARGUMENT. Selection over a holdout is selection, and the score it
reports is optimistic by construction — the same reason the member axis is closed on the board rather
than on this ruler. Read it as the CEILING of the priceable pool, which is what prices an arm: an arm
that cannot beat this pool's best member is not going to move an entry built from it.
"""

import logging
from pathlib import Path

import numpy as np

from core import cli
from kneemri.evaluation.record_mix import VOTE_RULES, RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.runs.holdout_record import HoldoutRecord

logger = logging.getLogger(__name__)

MIN_GAIN = 0.0002
"""The forward-selection stopping rule: a step that buys less than this has stopped buying anything.

Well under the 0.0053 the board can resolve, deliberately — the point of the last few steps is not that
each one is real but that the pool's ceiling is not understated by stopping early.
"""

DEFAULT_WIDTH = 20
"""A hard cap on the blend, so a pathological pool cannot walk to two hundred seats."""

ANCHOR_OFFLINE = 0.8749
ANCHOR_BOARD = 0.937
"""`distinct14` (ref 55943714): the only entry above board 0.936 that carries an offline number."""

COMPRESSION = 0.82
COMPRESSION_LOW = 0.71
COMPRESSION_HIGH = 0.93
"""`board_transfer`'s top regime, at or above offline 0.8695: four resolved pairs, median and interval.

Direction-only by that tool's own verdict, which is why every prediction here is printed as a band.
"""

IN_SAMPLE_MARKERS = ("_ff", "_cf", "final_ws")
"""A member whose name says it saw the whole holdout, so it can never appear in a priceable blend."""


class PriceableBlend:
    """Forward selection over the members that recorded a holdout, scored on the split they share."""

    def __init__(self, models: Path, vote: str, width: int) -> None:
        self.models = models
        self.vote = vote
        self.width = width

    def candidates(self) -> list[Path]:
        """Every checkpoint with a holdout record whose name does not mark it as full-fit."""
        records = sorted(self.models.glob("*.holdout.npz"))
        names = [record.name.removesuffix(".holdout.npz") for record in records]
        kept = [name for name in names if not any(marker in name for marker in IN_SAMPLE_MARKERS)]
        return [self.models / f"{name}.pt" for name in kept if (self.models / f"{name}.pt").exists()]

    def on_the_corpus_ruler(self, members: list[Path]) -> list[Path]:
        """The members scored against the corpus label table, dropping any that carry a re-mapped one.

        A member fitted on a re-mapped supervision table records its holdout against THAT table, so its
        AUC answers a different question and cannot be compared with the rest. `recal_holdannot_s2026`
        is the live case: its truth differs from the corpus table in 10,752 of 10,908 cells, it reads
        0.8964 where the best comparable member reads 0.8642, and forward selection picks it first and
        then cannot add a second seat, because `RecordMix.narrow` drops it the moment it has company.
        The ruler is chosen by MAJORITY rather than by name, so a future re-mapped arm is caught too.
        """
        tables: dict[bytes, list[Path]] = {}
        for member in members:
            record = HoldoutRecord(member)
            if not record.exists():
                continue
            tables.setdefault(np.load(record.path, allow_pickle=True)["truth"].tobytes(), []).append(member)
        return max(tables.values(), key=len) if tables else []

    def scored(self, members: list[Path]) -> float:
        """The blend's mean per-label AUC over the studies all of its members held out."""
        mix = RecordMix(members, foreign=True)
        mix.narrow()
        pool = mix.per_member(VOTE_RULES[self.vote]())
        return LabelScores().mean(pool.combined(list(range(len(pool)))), pool.truth)

    def chosen(self) -> list[tuple[str, float]]:
        """The greedy blend, as (member name, the blend's score once it joined)."""
        pool = self.on_the_corpus_ruler(self.candidates())
        picked: list[Path] = []
        history: list[tuple[str, float]] = []
        best = 0.0
        while len(picked) < self.width:
            step = max(
                ((self.scored([*picked, one]), one) for one in pool if one not in picked),
                default=(0.0, None),
            )
            if step[1] is None or step[0] - best < MIN_GAIN:
                break
            best, winner = step
            picked.append(winner)
            history.append((winner.stem, best))
        return history

    def predicted(self, offline: float) -> tuple[float, float]:
        """The board band this offline score implies, through the top regime compression."""
        low = ANCHOR_BOARD + COMPRESSION_LOW * (offline - ANCHOR_OFFLINE)
        high = ANCHOR_BOARD + COMPRESSION_HIGH * (offline - ANCHOR_OFFLINE)
        return low, high

    def fixed(self, names: list[str]) -> list[tuple[str, float]]:
        """A named set scored as one blend, so a kernel can be priced under the rule it will SERVE.

        The search runs under one vote rule and a kernel serves another unless someone checks; a
        prediction taken under `rank` and served under `LogitAverage` is a prediction about a different
        ensemble. This is how the two are held to the same rule.
        """
        members = [self.models / f"{name}.pt" for name in names]
        return [(", ".join(names), self.scored(members))]

    def report(self, names: list[str] | None = None) -> list[str]:
        """One line per accepted seat, then the prediction the blend implies."""
        history = self.fixed(names) if names else self.chosen()
        if not history:
            return ["REFUSED: no priceable member is on disk. A search that measured nothing has not passed."]
        lines = [f"{index + 1:2d}. {name} -> {score:.4f}" for index, (name, score) in enumerate(history)]
        offline = history[-1][1]
        low, high = self.predicted(offline)
        centre = ANCHOR_BOARD + COMPRESSION * (offline - ANCHOR_OFFLINE)
        lines.append(f"blend of {len(history)} priceable members, offline {offline:.4f}")
        lines.append(f"against distinct14 offline {ANCHOR_OFFLINE:.4f} board {ANCHOR_BOARD:.3f}")
        lines.append(f"predicted board {centre:.3f} [{low:.3f}, {high:.3f}] — DIRECTION ONLY, four pairs")
        return lines


def main() -> None:
    """Search the priceable pool for its best blend, and say what the board would make of it."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--vote", choices=tuple(VOTE_RULES), default="rank")
    command.parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="the most seats to accept")
    command.parser.add_argument("--members", nargs="*", help="score exactly these members instead of searching")
    arguments = command.parsed()
    root = command.rooted(arguments)

    blend = PriceableBlend(root.models(), arguments.vote, arguments.width)
    command.report(blend.report(arguments.members))


if __name__ == "__main__":
    main()
