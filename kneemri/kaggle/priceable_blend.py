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

THE NUMBER THIS PRINTS IS NOT A SEAT ARGUMENT, and `--optimism` says by how much. AT THE DEFAULT TEN
FOLDS — the regime every offline number in `comp/board.jsonl` was taken in, because the search reads 90%
of the shared studies — the optimism is **+0.0019**, measured 2026-09-06 over the 206-member pool, with
a per-fold spread of -0.0199 to +0.0254. Each fold prices on 91 studies, so the SPREAD is what one small
pricing set does and the MEAN is the bias; ten folds put that bias near a fifth of a resolvable board
point.

This corrects a number this file quoted for three weeks. The **+0.0201** in `DEFAULT_FOLDS`' note below
is a TWO-fold reading — search on 455 studies, price on the 454 it never saw — and it answers a
different question: what a search overfits when it reads HALF the corpus. It is an upper bracket, it was
taken once, and it was being cited as the optimism of a normal run. It is not. A searched blend is very
nearly an unbiased estimate of itself, and the honest correction is +0.0019 rather than a tenth of the
distance to first place.

Still subtract it before comparing a searched offline with an unsearched one, and still read the raw
number as a CEILING rather than an estimate: an arm that cannot beat this pool's best member is not
going to move an entry built from it.
"""

import logging
from pathlib import Path

import numpy as np

from core import cli
from kneemri.evaluation.record_mix import VOTE_RULES, RecordMix
from kneemri.evaluation.scores import LabelScores
from kneemri.runs.artefact import Artefact
from kneemri.runs.holdout_record import HoldoutRecord

logger = logging.getLogger(__name__)

MIN_GAIN = 0.0002
"""The forward-selection stopping rule: a step that buys less than this has stopped buying anything.

Well under the 0.0053 the board can resolve, deliberately — the point of the last few steps is not that
each one is real but that the pool's ceiling is not understated by stopping early.
"""

DEFAULT_WIDTH = 20
"""A hard cap on the blend, so a pathological pool cannot walk to two hundred seats."""

DEFAULT_FOLDS = 10
"""Ten, because the measurement is meant to price a search run at nearly full data.

A two-fold cut answers a different question — what a search overfits when it reads half the corpus — and
that reading (+0.0201 over the 195-member pool) is an upper bracket on the one this module is asked for.
At ten the search reads 90% of the shared studies, which is the regime every offline number in
`comp/board.jsonl` was actually taken in.
"""

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
        names = [Artefact.stem(record) for record in records]
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

    def scored_on(self, members: list[Path], studies: set[str] | None) -> float:
        """The blend's mean per-label AUC, over `studies` if given and over the whole shared split if not.

        Restricted by study NAME rather than by row index on purpose: every subset of members intersects
        to its own shared split, so a row that is study 40 for one pair is study 37 for the next. A name
        survives that; an index quietly prices a different set of patients.
        """
        mix = RecordMix(members, foreign=True)
        mix.narrow()
        pool = mix.per_member(VOTE_RULES[self.vote]())
        combined = pool.combined(list(range(len(pool))))
        if studies is None:
            return LabelScores().mean(combined, pool.truth)
        rows = [index for index, name in enumerate(mix.shared()) if name in studies]
        return LabelScores().mean(combined[rows], pool.truth[rows])

    def scored(self, members: list[Path]) -> float:
        """The blend's mean per-label AUC over the studies all of its members held out."""
        return self.scored_on(members, None)

    def folds(self, count: int = DEFAULT_FOLDS) -> list[set[str]]:
        """The shared studies cut into `count` parts, so a search and the price of what it found share none.

        Cut by sorted position rather than at random: the split has to be the same one on every run, or
        the optimism it measures is partly the split moving underneath it.

        THE COUNT IS THE MEASUREMENT'S SECOND VARIABLE and it is why this is not fixed at two. Selection
        overfit grows as the ruler the search reads gets noisier, and a two-fold cut halves the studies
        the search sees. At `count` folds the search reads `count - 1` of them, so ten folds put the
        search on 90% of the corpus ruler rather than 50% of it, and the gap it reports is the gap of a
        search run at nearly full data. A two-fold number is an upper bracket on that, never its value.
        """
        mix = RecordMix(self.on_the_corpus_ruler(self.candidates()), foreign=True)
        mix.narrow()
        studies = sorted(mix.shared())
        return [set(studies[start::count]) for start in range(count)]

    def chosen(self, studies: set[str] | None = None) -> list[tuple[str, float]]:
        """The greedy blend, as (member name, the blend's score once it joined)."""
        pool = self.on_the_corpus_ruler(self.candidates())
        picked: list[Path] = []
        history: list[tuple[str, float]] = []
        best = 0.0
        while len(picked) < self.width:
            step = max(
                ((self.scored_on([*picked, one], studies), one) for one in pool if one not in picked),
                default=(0.0, None),
            )
            if step[1] is None or step[0] - best < MIN_GAIN:
                break
            best, winner = step
            picked.append(winner)
            history.append((winner.stem, best))
        return history

    def optimism(self, count: int = DEFAULT_FOLDS) -> list[str]:
        """How much of a searched blend's offline score is the search finding the holdout rather than a
        better ensemble: select on all folds but one, price on the one held out, report the drop.

        This is the number every prediction in this module has been missing. `report` compares a SEARCHED
        offline against `distinct14`'s, and if that anchor was not itself searched the comparison hands
        the searched side a head start of exactly this size. Subtract it before predicting a board score,
        or the prediction is optimistic by construction and will read as a ruler inversion when the board
        answers lower.

        AT `count` = 2 THIS IS AN UPPER BRACKET, NOT A VALUE, and the distinction cost a wrong number in
        a letter and a wrong branch in a pre-registration. The two-fold reading of +0.0201 was taken with
        the search on 455 studies and the widest pool this project owns; `kneemri.evaluation.selection_bias`
        reads +0.0025 to +0.0031 with the search on all of them and a narrower pool. Both are real and
        they measure different regimes. Raising `count` holds the pool where it is and gives the search
        back its studies, which is the only one of the two knobs this module controls.
        """
        parts = self.folds(count)
        gaps: list[float] = []
        lines = [f"{count} folds over {sum(len(part) for part in parts)} shared studies"]
        for index, held in enumerate(parts):
            selection = set().union(*(part for position, part in enumerate(parts) if position != index))
            history = self.chosen(selection)
            if not history:
                return ["REFUSED: no priceable member is on disk. A search that measured nothing has not passed."]
            members = [self.models / f"{name}.pt" for name, _ in history]
            inside = self.scored_on(members, selection)
            outside = self.scored_on(members, held)
            gaps.append(inside - outside)
            lines.append(
                f"  fold {index + 1}: selected on {len(selection)}, priced on {len(held)}; "
                f"in-sample {inside:.4f}, held out {outside:.4f}, gap {inside - outside:+.4f} "
                f"({len(history)} seats: {', '.join(name for name, _ in history)})"
            )
        lines.append(
            f"optimism {sum(gaps) / len(gaps):+.4f} over {len(gaps)} folds, "
            f"spread {min(gaps):+.4f} to {max(gaps):+.4f} — subtract it from any searched offline"
        )
        return lines

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
    command.parser.add_argument(
        "--optimism",
        action="store_true",
        help="measure how much of a searched score is the search itself: select on all folds but one, "
        "price on the one held out",
    )
    command.parser.add_argument(
        "--folds",
        type=int,
        default=DEFAULT_FOLDS,
        help="how many folds --optimism cuts the studies into; two halves the ruler the search reads and "
        "reports an upper bracket, ten leaves the search 90 percent of it",
    )
    arguments = command.parsed()
    root = command.rooted(arguments)

    blend = PriceableBlend(root.models(), arguments.vote, arguments.width)
    command.report(blend.optimism(arguments.folds) if arguments.optimism else blend.report(arguments.members))


if __name__ == "__main__":
    main()
