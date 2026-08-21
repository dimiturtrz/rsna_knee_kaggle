"""What does each additional ensemble member actually buy?

`python -m kneemri.evaluation.member_curve --checkpoints a.pt b.pt c.pt --labels labels.csv`

This question was answered once already and the answer was void. The members were trained at different
model seeds, the fold split was derived from that seed, and so each member was scored on a holdout that
was training data for the others — which reported an ensemble of five at 0.8807 when its own members
logged 0.83. The split is now a property of the corpus (`--split-seed`), so members that differ only in
initialisation share one holdout and the comparison means something.

The curve is the deciding number for two open questions at once. It says whether our fifth member earns
its forward passes at all, and the Efficiency Prize track turns entirely on that: teams scoring exactly
our 0.888 sit at efficiency ranks 29-50, while the best score on that board sits at 638, so a cheap
submission at nearly our accuracy is worth far more there than a costly one.

Every subset of each size is scored, not one arbitrary subset, because with three or five members the
spread between the best and worst subset of a given size is the honest error bar on the mean.
"""

import argparse
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from core.paths import DataRoot
from core.wakelock import Wakelock
from kneemri.corpus.folds import ReportFolds
from kneemri.device import VisibleCards
from kneemri.evaluation.label_combine import LabelCombine
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.holdout_predictions import HoldoutPredictions
from kneemri.inference.member_readings import MemberReadings
from kneemri.inference.members import Member
from kneemri.inference.prediction_cache import PredictionCache
from kneemri.inference.test_time import SpatialViews
from kneemri.inference.vote import LogitAverage, QualityWeightedRank, RankAverage
from kneemri.labels.targets import Targets

logger = logging.getLogger(__name__)

COMBINE_SEED = 2026
"""Which studies fit the per-label choice and which report it — the corpus split seed, since both are
arbitrary-but-fixed partitions of the same corpus and a second free number would only be a second thing
to accidentally vary."""

BOARD_RESOLVABLE = 0.0015
"""The offline seat delta below which this ruler has never been shown to pick a board tick.

IT IS A SPENDING GUARD, NOT A TARGET, and the distinction is the one thing to keep straight about it.
This project's bar for a RESULT is ~1.5% absolute and its noise floor is 0.0062; 0.0015 is under both by
an order of magnitude. So nothing here is a number to chase or to plan an arm around — it exists to stop
a slot being spent on a move the board cannot read, which is a question about a submission rather than
about a method. An axis worth arming has to argue for 0.015, not for this.

Six entries pair an offline joint-holdout score with a board score (knee-825t) and the chain rises across
the first 0.003 of offline spread, then plateaus: `longseat` carried +0.0012 — the largest single move in
a sweep of all 56 same-split checkpoints, six times the runner-up — and TIED the board. 0.0015 sits above
that plateau and below the smallest gap the chain does track, so it is a floor on what has been
DEMONSTRATED, not a threshold anything is known to cross.

It documents this constant rather than the seed above it, which is what it used to do — a copy-paste that
left the number gating a submission decision described as a corpus partition."""

BOARD_COST_MULTIPLE = 3
"""What the board charges for a negative offline seat delta, as a multiple of the offline number.

Three paired entries: -0.0013 offline became -0.004 and -0.0023 became -0.005 (knee-c0u9). The holdout
ranks correctly and errs in magnitude, and only in the direction of understating a weak member's cost —
so an offline delta inside the noise floor is not a free look, and this is the factor that says so."""


@dataclass(frozen=True)
class Pool:
    """What several members predicted for the same studies, in one order — every reading's input.

    A type rather than three parameters threaded through nine signatures. The arrays are only meaningful
    TOGETHER: a name list reordered against a prediction list mislabels every row of every report and
    nothing fails, and a truth vector from a different fold scores the wrong studies just as quietly.
    Passed as one value, the three cannot come apart at a call site.

    It carries the operations every reading repeated by hand — combine a subset, name a subset, score a
    member alone — because `RankAverage().combine([predictions[i] for i in subset])` appeared in four
    methods and each copy was a chance to combine one list while naming another.
    """

    names: list[str]
    predictions: list[Float[np.ndarray, "studies targets"]]
    truth: Float[np.ndarray, "studies targets"]

    def __len__(self) -> int:
        """How many members — the range every subset enumeration runs over."""
        return len(self.predictions)

    def combined(self, subset: tuple[int, ...] | list[int]) -> Float[np.ndarray, "studies targets"]:
        """These members under the shipped rule, which is the flat per-label rank-average."""
        return RankAverage().combine([self.predictions[index] for index in subset])

    def named(self, subset: tuple[int, ...] | list[int]) -> str:
        """The subset as the names a submission would package."""
        return " + ".join(self.names[index] for index in subset)

    def alone(self, scores: LabelScores) -> list[str]:
        """Each member's own mean score on the shared studies — the column every mix is read against."""
        return [
            f"{name} alone: {scores.mean(member, self.truth):.4f}"
            for name, member in zip(self.names, self.predictions, strict=True)
        ]

    def split(self, roster: list[Path]) -> tuple["Pool", "Pool"]:
        """The roster as a pool, and the rest as candidates against it.

        The roster comes back NAMED rather than as bare arrays. A swap has to say which incumbent it
        displaces — `calib_span for calibrated_targets` is the whole content of the line — and a list of
        arrays cannot say it. An add did not need the names, which is why they were dropped here.

        The roster is read through the SAME join as the candidates rather than scored on its own: the
        shared-study intersection and the label-table refusal are the reason this type exists, and a
        roster loaded separately would be compared to candidates over a different denominator.

        A named roster member that carries no record is an error, not a skip. Elsewhere a missing
        record means an old checkpoint and skipping it costs one row; here it silently scores a
        four-member roster as three and every delta below it is against the wrong base.
        """
        wanted = {path.name for path in roster}
        found = {name for name in self.names if name in wanted}
        if found != wanted:
            message = f"roster members with no holdout record: {', '.join(sorted(wanted - found))}"
            raise ValueError(message)
        paired = list(zip(self.names, self.predictions, strict=True))
        held = [(name, member) for name, member in paired if name in wanted]
        rest = [(name, member) for name, member in paired if name not in wanted]
        return (
            Pool([name for name, _ in held], [member for _, member in held], self.truth),
            Pool([name for name, _ in rest], [member for _, member in rest], self.truth),
        )


class MemberCurve:
    """Mean score against ensemble size, over every subset of that size."""

    def __init__(self, scores: LabelScores, combine_seed: int = COMBINE_SEED) -> None:
        self.scores = scores
        self.combine_seed = combine_seed

    def sizes(self, pool: Pool) -> dict[int, list[float]]:
        """Every subset's score, grouped by how many members it holds.

        All subsets rather than a sample: with five members that is 31 combinations and a few seconds of
        numpy, and the spread within a size is the only error bar available without retraining.
        """
        return {size: [score for score, _ in self.subsets(pool, size)] for size in range(1, len(pool) + 1)}

    def subsets(self, pool: Pool, size: int) -> list[tuple[float, tuple[int, ...]]]:
        """Every subset of one size, as (score, member indices).

        The indices are the point. `sizes` reported that five members reach 0.8601 and never said WHICH
        five, so the best number this analysis produces could not be acted on — the next entry has to
        name its members. An analysis that reports a score nobody can reproduce as a submission is doing
        half its job.
        """
        return [
            (self.scores.mean(pool.combined(subset), pool.truth), subset)
            for subset in itertools.combinations(range(len(pool)), size)
        ]

    def greedy(self, pool: Pool) -> list[str]:
        """Forward selection: the member that adds most, added, until adding one stops paying.

        `subsets` is exhaustive and therefore exponential — five members is 31 combinations and sixteen
        is 65,535 — so the pool this project has actually accumulated cannot be searched that way at
        all. Greedy is O(n^2) combines and answers the question the exhaustive search is run for: which
        members to package, and where the count saturates.

        It is a HEURISTIC and the ledger's own numbers say why that is tolerable here: the curve is
        flat once past four members, so the difference between the greedy path and the best path is
        inside the offline-to-board transfer error that already governs the decision. What it must not
        be read as is a score — see this module's caller for the two reasons.
        """
        chosen: list[int] = []
        lines = []
        remaining = set(range(len(pool)))
        while remaining:
            score, index = max(
                (
                    (self.scores.mean(pool.combined([*chosen, candidate]), pool.truth), candidate)
                    for candidate in remaining
                ),
                key=lambda pair: pair[0],
            )
            chosen.append(index)
            remaining.discard(index)
            lines.append(f"greedy {len(chosen)}: {score:.4f}  +{pool.names[index]}")
        return lines

    def swap(self, roster: Pool, candidates: Pool) -> list[str]:
        """What each candidate is worth in the best SEAT it could take, holding the member count fixed.

        THE READING THE ROSTER POLICY ACTUALLY PERMITS, and the one nothing computed. `marginal` prices a
        candidate as an ADDITIONAL member, and `comp/champion.json`'s FOUR MEMBERS principle says the best
        subset falls monotonically past five — so every number `marginal` prints is the price of a move the
        pin forbids. The champion's own history is swaps: `res392_ws` out and `band_dense` in took 0.925 to
        0.928, and four different members occupying ONE seat is the experiment that produced the ruler
        caveat that pin now carries. The tool that decides which member to ship could not express any of it.

        The BEST seat, not every seat, because the decision is one line: whether this candidate belongs in
        the package at all, and at whose expense. Reporting four rows per candidate turns a roster of four
        against a dozen candidates into forty-eight numbers to eyeball, and the maximum is what a
        submission would take.

        A negative best-seat number is the honest answer that the roster should not be disturbed, and it
        is read CONSERVATIVELY — the opposite of what this docstring used to say (knee-c0u9). The old
        wording put a small negative "inside the region where this ruler is known wrong" on the 2.4x
        diversity under-pricing of knee-7d1, which reads as a licence to ship one. Two entries were fired
        on that theory and both were refuted. That 2.4x came off the DECODE axis, by collapsing three
        members into one pixel group, and it does not carry to a member swap. Three paired entries measure
        the opposite: the holdout RANKS correctly (0.8691 / 0.8678 / 0.8668 offline against 0.930 / 0.926
        / 0.925 on the board) and errs only in MAGNITUDE, understating the cost — -0.0013 offline became
        -0.004 and -0.0023 became -0.005. So `BOARD_COST_MULTIPLE` is what a negative is reported at, and
        a delta inside the noise floor is an argument for NOT spending the slot rather than a free look.

        `BOARD_RESOLVABLE` IS WHERE A POSITIVE STOPS MEANING ANYTHING, and the report says so rather than
        leaving the reader to remember it. Six entries now pair an offline joint-holdout score with a board
        score (knee-825t): the chain rises across the first 0.003 of offline spread and then stops, with
        `longseat` carrying +0.0012 — the largest single move in a sweep of all 56 same-split checkpoints,
        six times the runner-up — and TYING the board rather than moving it. 0.0015 is the value above
        that plateau and below the smallest gap the chain does track; it is a floor on what has been
        DEMONSTRATED, not a threshold anything is known to cross. So the trailing count is deliberately not
        a filter: the ordering is this tool's product and stays whole, and the line only refuses to let an
        argmax of +0.0002 read as a submission.
        """
        truth = candidates.truth
        base = self.scores.mean(RankAverage().combine(roster.predictions), truth)
        priced = []
        for candidate, name in zip(candidates.predictions, candidates.names, strict=True):
            seats = [
                (
                    self.scores.mean(
                        RankAverage().combine([*roster.predictions[:seat], candidate, *roster.predictions[seat + 1 :]]),
                        truth,
                    ),
                    roster.names[seat],
                )
                for seat in range(len(roster))
            ]
            score, displaced = max(seats, key=lambda pair: pair[0])
            priced.append((score, name, displaced))
        lines = [f"holding {len(roster)} members, the best seat each candidate could take:"]
        resolvable = sum(1 for score, _, _ in priced if score - base >= BOARD_RESOLVABLE)
        best = max(score for score, _, _ in priced) - base
        return [
            *lines,
            *(
                f"  {name} for {displaced}: {score:.4f}  ({score - base:+.4f})"
                for score, name, displaced in sorted(priced, key=lambda priced: -priced[0])
            ),
            f"{resolvable} of {len(priced)} clear +{BOARD_RESOLVABLE:.4f}, under which this ruler has "
            f"never been shown to pick a board tick",
            self.board_cost(best),
        ]

    @staticmethod
    def board_cost(best: float) -> str:
        """What the board would charge for shipping the best candidate, when the best one is negative.

        SAID ON THE REPORT rather than left in a docstring, because the reading it corrects is the one a
        reader reaches for unprompted: a best seat of -0.0010 sits inside the 0.0062 noise floor and
        looks free. It is not — three paired entries put the board cost at about `BOARD_COST_MULTIPLE`
        times the offline number (knee-c0u9), which turns that -0.0010 into roughly -0.003, twice the
        tick the same ruler needs +0.0015 to earn.
        """
        if best >= BOARD_RESOLVABLE:
            return f"best clears +{BOARD_RESOLVABLE:.4f}; above that this ruler orders correctly but saturates"
        if best >= 0:
            return f"best is +{best:.4f}, inside +{BOARD_RESOLVABLE:.4f}: this ruler cannot tell it from the incumbent"
        return (
            f"best is {best:+.4f} and NEGATIVE, which the board charges at about {BOARD_COST_MULTIPLE}x, "
            f"so roughly {best * BOARD_COST_MULTIPLE:+.4f} — not a free look (knee-c0u9)"
        )

    def marginal(self, roster: Pool, candidates: Pool) -> list[str]:
        """What each candidate would add to the roster that already ships, best first.

        THE NUMBER AN ARM IS ACTUALLY DECIDED ON, and the one nothing here computed. Every arm is read
        against a MATCHED control — the same recipe with one field changed — which is the right ruler
        for `did this knob do anything` and the wrong one for `does this earn a seat`. The two came
        apart once the roster stopped being the control's family: a member can beat its control by a
        clear margin and still be the pick that lowers the ensemble.

        `res392_ws` is the case that forces this. It scores 0.8523 alone, it was a member of the four
        that scored 0.925 on the board, and over 22 same-split members the greedy takes it NINTH — the
        first pick that does not raise the mix. Solo score ranked it into a submission; marginal
        contribution never would have.

        Distinct from `greedy`, which builds a roster from nothing and answers which members to
        package. This holds the roster FIXED, because the roster is a shipped artefact with a board
        number on it and the question an arm asks is whether to disturb it.

        It prices a FIFTH member, which the pin's four-member principle says is negative return — so read
        it as the diversity signal (does this member bring anything the four do not) and read `swap` for
        the decision.
        """
        truth = candidates.truth
        base = self.scores.mean(RankAverage().combine(roster.predictions), truth)
        added = [
            (self.scores.mean(RankAverage().combine([*roster.predictions, candidate]), truth), name)
            for candidate, name in zip(candidates.predictions, candidates.names, strict=True)
        ]
        lines = [f"roster of {len(roster)}: {base:.4f}"]
        lines += [
            f"  +{name}: {score:.4f}  ({score - base:+.4f})" for score, name in sorted(added, key=lambda pair: -pair[0])
        ]
        return lines

    def best(self, pool: Pool) -> list[str]:
        """The winning subset at each size, named — the line a submission is built from."""
        lines = []
        for size in range(1, len(pool) + 1):
            score, subset = max(self.subsets(pool, size), key=lambda pair: pair[0])
            lines.append(f"best {size}: {score:.4f}  {pool.named(subset)}")
        return lines

    def paired(self, pool: Pool) -> list[str]:
        """Each member's per-label delta against the FIRST, with the interval that says if it is real.

        Against the first because a paired interval is a two-arm statement and the caller names its
        control by putting it first — the same convention `run_comparison` uses.

        A star means the interval excludes zero. Without one, `run_comparison` prints the UNPAIRED
        bound, which ignores that both arms are scored on the same studies and runs about eleven times
        wider — wide enough that the calibrated-targets arm returned twelve deltas of which none
        cleared, and its whole falsifiable claim was per-label.
        """
        lines = []
        truth = pool.truth
        for index in range(1, len(pool)):
            control = self.scores.per_label(pool.predictions[0], truth)
            observed = self.scores.per_label(pool.predictions[index], truth)
            interval = self.scores.paired_interval(pool.predictions[0], pool.predictions[index], truth)
            lines.append(f"paired deltas, {pool.names[index]} against {pool.names[0]}:")
            for label, (low, high) in interval.items():
                clears = "*" if low > 0.0 or high < 0.0 else ""
                delta = observed[label] - control[label]
                lines.append(f"  {label:18s} {delta:+.4f}  [{low:+.4f}, {high:+.4f}]{clears}")
        return lines

    def oracle(self, pool: Pool) -> dict[str, float]:
        """The CEILING of combining these members, per label — against the best subset actually reachable.

        A ceiling by construction, and it must never be quoted as a score: it picks, for each of the
        twelve findings, whichever member happens to score that finding best ON THIS HOLDOUT. Nothing
        can be selected that way at submission time, because the choice uses the answers.

        WHAT IT IS FOR. It prices a whole class of work before any of it is done. Our current members
        are +0.02 better on this ruler than the five that scored 0.888, and the entry built from them
        scored 0.882 (knee-jk3). Either diversity dominates quality here, or holdout-to-leaderboard
        transfer is not what we assume. If this ceiling sits near the plain rank-average, then per-label
        weighting is worthless and so is another member of the same kind; if it sits far above, the gain
        is already sitting in the members we have and costs no training to collect.

        Reported beside the best PLAIN average so the two are read together — a ceiling alone tempts
        exactly the misreading it exists to prevent.
        """
        truth = pool.truth
        per_label = [self.scores.per_label(member, truth) for member in pool.predictions]
        labels = list(per_label[0])
        best = {label: max(scored[label] for scored in per_label) for label in labels}
        return {
            "oracle": float(np.nanmean(list(best.values()))),
            "average": self.scores.mean(pool.combined(list(range(len(pool)))), truth),
            "best_single": max(self.scores.mean(member, truth) for member in pool.predictions),
        }

    def votes(self, pool: Pool) -> list[str]:
        """Every voting rule on the SAME members — the ensemble axis that has never been varied.

        Count, which members, per-label selection and decode diversity have all been measured. The vote
        itself has not: every number in the ledger came out of a flat per-label rank-average, chosen
        because the metric reads rank order — which argues for RANKING and says nothing about weighting
        six members equally when 0.031 separates the best from the worst.

        The fitted rule is reported SPLIT-HALF, and that is the only column worth reading for it: its
        weights come off the same studies it would otherwise be scored on, and per-label selection died
        of exactly that — an in-sample gain that did not survive being chosen elsewhere.
        """
        flat = RankAverage()
        members, truth = pool.predictions, pool.truth
        lines = [f"{flat.named():22s} {self.scores.mean(flat.combine(members), truth):.4f}   (the shipped rule)"]
        lines.append(
            f"{LogitAverage().named():22s} {self.scores.mean(LogitAverage().combine(members), truth):.4f}"
            "   (keeps the margin a rank discards; exposed to a miscalibrated member)"
        )
        lines.append(
            f"{'QualityWeightedRank':22s} {self.quality_weighted(pool):.4f}   (split-half; weights "
            "fitted on the studies it is NOT scored on)"
        )
        return lines

    def quality_weighted(self, pool: Pool) -> float:
        """The weighted rule's HONEST score: weights fitted on one half, read on the other.

        Its own method rather than three lines inside `votes`, because it is the only rule here that fits
        anything and so the only one whose reported number depends on a split — which is a claim to state
        once, in the place that makes it, rather than a nested comprehension inside a report builder.
        """
        members, truth = pool.predictions, pool.truth
        halves = LabelCombine(self.scores).halves(len(truth), self.combine_seed)
        honest = [
            self.scores.mean(
                QualityWeightedRank([self.scores.mean(member[fit], truth[fit]) for member in members]).combine(
                    [member[report] for member in members]
                ),
                truth[report],
            )
            for fit, report in halves
        ]
        return float(np.mean(honest))

    def logged(self, pool: Pool, *, paired: bool) -> list[str]:
        """Every reading of these members' arrays — the half of this CLI that needs no GPU and no corpus.

        Split from `main` because it is the whole analysis: `main` finds checkpoints, builds a holdout
        and fills a cache, and this reads what came back. Together they made one function that could
        only run with a card, a corpus and minutes, which is to say one that nothing could exercise.
        """
        lines = list(self.report(self.sizes(pool)))
        # The subsets BY NAME, because the mean and the range describe the ensemble question while a
        # submission needs the answer: which members to package.
        lines += self.best(pool)
        if paired:
            lines += self.paired(pool)
        # The CEILING beside the reachable numbers, never alone: it selects per label using the answers,
        # so quoting it as a score would be quoting a cheat. What it prices is whether per-label
        # weighting - or another member of the same kind - is worth anything before either is built.
        ceiling = self.oracle(pool)
        lines.append(
            f"oracle {ceiling['oracle']:.4f}  vs plain average {ceiling['average']:.4f}  "
            f"vs best single {ceiling['best_single']:.4f}  "
            f"(headroom {ceiling['oracle'] - ceiling['average']:+.4f})"
        )
        # THE REACHABLE HALF OF THAT CEILING. The oracle says what per-label selection would be worth if
        # the answers were available; this says what is left once they are not. Its last row is the flat
        # rank-average under the same split-half estimator, so the two columns are a matched comparison
        # rather than a number beside a differently-measured number.
        lines += LabelCombine(self.scores).curve(pool.predictions, pool.truth, self.combine_seed)
        # THE VOTE ITSELF, last, because every line above it was produced BY the flat rule and this
        # is the one row that asks whether that rule was the right one to produce them with.
        return lines + self.votes(pool)

    def report(self, by_size: dict[int, list[float]]) -> list[str]:
        """One line per ensemble size: its mean, its range, and what the member added."""
        lines = []
        previous = None
        for size, values in sorted(by_size.items()):
            mean = float(np.mean(values))
            gain = "" if previous is None else f"   {mean - previous:+.4f} over {size - 1}"
            lines.append(f"{size} member(s): mean {mean:.4f}   [{min(values):.4f}, {max(values):.4f}]{gain}")
            previous = mean
        return lines


def main() -> None:
    """Score every subset of the given checkpoints on their shared holdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True, help="members sharing ONE split")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--slices", type=int, default=12)
    parser.add_argument("--overlap", action="store_true", help="every window start, not just disjoint ones")
    parser.add_argument(
        "--paired",
        action="store_true",
        help="per-label 95%% interval of each member's delta against the FIRST, by bootstrap over "
        "studies. The unpaired bound run_comparison prints is ~11x wider because it ignores that both "
        "arms are scored on the same studies - wide enough that a whole arm's per-label claim came "
        "back unread. Costs a thousand rescorings of arrays already in memory, and no GPU.",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="average over SpatialViews (rotations 0/-6/+6) as well as windows. A whole-stack member "
        "reads one window, so window averaging - worth a measured +0.0156 to the windowed members - is "
        "structurally unavailable to it and rotation is the only test-time average it can have. Costs "
        "three forward passes per member.",
    )
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    parser.add_argument(
        "--combine-seed",
        type=int,
        default=COMBINE_SEED,
        help="which half of the holdout fits the per-label member choice and which reports it. Fixed by "
        "default so the curve is reproducible; vary it to see how much of a gain is the split.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-read every member through its encoder instead of using cached probabilities. The cache "
        "keys on the checkpoint's size and mtime, the scored studies and the whole reading policy, so a "
        "stale hit should not be possible - this is the escape hatch for when it is.",
    )
    parser.add_argument(
        "--device",
        default=VisibleCards().device().type,
        choices=("cuda", "cpu"),
        help="where to read misses. This analysis runs WHILE the card belongs to a trainer, so saying "
        "cpu explicitly stays available — but the default now answers it too, since `VisibleCards` "
        "counts visible cards rather than asking whether CUDA is installed.",
    )
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = DataRoot.from_config(arguments.paths)
    device = torch.device(arguments.device)
    views = SpatialViews() if arguments.tta else None
    predictions = HoldoutPredictions(device, views)
    scores = LabelScores()

    # EACH MEMBER'S OWN READING, never one cache and one size for all of them. This CLI loaded a single
    # 336 px cache, passed IMAGE_SIZE as the read size for every member, and windowed by GROUP whatever
    # the member was - so a whole-stack member was scored on three slices at a time and a member fitted
    # on the 448 px cache could not be scored at all. It is the tool that decides WHICH members to
    # submit, so a wrong number here chooses the wrong ensemble.
    # KEYED ON THE CACHE'S OWN NAME, which is the only thing that promises two members read the same
    # pixels. This grouped by size, depth and scheme — three of the six fields that name a cache — so
    # two members differing in crop, band or tie-break shared one decode, and the member built its own
    # `StudyCache` from the three it happened to remember.
    readings = MemberReadings([Member.from_checkpoint(path) for path in arguments.checkpoints], root.processed())

    # The split is taken from the FIRST cache and reused, so every member is scored on the same studies
    # in the same order. Any other cache that disagrees about that order is refused: its rows would be
    # silently misaligned against the truth vector, which is a wrong number rather than an error.
    corpus, reference = readings.reference()
    folds = ReportFolds(arguments.folds, arguments.split_seed)
    _, holdout = folds.holdout(folds.assign(str(root.raw() / "train.csv"), reference), arguments.fold)
    truth = Targets.load(arguments.labels, root.raw() / "train.csv", reference).scores[holdout]

    # Keyed on the studies actually scored, so a different fold or split cannot hand back these rows.
    cache = PredictionCache(
        root.processed() / "holdout_predictions",
        [reference[i] for i in holdout],
        predictions.views,
        refresh=arguments.refresh,
    )

    per_member: list[NDArray[np.float64]] = []
    names: list[str] = []
    # The walk is the long part — every member that is not already cached is a full inference pass, and
    # a sweep over the pool runs unattended for hours past the idle timer (knee-wyzj).
    with Wakelock():
        for member, slab, held in readings.slabs(holdout, reference, corpus):
            window = member.window(held.slices, overlap=arguments.overlap)
            scored = predictions.cached(member, slab, cache, window)
            per_member.append(scored)
            names.append(member.file.name)
            logger.info("%s alone: %.4f", member.file.name, scores.mean(scored, truth))

    pool = Pool(names, per_member, truth)
    for line in MemberCurve(scores, arguments.combine_seed).logged(pool, paired=arguments.paired):
        logger.info(line)


if __name__ == "__main__":
    main()
