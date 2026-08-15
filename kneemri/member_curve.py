"""What does each additional ensemble member actually buy?

`python -m kneemri.member_curve --checkpoints a.pt b.pt c.pt --labels labels.csv`

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
from pathlib import Path

import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from core.paths import DataRoot
from kneemri.ensemble import Ensemble
from kneemri.folds import ReportFolds
from kneemri.holdout_predictions import HoldoutPredictions
from kneemri.holdout_slab import HoldoutSlab
from kneemri.label_combine import LabelCombine
from kneemri.members import Member
from kneemri.prediction_cache import PredictionCache
from kneemri.scores import LabelScores
from kneemri.targets import Targets
from kneemri.test_time import SpatialViews

logger = logging.getLogger(__name__)

COMBINE_SEED = 2026
"""Which studies fit the per-label choice and which report it — the corpus split seed, since both are
arbitrary-but-fixed partitions of the same corpus and a second free number would only be a second thing
to accidentally vary."""


class MemberCurve:
    """Mean score against ensemble size, over every subset of that size."""

    def __init__(self, scores: LabelScores, combine_seed: int = COMBINE_SEED) -> None:
        self.scores = scores
        self.combine_seed = combine_seed

    def sizes(self, per_member: list[NDArray[np.float64]], truth: Float[np.ndarray, "b t"]) -> dict[int, list[float]]:
        """Every subset's score, grouped by how many members it holds.

        All subsets rather than a sample: with five members that is 31 combinations and a few seconds of
        numpy, and the spread within a size is the only error bar available without retraining.
        """
        return {
            size: [score for score, _ in self.subsets(per_member, truth, size)]
            for size in range(1, len(per_member) + 1)
        }

    def subsets(
        self, per_member: list[NDArray[np.float64]], truth: Float[np.ndarray, "b t"], size: int
    ) -> list[tuple[float, tuple[int, ...]]]:
        """Every subset of one size, as (score, member indices).

        The indices are the point. `sizes` reported that five members reach 0.8601 and never said WHICH
        five, so the best number this analysis produces could not be acted on — the next entry has to
        name its members. An analysis that reports a score nobody can reproduce as a submission is doing
        half its job.
        """
        return [
            (self.scores.mean(Ensemble.combine([per_member[index] for index in subset]), truth), subset)
            for subset in itertools.combinations(range(len(per_member)), size)
        ]

    def best(
        self, per_member: list[NDArray[np.float64]], truth: Float[np.ndarray, "b t"], names: list[str]
    ) -> list[str]:
        """The winning subset at each size, named — the line a submission is built from."""
        lines = []
        for size in range(1, len(per_member) + 1):
            score, subset = max(self.subsets(per_member, truth, size), key=lambda pair: pair[0])
            lines.append(f"best {size}: {score:.4f}  " + " + ".join(names[index] for index in subset))
        return lines

    def paired(
        self, per_member: list[NDArray[np.float64]], truth: Float[np.ndarray, "b t"], names: list[str]
    ) -> list[str]:
        """Each member's per-label delta against the FIRST, with the interval that says if it is real.

        Against the first because a paired interval is a two-arm statement and the caller names its
        control by putting it first — the same convention `run_comparison` uses.

        A star means the interval excludes zero. Without one, `run_comparison` prints the UNPAIRED
        bound, which ignores that both arms are scored on the same studies and runs about eleven times
        wider — wide enough that the calibrated-targets arm returned twelve deltas of which none
        cleared, and its whole falsifiable claim was per-label.
        """
        lines = []
        for index in range(1, len(per_member)):
            control = self.scores.per_label(per_member[0], truth)
            observed = self.scores.per_label(per_member[index], truth)
            interval = self.scores.paired_interval(per_member[0], per_member[index], truth)
            lines.append(f"paired deltas, {names[index]} against {names[0]}:")
            for label, (low, high) in interval.items():
                clears = "*" if low > 0.0 or high < 0.0 else ""
                delta = observed[label] - control[label]
                lines.append(f"  {label:18s} {delta:+.4f}  [{low:+.4f}, {high:+.4f}]{clears}")
        return lines

    def oracle(self, per_member: list[NDArray[np.float64]], truth: Float[np.ndarray, "b t"]) -> dict[str, float]:
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
        per_label = [self.scores.per_label(member, truth) for member in per_member]
        labels = list(per_label[0])
        best = {label: max(scored[label] for scored in per_label) for label in labels}
        combined = Ensemble.combine(per_member)
        return {
            "oracle": float(np.nanmean(list(best.values()))),
            "average": self.scores.mean(combined, truth),
            "best_single": max(self.scores.mean(member, truth) for member in per_member),
        }

    def logged(
        self,
        per_member: list[NDArray[np.float64]],
        truth: Float[np.ndarray, "b t"],
        names: list[str],
        *,
        paired: bool,
    ) -> list[str]:
        """Every reading of these members' arrays — the half of this CLI that needs no GPU and no corpus.

        Split from `main` because it is the whole analysis: `main` finds checkpoints, builds a holdout
        and fills a cache, and this reads what came back. Together they made one function that could
        only run with a card, a corpus and minutes, which is to say one that nothing could exercise.
        """
        lines = list(self.report(self.sizes(per_member, truth)))
        # The subsets BY NAME, because the mean and the range describe the ensemble question while a
        # submission needs the answer: which members to package.
        lines += self.best(per_member, truth, names)
        if paired:
            lines += self.paired(per_member, truth, names)
        # The CEILING beside the reachable numbers, never alone: it selects per label using the answers,
        # so quoting it as a score would be quoting a cheat. What it prices is whether per-label
        # weighting - or another member of the same kind - is worth anything before either is built.
        ceiling = self.oracle(per_member, truth)
        lines.append(
            f"oracle {ceiling['oracle']:.4f}  vs plain average {ceiling['average']:.4f}  "
            f"vs best single {ceiling['best_single']:.4f}  "
            f"(headroom {ceiling['oracle'] - ceiling['average']:+.4f})"
        )
        # THE REACHABLE HALF OF THAT CEILING. The oracle says what per-label selection would be worth if
        # the answers were available; this says what is left once they are not. Its last row is the flat
        # rank-average under the same split-half estimator, so the two columns are a matched comparison
        # rather than a number beside a differently-measured number.
        return lines + LabelCombine(self.scores).curve(per_member, truth, self.combine_seed)

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
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    root = DataRoot.from_config(arguments.paths)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    members = [Member.from_checkpoint(checkpoint) for checkpoint in arguments.checkpoints]
    readings = [(member.cache(root.processed()), member) for member in members]
    caches = {cache.key: cache for cache, _ in readings}
    keys = list(caches)

    # The split is taken from the FIRST cache and reused, so every member is scored on the same studies
    # in the same order. Any other cache that disagrees about that order is refused: its rows would be
    # silently misaligned against the truth vector, which is a wrong number rather than an error.
    corpus, reference = caches[keys[0]].load()
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
    for key in keys:
        if key != keys[0]:
            corpus, identifiers = caches[key].load()
            if identifiers != reference:
                message = f"cache {key} indexes studies in a different order — its rows would not line up"
                raise ValueError(message)
        slab = HoldoutSlab(corpus, holdout)
        for reading in (member for cache, member in readings if cache.key == key):
            window = reading.window(corpus.slices, overlap=arguments.overlap)
            scored = predictions.cached(reading, slab, cache, window)
            per_member.append(scored)
            names.append(reading.file.name)
            logger.info("%s alone: %.4f", reading.file.name, scores.mean(scored, truth))

    for line in MemberCurve(scores, arguments.combine_seed).logged(per_member, truth, names, paired=arguments.paired):
        logger.info(line)


if __name__ == "__main__":
    main()
