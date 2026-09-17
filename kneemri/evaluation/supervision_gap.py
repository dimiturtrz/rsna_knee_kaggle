"""Is a finding held back by its LABELS or by the MODEL? `python -m kneemri.evaluation.supervision_gap`.

Every local number in this project scores agreement with LLM-read report labels, while the competition
scores agreement with radiologist annotations — and where the report says YES the radiologist agrees
only 65.3% of the time. A third of the positives we train on are not positives.

That fact alone says the labels are noisy. It does NOT say what to do about it, because the answer is
different for different findings, and this module is the measurement that separates them. Two numbers
per finding against the SAME radiologist truth, on the studies train.csv annotates:

    REFERENCE — the report label itself, used as a predictor. How much signal the supervision carries.
    ACHIEVED  — what a model fitted on those labels actually reaches.

"CEILING" IS THE WRONG WORD FOR THE REFERENCE and this module deliberately avoids it: a model fitted on
noisy labels CAN beat them by looking at the image, and measured on the published members it does so on
half the findings. Effusion's report label predicts the radiologist at 0.8137 and the model reaches
0.9453 — it is correcting the reader. The reference is a comparison, never a bound.

WHAT THE SPLIT DECIDES, AND WHOSE MODEL IT IS A FACT ABOUT. Where the model already beats the report,
the reader is the weak part and a better label reader is the lever. Where the report describes a finding
well and the model cannot reproduce it, the signal is in the label and the pixels are where it is being
lost. WHICH FINDINGS FALL ON WHICH SIDE IS A PROPERTY OF THE MEMBER, NOT OF THIS CORPUS. On the
published pool the split reads `labels - model: +0.0266 [-0.0082, +0.0610]` and puts six findings below
the line, ACL and Lateral Meniscus deepest. On our champion, measured 2026-09-17 on the same 58 studies
against the same truth, it reads `-0.0616 [-0.0850, -0.0385]` with all twelve above it — Lateral
Meniscus moves by +0.2429 between the two. So quote this tool's verdict with the `--records` it was run
on, never as a statement about the task. See
`interpretations/kneemri/2026-08-14_labels-or-pixels.md`.

IT NEEDS PREDICTIONS THAT DID NOT TRAIN ON THE ANNOTATIONS. Most of ours are fitted with the annotated
studies upweighted, so scoring those here would be scoring training data; a member fitted
`--hold-annotations` is exempt and is why `--records` can now be pointed at the champion. The published
members are fitted on `report_labels_v2.csv` with no annotation weighting and carry out-of-fold
predictions for every study, so the radiologist labels are genuinely unseen by them. See
`kneemri.evaluation.oof_oracle`.

THE DENOMINATOR. 58 annotated studies, 696 cells, 34.5% positive, per-finding positives 9 to 35. The
aggregate and the bands that exclude zero are claims; a single finding's delta is a shape.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float

from core import cli
from core.bootstrap import STUDY_RESAMPLES
from core.findings import TARGETS
from kneemri.evaluation.gold_rows import PRESENT, GoldRows
from kneemri.evaluation.published_ruler import PublishedRuler
from kneemri.evaluation.scores import LabelScores
from kneemri.inference.vote import RankAverage

logger = logging.getLogger(__name__)

CUTS = (0.5, 0.6, 0.75, 0.88)
"""The reader emits 0.08/0.28/0.68/0.82/0.94, so these cuts fall BETWEEN its levels: 0.5 admits the
mild band, 0.75 excludes it, 0.88 keeps only the top level."""


@dataclass(frozen=True)
class Bottleneck:
    """Which half of the pipeline is the weaker one, per finding — the split, as data.

    IT DECIDES WHAT AN ARM IS ALLOWED TO BE JUDGED ON, which is why it is a value and not a printed
    line. A better label reader can only pay where the reader is the weak part; showing the model more
    of the image can only pay where it is not. Averaged over all twelve, either change is diluted by
    the six it cannot affect — six findings of noise added to six of signal, on a ruler whose floor is
    already 0.0053. The rule ("a label-reader change is evaluated on `label_side`, an imaging change on
    `model_side`") lived in a bead while the measurement lived here, which is the split this project
    keeps paying for: the knowledge existed and did not reach the place that needed it.
    """

    label_side: tuple[str, ...]
    """The model already beats its own labels here, so the READER is what holds these back."""

    model_side: tuple[str, ...]
    """The report describes these well and the model cannot reproduce it — a VISION failure."""


class SupervisionGap:
    """The labels' own predictive power beside the model's, per finding."""

    def __init__(self, scores: LabelScores | None = None) -> None:
        self.scores = scores or LabelScores()
        self.gold = GoldRows()

    def annotated(self, train_table: Path) -> tuple[list[str], Float[np.ndarray, "studies targets"]]:
        """The annotated studies and their annotations — see `GoldRows`, which owns the read."""
        return self.gold.annotated(train_table)

    def reported(self, labels: Path, identifiers: list[str]) -> Float[np.ndarray, "studies targets"]:
        """A report table on those studies in that order — see `GoldRows`, which owns the read."""
        return self.gold.reported(labels, identifiers)

    def against(
        self, values: Float[np.ndarray, "studies targets"], truth: Float[np.ndarray, "studies targets"]
    ) -> dict[str, float]:
        """One AUC per finding, of anything shaped like predictions against the annotations.

        The same method serves both columns of the report — the labels used as a predictor and the
        model's probabilities — because they are the same measurement and giving them separate methods
        would let the two drift into meaning different things.
        """
        return {
            target: self.scores.area_under_curve(
                values[:, column].astype(np.float64), (truth[:, column] > PRESENT).astype(np.int64)
            )
            for column, target in enumerate(TARGETS[: values.shape[1]])
        }

    def thresholds(
        self,
        values: Float[np.ndarray, "studies targets"],
        truth: Float[np.ndarray, "studies targets"],
        cuts: tuple[float, ...] = CUTS,
    ) -> list[str]:
        """Where to call a graded report label POSITIVE, priced against the radiologist.

        THE POSITIVE RATE IS THE HEADLINE. Our ruler and our training targets binarise at 0.5, which
        calls 45.5% of cells positive while the radiologist calls 34.4% — we over-label by a third, and
        it is all one band: the reader's lowest YES level, which `Supervision` measures the radiologist
        agreeing with only 37.8% of the time, and which `knee-37f` found is the report faithfully
        recording a trace or low-grade finding the annotator did not count.

        WHAT THIS IS NOT. It is not an argument to hard-threshold the training targets — the three YES
        levels are live signal the model already orders correctly (11 of 11 findings, knee-mxj), and
        cutting them to binary throws that away. The graded calibration is the way to use it.

        WHAT IT IS. Evidence that the local ruler measures a measurably different task from the one
        scored: a positive set a third larger than the radiologist's. That is a candidate mechanism for
        the local-to-leaderboard compression knee-7d1 measured from the other side, and it is testable —
        a ruler cut at 0.75 should track the leaderboard better than one cut at 0.5.

        Changing the ruler is NOT free and is deliberately not done here: every historical number was
        computed at 0.5, so moving it silently would make the ledger incomparable with itself.
        """
        lines = [
            f"{truth.size} cells; the radiologist calls {float((truth > PRESENT).mean()):.1%} positive",
            f"{'cut':>6s} {'ours':>8s} {'agree':>8s} {'precision':>10s} {'recall':>8s}",
        ]
        positive = truth > PRESENT
        for cut in cuts:
            said = values > cut
            hit = float((said & positive).sum())
            lines.append(
                f"{cut:6.2f} {float(said.mean()):8.1%} {float((said == positive).mean()):8.1%} "
                f"{hit / max(float(said.sum()), 1.0):10.1%} {hit / max(float(positive.sum()), 1.0):8.1%}"
            )
        return lines

    def aligned(
        self, labels: Path, train_table: Path, published: Path
    ) -> tuple[Float[np.ndarray, "kept targets"], Float[np.ndarray, "kept targets"], Float[np.ndarray, "kept targets"]]:
        """The report labels, the held-out predictions and the annotations, on one set of studies in one order.

        Three files that each carry a different subset in a different order, and the alignment between
        them is the part that can silently be wrong: a prediction row read against another study's
        annotations produces a number rather than an error. It lives here, once, because both readers of
        this measurement — the report and `run_comparison`'s split — would otherwise each write it.
        """
        names, per_seed = PublishedRuler(published).per_seed()
        return self.matched(labels, train_table, names, RankAverage().combine(per_seed))

    def matched(
        self, labels: Path, train_table: Path, names: list[str], combined: Float[np.ndarray, "read targets"]
    ) -> tuple[Float[np.ndarray, "kept targets"], Float[np.ndarray, "kept targets"], Float[np.ndarray, "kept targets"]]:
        """The same three arrays, from predictions a caller already holds rather than the published pool.

        Split out because the published members are no longer the only readings that never saw the
        annotations: `annotated_readings` produces them for a run fitted with `--hold-annotations`, which
        is the only way this split can be measured on a model WE fitted. The alignment is the part that
        can silently be wrong, so it stays here, once, whoever the predictions came from.
        """
        identifiers, truth = self.annotated(train_table)
        position = {name: index for index, name in enumerate(names)}
        keep = [index for index, name in enumerate(identifiers) if name in position]
        rows = np.array([position[identifiers[index]] for index in keep], dtype=np.int64)
        values = self.reported(labels, [identifiers[index] for index in keep])
        return values, np.asarray(combined[rows], dtype=np.float32), truth[keep]

    def measured(self, labels: Path, train_table: Path, published: Path) -> Bottleneck:
        """The split itself, from the three files it is measured on — the value a comparison asks for.

        `run_comparison` needs the Bottleneck and nothing else this module computes, and handing it the
        three paths keeps the alignment on this side of the boundary rather than making every caller
        reassemble it correctly.
        """
        values, predictions, annotations = self.aligned(labels, train_table, published)
        return self.bottlenecked(self.against(values, annotations), self.against(predictions, annotations))

    def bottlenecked(self, reference: dict[str, float], achieved: dict[str, float]) -> Bottleneck:
        """Split the findings by whether their LABELS or their MODEL is the weaker half.

        Ordered by the gap rather than by either score, because the question is which side is weaker:
        an ordering by score puts the findings we do well beside the findings whose labels are easy,
        and those are not the same group.

        A finding whose gap will not compare — either score missing or NaN — falls on the label side
        rather than being dropped, because a finding silently absent from both groups is a finding no
        arm is ever judged on.
        """
        shared = [target for target in reference if target in achieved]
        ranked = sorted(shared, key=lambda target: achieved[target] - reference[target], reverse=True)
        model_side = tuple(target for target in ranked if achieved[target] - reference[target] < 0)
        return Bottleneck(
            label_side=tuple(target for target in ranked if target not in set(model_side)),
            model_side=model_side,
        )

    def verdict(self, low: float, high: float) -> str:
        """Which half the paired interval convicts, and it has THREE answers, not two.

        The sign convention is `labels - model`, so a wholly POSITIVE interval says the labels score
        higher and the model is the weaker half, and a wholly NEGATIVE one says the model beats its own
        supervision and the READER is what to improve. Only an interval containing zero is silent.

        This was one-sided until 2026-08-29 and tested `low > 0` alone, so the model-wins case printed
        "SPANS zero" over an interval of [-0.0742, -0.0255] — the first time our own held-annotation
        members were read through it, and the exact reading that would have steered the queue. The
        per-finding count on the next line had been two-sided all along, so the block contradicted
        itself in the same breath: two resolved bands under an aggregate calling itself unresolved.
        """
        if low > 0.0:
            return "  the interval EXCLUDES zero: the model is the weaker half on our own evidence"
        if high < 0.0:
            return "  the interval EXCLUDES zero: the model beats its own labels, so the READER is the weaker half"
        return "  the interval SPANS zero: these studies cannot say which half is weaker"

    def separation(
        self,
        values: Float[np.ndarray, "kept targets"],
        predictions: Float[np.ndarray, "kept targets"],
        truth: Float[np.ndarray, "kept targets"],
        resamples: int = STUDY_RESAMPLES,
    ) -> list[str]:
        """How far the model is from its own labels, WITH the interval the aggregate never carried.

        `report` prints two aggregates and their difference. That difference is the number this project
        reasons from — it decides whether the next arm buys a better READER or a better MODEL — and it
        was published unbounded: `2026-08-25_forum-bottleneck-vs-our-own-diagnostic.md` says so in its
        own words, that 58 studies and ~696 cells separate the 0.027 from zero by nothing stated. A
        number that steers the queue and cannot say whether it is zero is not a measurement yet.

        PAIRED, and it must be. Both columns are scored on the SAME studies against the SAME
        annotations, so the difficulty of a study — which dominates either absolute AUC on 58 rows —
        cancels in the difference. `mean_interval` on each column separately would report two wide bands
        that overlap heavily and say nothing about whether one beats the other, which is the exact
        mistake `paired_mean_interval` was written to stop.

        The per-finding count comes with the same warning `MemberCurve` prints for a searched roster:
        twelve bands read together are twelve chances to exclude zero, so the count is a shape of the
        evidence and never twelve findings. The AGGREGATE is the one comparison named in advance.
        """
        low, high = self.scores.paired_mean_interval(predictions, values, truth, resamples)
        bands = self.scores.paired_interval(predictions, values, truth, resamples)
        resolved = [target for target, (lower, upper) in bands.items() if lower > 0.0 or upper < 0.0]
        return [
            f"labels - model: {self.scores.mean(values, truth) - self.scores.mean(predictions, truth):+.4f}  "
            f"[{low:+.4f}, {high:+.4f}]  over {truth.shape[0]} studies",
            self.verdict(low, high),
            f"  {len(resolved)} of {len(bands)} per-finding bands exclude zero — twelve reads, so a shape, "
            f"not twelve findings: {', '.join(resolved) or 'none'}",
        ]

    def confusion(
        self,
        values: Float[np.ndarray, "kept targets"],
        truth: Float[np.ndarray, "kept targets"],
        cut: float = PRESENT,
    ) -> list[str]:
        """Which DIRECTION the reader is wrong in, per finding — the half an AUC cannot show.

        `report`'s labels column ranks the reader; it cannot say whether a weak column is the reader
        inventing findings or missing them, and those two want opposite treatments. `thresholds` prices
        the same question over all 696 cells at once, which averages a column that over-labels against a
        column that under-labels into a middling aggregate.

        FALSE POSITIVE is over the gold NEGATIVES and FALSE NEGATIVE over the gold POSITIVES, so each
        rate has its own denominator printed beside it: a finding with four positives can post a 25% FN
        on one cell, and a rate whose denominator is not stated is the defect this project keeps paying
        for. No band is drawn — 58 studies binarised per column is far too little for one, and a rate
        that arrived with an interval would be read as a claim about a finding rather than a shape.
        """
        said, positive = values > cut, truth > PRESENT
        lines = [f"{'finding':20s} {'gold+':>6s} {'reader+':>8s} {'FP':>8s} {'FN':>8s}"]
        for column, target in enumerate(TARGETS[: values.shape[1]]):
            hit, gold = said[:, column], positive[:, column]
            positives, negatives = int(gold.sum()), int((~gold).sum())
            false_positive = float((hit & ~gold).sum()) / max(negatives, 1)
            false_negative = float((~hit & gold).sum()) / max(positives, 1)
            lines.append(
                f"{target:20s} {positives:6d} {int(hit.sum()):8d} "
                f"{false_positive:7.1%}{'':1s} {false_negative:7.1%}"
                f"   ({negatives} neg, {positives} pos)"
            )
        return lines

    def report(self, reference: dict[str, float], achieved: dict[str, float]) -> list[str]:
        """Every finding ordered by what holds it back, with the dividing line drawn explicitly.

        Sorted by the gap rather than by either number, because the QUESTION is which side is weaker —
        an ordering by score would put the findings we do well beside the findings whose labels are
        easy, and those are not the same group.
        """
        shared = [target for target in reference if target in achieved]
        split = self.bottlenecked(reference, achieved)
        lines = [
            f"aggregate: labels {np.nanmean([reference[t] for t in shared]):.4f}   "
            f"model {np.nanmean([achieved[t] for t in shared]):.4f}",
            f"{'finding':20s} {'labels':>8s} {'model':>8s} {'model-labels':>13s}",
        ]

        def row(target: str) -> str:
            return (
                f"{target:20s} {reference[target]:8.4f} {achieved[target]:8.4f} "
                f"{achieved[target] - reference[target]:+13.4f}"
            )

        lines += [row(target) for target in split.label_side]
        lines.append("  --- above: the model beats its own labels, so the READER is the weak part ---")
        lines += [row(target) for target in split.model_side]
        return lines


def main() -> None:
    """Split the findings by whether their labels or their model is the weaker half."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, required=True, help="the report-derived table to price")
    command.parser.add_argument(
        "--records",
        type=Path,
        nargs="*",
        default=[],
        help="OUR members, each fitted with --hold-annotations, read over every annotated study instead "
        "of the published pool's out-of-fold arrays. This is the only way the split can be measured on a "
        "model we fitted; without it the page describes a stranger's model and says so.",
    )
    command.parser.add_argument("--device", default=None, help="where to read prediction-cache misses")
    arguments = command.parsed()

    root = command.rooted(arguments)
    gap = SupervisionGap()
    train_table = root.raw() / "train.csv"
    if arguments.records:
        import torch  # noqa: PLC0415 - CLI-only, and the published path must not pay for a GPU import

        from kneemri.device import VisibleCards  # noqa: PLC0415 - CLI-only
        from kneemri.evaluation.annotated_readings import AnnotatedReadings  # noqa: PLC0415 - CLI-only

        device = torch.device(arguments.device or VisibleCards().device().type)
        identifiers, _ = gap.annotated(train_table)
        names, per_member = AnnotatedReadings(root, device).readings(arguments.records, identifiers)
        values, predictions, annotations = gap.matched(
            arguments.labels, train_table, names, RankAverage().combine([values for _, values in per_member])
        )
    else:
        values, predictions, annotations = gap.aligned(arguments.labels, train_table, root.processed() / "published")
    logger.info("%d annotated studies with a held-out prediction", len(values))

    command.report(gap.report(gap.against(values, annotations), gap.against(predictions, annotations)))
    # The aggregate above steers which arm gets built next, so it never prints without its interval.
    command.report(gap.separation(values, predictions, annotations))
    # WHERE the graded label becomes a positive, which decides the base rate our ruler scores against.
    command.report(gap.thresholds(values, annotations))
    # And WHICH DIRECTION each column is wrong in, which decides whether a reader fix adds or removes.
    command.report(gap.confusion(values, annotations))


if __name__ == "__main__":
    main()
