"""Is the medial-over-lateral deficit PREVALENCE, or is it vision? `python -m kneemri.evaluation.prevalence_match`.

knee-yrg measured, corpus-wide on the published out-of-fold ruler, that the two compartment pairs are
far apart: Medial Meniscus − Lateral Meniscus = +0.0622, Medial OA − Lateral OA = +0.0472. It refuted
the laterality-flip explanation (the gap survives where the DICOM tag and the geometric rule agree) and
knee-1v0n refuted the slice-band one (the band's cut is symmetric, and a symmetric deletion cannot make
an asymmetric deficit). Two candidates were left untested, and this is the cheaper one: the lateral
findings have about HALF the positives (1,114 vs 2,341 meniscus, 1,082 vs 1,612 OA).

WHY PREVALENCE COULD DO IT WITHOUT ANY VISION DIFFERENCE. An AUC does not move with the base rate on
its own — it is a rank statistic over positive/negative pairs — but the LEARNING does: a head with half
the positives sees half the gradient signal for that finding, and the estimate itself is noisier.

WHAT THIS MEASURES AND WHAT IT CANNOT. It subsamples the RICH label down to the POOR label's exact
positive and negative counts and recomputes the AUC many times, giving the rich finding's score at the
poor finding's sample size. That controls the MEASUREMENT half of the story completely, and the
TRAINING half not at all — the rich head was still fitted on every positive it had. So a gap that
SURVIVES matching is not explained by measurement precision and is either vision or supervision volume;
a gap that DISAPPEARS was never a finding about anatomy. Only the first outcome is a claim about the
images, and the report says so in the line it prints rather than leaving it to the reader.

THE DENOMINATOR IS PRINTED WITH EVERY ROW, because that is the mistake this project has already paid
for twice: a rate computed inside the subset that happened to parse is not a rate over the corpus.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float

from core.findings import TARGETS
from kneemri.evaluation.published_ruler import PublishedRuler
from kneemri.evaluation.scores import BOOTSTRAP_SEED, MIDPOINT, RESAMPLES, LabelScores
from kneemri.evaluation.supervision_gap import SupervisionGap
from kneemri.inference.vote import RankAverage

logger = logging.getLogger(__name__)

PAIRS: tuple[tuple[str, str], ...] = (("Medial Meniscus", "Lateral Meniscus"), ("Medial OA", "Lateral OA"))
"""The two compartment pairs the asymmetry was measured on — a rich finding beside its lateral twin."""

LOW_PERCENTILE = 2.5
HIGH_PERCENTILE = 97.5


@dataclass(frozen=True)
class MatchedGap:
    """One pair's answer: the rich finding scored at the poor finding's sample size.

    BESIDE ITS PRODUCER rather than returned as a tuple, because the verdict is an arithmetic relation
    between four of these numbers and a caller reconstructing it from positions is how a comparison gets
    read backwards.
    """

    rich: str
    poor: str
    poor_auc: float
    rich_auc: float
    difference: tuple[float, float, float]
    """The rich MINUS the poor AUC, both drawn at shared counts: 2.5th percentile, median, 97.5th."""

    positives: int
    negatives: int
    """The counts BOTH sides were scored at — the denominator, carried rather than described."""

    def explained(self) -> bool:
        """Whether sample size accounts for the gap: the matched difference straddles zero."""
        low, _, high = self.difference
        return low <= 0.0 <= high

    def line(self) -> str:
        """One row, ending in the verdict rather than in numbers a reader has to combine."""
        low, median, high = self.difference
        verdict = "explained by prevalence" if self.explained() else "SURVIVES matching"
        return (
            f"{self.rich:17s} {self.rich_auc:.4f}  {self.poor:17s} {self.poor_auc:.4f}  "
            f"full {self.rich_auc - self.poor_auc:+.4f}  matched {median:+.4f} [{low:+.4f}, {high:+.4f}]  "
            f"n={self.positives}+/{self.negatives}-  {verdict}"
        )


class PrevalenceMatch:
    """Each compartment pair's gap, re-measured with the two findings' sample sizes made equal."""

    def __init__(self, scores: LabelScores | None = None, resamples: int = RESAMPLES) -> None:
        self.scores = scores or LabelScores()
        self.resamples = resamples

    def counted(self, truth: Float[np.ndarray, "studies targets"], target: str) -> tuple[int, int]:
        """How many positives and negatives a finding carries, binarised where the rest of the project cuts."""
        positive = truth[:, TARGETS.index(target)] > MIDPOINT
        return int(positive.sum()), int((~positive).sum())

    def matched(
        self,
        predictions: Float[np.ndarray, "studies targets"],
        truth: Float[np.ndarray, "studies targets"],
        rich: str,
        poor: str,
    ) -> MatchedGap:
        """Both findings drawn repeatedly at ONE shared positive and negative count, and differenced.

        BOTH SIDES ARE SUBSAMPLED, not just the rich one, and that is a correction to the obvious
        design: the rich finding has MORE positives and therefore FEWER negatives, so "draw the rich
        label at the poor label's counts" is unsatisfiable — Medial Meniscus has 2,341 positives and
        cannot supply the 3,293 negatives its lateral twin has. Taking the smaller of each count gives
        one population both findings can be scored on, at one base rate, which is the comparison the
        question actually asks for.

        Positives and negatives are subsampled SEPARATELY and WITHOUT replacement, so every draw is a
        genuine subset of the corpus — a bootstrap with replacement would also add duplicate studies,
        which is a different question and a slightly optimistic one.

        The DIFFERENCE is taken within each draw rather than between two intervals: the two findings are
        scored on studies drawn independently but at identical counts, and a paired difference is what
        the verdict reads, so a hard corpus draw does not count twice.

        Seeded from the project's one bootstrap seed, so an interval quoted in a bead is the interval
        anyone else recomputes, and so the unit gate's ban on unseeded randomness is honoured.
        """
        generator = np.random.default_rng(BOOTSTRAP_SEED)
        columns = {name: TARGETS.index(name) for name in (rich, poor)}
        values = {name: predictions[:, column].astype(np.float64) for name, column in columns.items()}
        labels = {name: (truth[:, column] > MIDPOINT).astype(np.int64) for name, column in columns.items()}
        counts = [self.counted(truth, name) for name in (rich, poor)]
        positives = min(count[0] for count in counts)
        negatives = min(count[1] for count in counts)

        differences = []
        for _ in range(self.resamples):
            drawn = {}
            for name in (rich, poor):
                rows = np.flatnonzero(labels[name] == 1)
                other = np.flatnonzero(labels[name] == 0)
                taken = np.concatenate(
                    [
                        generator.choice(rows, size=positives, replace=False),
                        generator.choice(other, size=negatives, replace=False),
                    ]
                )
                drawn[name] = self.scores.area_under_curve(values[name][taken], labels[name][taken])
            differences.append(drawn[rich] - drawn[poor])
        return MatchedGap(
            rich=rich,
            poor=poor,
            poor_auc=self.scores.area_under_curve(values[poor], labels[poor]),
            rich_auc=self.scores.area_under_curve(values[rich], labels[rich]),
            difference=(
                float(np.percentile(differences, LOW_PERCENTILE)),
                float(np.median(differences)),
                float(np.percentile(differences, HIGH_PERCENTILE)),
            ),
            positives=positives,
            negatives=negatives,
        )

    def report(
        self,
        predictions: Float[np.ndarray, "studies targets"],
        truth: Float[np.ndarray, "studies targets"],
        pairs: tuple[tuple[str, str], ...] = PAIRS,
    ) -> list[str]:
        """Every pair's row, under a header naming the corpus the shares are taken over."""
        lines = [f"{predictions.shape[0]} studies, {self.resamples} matched draws per pair"]
        lines += [self.matched(predictions, truth, rich, poor).line() for rich, poor in pairs]
        return lines


def main() -> None:
    """Re-price the medial-over-lateral gap with the two findings' sample sizes made equal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="the report-derived table to score against")
    parser.add_argument("--paths", type=Path, default=Path("paths.yaml"))
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from core.paths import DataRoot  # noqa: PLC0415 - CLI-only, and this module is imported as a library

    root = DataRoot.from_config(arguments.paths)
    names, per_seed = PublishedRuler(root.processed() / "published").per_seed()
    predictions = np.asarray(RankAverage().combine(per_seed), dtype=np.float32)
    truth = SupervisionGap().reported(arguments.labels, names)
    for line in PrevalenceMatch(resamples=arguments.resamples).report(predictions, truth):
        logger.info(line)


if __name__ == "__main__":
    main()
