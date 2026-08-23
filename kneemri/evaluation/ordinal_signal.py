"""Whether the graded YES levels in the report table carry ORDINAL information, or only certainty.

knee-mxj argues that finer severity compounds with the ranking term, because the competition metric is
a ranking metric and severity is a ranking signal. That argument rests on a premise nobody has measured:
that 0.94 means "worse" rather than "the reader was surer". The two are indistinguishable in the table
and produce identical training curves — but they demand opposite work. If the levels already track
severity, the gradation is a live signal the loss consumes today; if they are pure LLM certainty, a
reader upgrade has to SUPPLY severity before any of that mechanism can fire.

The test needs no GPU and no labels file: a `HoldoutRecord` carries a member's per-study probabilities
AND the truth it was scored against, so the question is arithmetic over two arrays already on disk —
among the studies a model never trained on, is a 0.94 study ranked above a 0.68 study more often than
chance? Chance is 0.5, and a model that only learned "present vs absent" scores exactly that.

ANNOTATED ROWS ARE EXCLUDED, which is the whole reason the band is open at both ends. `Targets.load`
overwrites derived values with hard 0.0/1.0 where a study has a human annotation, so a 1.0 is a
radiologist saying "present", not a reader saying "severe" — pooling it with the graded levels would
measure the binary signal this test exists to hold out.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core.findings import TARGETS
from kneemri.evaluation.scores import LabelScores
from kneemri.runs.holdout_record import HoldoutReading, HoldoutRecord

logger = logging.getLogger(__name__)

GRADED = (0.5, 1.0)
"""The open band a graded YES level falls in. Exclusive at both ends: 0.5 is the unmentioned default
`Targets.load` fills with, and 0.0/1.0 are annotations that overwrote the derived value."""

CHANCE = 0.5
"""What a model that learned only present-vs-absent scores on this test, by construction."""

COMPARABLE = 2
"""Levels a label needs before "higher outranks lower" is a question at all."""


@dataclass(frozen=True)
class OrdinalReading:
    """One label's answer: how often the model ranks a higher-graded study above a lower-graded one."""

    label: str
    concordance: float
    pairs: int
    levels: list[float]

    def line(self) -> str:
        """One row of the report, carrying its own denominator.

        The count travels with the number because it is not the holdout's size: it is the number of
        cross-level PAIRS among graded studies only, which for a rare finding is a small fraction of it.
        A concordance quoted without it invites reading a 30-pair label as evidence.
        """
        levels = ", ".join(f"{level:.2f}" for level in self.levels)
        return f"{self.label:<28} {self.concordance:>6.3f}  over {self.pairs:>7,d} pairs  [{levels}]"


@dataclass(frozen=True)
class OrdinalSummary:
    """Both means over a member's labels, because the two answer different questions.

    The tool printed twelve rows and no summary, so every consumer averaged them by hand — and the
    numbers already quoted in `interpretations/` as "mean ordinal concordance" came from that arithmetic
    with no record of which mean it was. The two differ a lot here: Effusion carries 79,925 pairs and
    Fracture 1,122, so weighting by evidence is nearly a different statistic from weighting each label
    alike. Reporting one without naming it is how a rate over a subset gets read as a rate over the
    corpus, which this project has paid for three times in one session.
    """

    label_mean: float
    pair_weighted: float
    labels: int
    pairs: int

    def lines(self) -> list[str]:
        """Both means, each carrying the denominator it was taken over — never one number called THE mean."""
        return [
            f"{'MEAN over labels':<28} {self.label_mean:>6.3f}  over {self.labels:>7,d} labels",
            f"{'MEAN over pairs':<28} {self.pair_weighted:>6.3f}  over {self.pairs:>7,d} pairs",
        ]


@dataclass(frozen=True)
class OrdinalScan:
    """Many members ranked against each other by the one ruler that has beaten the binary holdout.

    IT EXISTS BECAUSE THE SWEEP WAS BEING DONE BY HAND. The 23-member and 48-member scans quoted on
    knee-7d1 were produced by looping this module's single-checkpoint CLI in a shell, which is the
    REPL-session shape the project forbids for exactly the reason it bit here: the numbers reached the
    beads and the interpretations with no committed command that reproduces them.

    RANKED BY THE PAIR-WEIGHTED MEAN, with the label mean carried beside it rather than folded in.
    `OrdinalSummary` refuses to name one of the two THE mean, and a ranking has to choose; pairs is the
    right choice for a displacement question because a seat is decided by evidence, not by giving
    Fracture's 1,122 pairs the same vote as Effusion's 79,925. The label mean travels in the row so a
    ranking that only holds under one of the two weightings is visible as such.
    """

    scanned: list[tuple[str, OrdinalSummary]]

    @classmethod
    def of(cls, checkpoints: list[Path]) -> "OrdinalScan":
        """Every member that HAS a record, read from disk and pooled.

        A checkpoint with no holdout record is skipped rather than seated at chance, for the reason
        `pooled` returns None: absence of evidence printed as 0.5 reads as a refutation. The count of
        answering members is reported beside the table so the skips are visible rather than silent.
        """
        scanned = []
        for checkpoint in checkpoints:
            reading = HoldoutRecord(checkpoint).read()
            if reading is None:
                continue
            summary = OrdinalSignal.pooled(OrdinalSignal(reading).per_label())
            if summary is not None:
                scanned.append((checkpoint.name.split(".")[0], summary))
        return cls(scanned)

    def ranked(self) -> list[tuple[str, OrdinalSummary]]:
        """Best first, by pair-weighted concordance — the order a seat displacement is read from."""
        return sorted(self.scanned, key=lambda scanned: -scanned[1].pair_weighted)

    def lines(self, seated: tuple[str, ...] = ()) -> list[str]:
        """The ranking, with seated members marked so a displacement is read rather than computed.

        The mark is the whole point of passing the roster in: a candidate outranking a seated member is
        the licensed move, and a candidate outranking only unseated ones is not a submission.
        """
        rows = [f"{'member':<34} {'pairs':>6} {'labels':>7}  {'seat':>4}"]
        for name, summary in self.ranked():
            seat = "SEAT" if name in seated else ""
            rows.append(f"{name:<34} {summary.pair_weighted:>6.3f} {summary.label_mean:>7.3f}  {seat:>4}")
        return rows


class OrdinalSignal:
    """A member's holdout reading, asked whether its ranking respects the graded levels."""

    def __init__(self, reading: HoldoutReading) -> None:
        self.reading = reading

    def per_label(self) -> list[OrdinalReading]:
        """Each label's cross-level concordance, over the graded studies it has.

        Labels with fewer than two graded levels present are omitted rather than reported as chance —
        the question is undefined there, and a 0.5 in the table would read as a refutation.
        """
        readings = []
        for column, label in enumerate(TARGETS):
            reading = self.concordance(self.reading.predictions[:, column], self.reading.truth[:, column])
            if reading is not None:
                readings.append(OrdinalReading(label, *reading))
        return readings

    @classmethod
    def detail(cls, checkpoint: Path) -> list[str]:
        """One member's twelve rows and both means, as lines — the single-checkpoint report."""
        reading = HoldoutRecord(checkpoint).read()
        if reading is None:
            return [f"{checkpoint.name} has no holdout record — train it or back-fill it first"]
        header = f"{checkpoint.name}, epoch {reading.epoch}, {reading.source} over {len(reading.studies)} studies"
        readings = cls(reading).per_label()
        summary = cls.pooled(readings)
        return [header, *(label.line() for label in readings), *(summary.lines() if summary else [])]

    @staticmethod
    def pooled(readings: list[OrdinalReading]) -> OrdinalSummary | None:
        """Both means over the labels that could answer, taking the readings rather than recomputing them.

        None when no label had two graded levels, which is the same silence `concordance` keeps: a mean
        over an empty set is not chance, it is an absence of evidence, and printing 0.5 would read as a
        refutation of the very thing this module exists to test.
        """
        if not readings:
            return None
        pairs = sum(reading.pairs for reading in readings)
        weighted = sum(reading.concordance * reading.pairs for reading in readings) / pairs
        label_mean = sum(reading.concordance for reading in readings) / len(readings)
        return OrdinalSummary(label_mean, weighted, len(readings), pairs)

    @staticmethod
    def concordance(
        predictions: Float[NDArray[np.float64], " studies"], truth: Float[NDArray[np.float32], " studies"]
    ) -> tuple[float, int, list[float]] | None:
        """How often a higher-graded study outranks a lower-graded one, pooled over every level pair.

        Pooled by CONCORDANT PAIRS rather than by averaging each level pair's rate: the pairs are the
        unit of evidence, and 0.94-vs-0.68 (rare, and the pair that carries the argument) would
        otherwise weigh as much as 0.82-vs-0.68 (common). Each pair's rate comes from
        `LabelScores.area_under_curve`, which is the same Mann-Whitney statistic with midranks — a tie
        between two predictions must be half-concordant, not settled by study order.

        None when fewer than two levels are present, because the question does not exist there.
        """
        graded = (truth > GRADED[0]) & (truth < GRADED[1])
        levels = sorted(float(level) for level in np.unique(truth[graded]))
        if len(levels) < COMPARABLE:
            return None
        concordant, pairs = 0.0, 0
        for position, high in enumerate(levels):
            for low in levels[:position]:
                chosen = np.isin(truth, [high, low])
                above = (truth[chosen] == high).astype(np.int64)
                counted = int(above.sum()) * int((1 - above).sum())
                concordant += LabelScores.area_under_curve(predictions[chosen], above) * counted
                pairs += counted
        return concordant / pairs, pairs, levels


def main() -> None:
    """Report ordinal concordance: per label for one member, or ranked across many."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", help="a member, its metrics file or its record")
    parser.add_argument("--directory", type=Path, help="scan every holdout record under it")
    parser.add_argument("--seated", nargs="*", default=(), help="member names to mark, so a displacement is read")
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    checkpoints = list(arguments.checkpoint or ())
    if arguments.directory is not None:
        checkpoints += sorted(arguments.directory.glob("*.holdout.npz"))
    if not checkpoints:
        parser.error("pass --checkpoint, --directory, or both")
    logger.info("chance is %.2f; above it means the graded levels carry severity, not only certainty", CHANCE)
    if len(checkpoints) == 1:
        for line in OrdinalSignal.detail(checkpoints[0]):
            logger.info("%s", line)
        return
    scan = OrdinalScan.of(checkpoints)
    logger.info("%d of %d members answered", len(scan.scanned), len(checkpoints))
    for line in scan.lines(tuple(arguments.seated)):
        logger.info("%s", line)


if __name__ == "__main__":
    main()
