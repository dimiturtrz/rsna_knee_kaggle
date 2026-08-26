"""Is one calibration constant right for all twelve findings? `python -m core.data.labels.level_agreement`.

`Supervision.applied` replaces each reader level with the rate it was measured to agree at, and those
rates (`MEASURED_AGREEMENT`) are ONE NUMBER PER LEVEL, pooled across every finding. `calib_span` spent an
arm on that remap and came back −0.0020 with the per-label pattern INVERTED against its own prediction:
Effusion, named as the label with most to gain, was the worst row on the board at −0.0212.

This module measures the same object the constants estimate, WITHOUT pooling: for each finding and each
level, how often the radiologist agreed. It needs no report text, so unlike `severity_boundary` it reads
the non-English 59.9% too, and its Effusion row rests on 27 cells rather than that module's 12.

AND THE SPREAD ITSELF NEEDED A NULL BEFORE IT COULD BE READ — 2026-08-26. `spread` prints a range and
`inverted` names the rows crossing 0.5, and the ledger spent both as the mechanism behind `calib_span`'s
inversion. Neither is evidence on its own: ten findings holding 5-27 annotated cells each, every one of
them genuinely drawn from the pooled constant, produce a wide range and cross 0.5 by sampling alone.
`heterogeneity` draws that null. Measured:

    level   findings   chi2 / df    p       range vs chance
    0.08       11      8.1 / 10    0.615    0.105 vs 0.211   consistent with pooling
    0.28       12     26.1 / 11    0.021    0.341 vs 0.429   beyond pooling
    0.68       12     18.0 / 11    0.100    0.778 vs 0.833   consistent with pooling
    0.82       12     22.9 / 11    0.024    0.800 vs 0.750   beyond pooling
    0.94       11     20.3 / 10    0.135    0.400 vs 0.286   consistent with pooling

THE LEVEL THE LEDGER LEANED ON IS THE ONE THAT DOES NOT SURVIVE. 0.68 carries 6,461 mild cells, holds
Effusion's 0.481 against 0.378 and Synovitis's 0.778, and its "0.000 to 0.778" range is the sentence the
calib_span row quotes — and that range is SMALLER than the 0.833 those same cells produce under one
constant. Heterogeneity survives at 0.28 and at 0.82, and 0.82 is exactly where four of the five
inversions sit and where `stability` reads 2.0 sigma, so the pooled constant there is unstable in both
directions at once.

HOW FAR THAT MAY BE PUSHED: five levels were tested, so two at p ~ 0.02 do not survive a Bonferroni
alpha of 0.01 individually. What does survive is the joint statement — under the null the expected count
of p < 0.05 across five tests is 0.25 and seeing two has probability ~0.023 — so per-finding structure
exists somewhere, while WHICH level carries it rests on a single uncorrected test each.

A LEVEL THAT CROSSES 0.5 IS NOT THE SAME MISTAKE AS ONE THAT DOES NOT. The remap writes a target VALUE,
and the metric ranks, so a constant that lands on the wrong SIDE of the finding's own rate inverts that
finding's low-grade positives, while one that merely overshoots on the correct side only softens them.
`inverted` names the findings where the pooled constant does the first thing.

IT ALSO ASKS WHETHER THE CONSTANTS ARE A POPULATION ESTIMATE OR A FIT TO THESE 58 — 2026-08-26. That
question is not decoration, because it decides how one number in the ledger may be read.
`MEASURED_AGREEMENT` is fitted on the 696 annotation cells and is a hardcoded module constant, so
`--hold-annotations` drops those cells from the training TABLE and leaves the CONSTANTS carrying them.
`kneemri.evaluation.annotated_readings.withheld` refuses a checkpoint on the table route alone — it reads
`hold_annotations` and nothing else — so a `--calibrated-targets --hold-annotations` run passes the guard
while its targets still encode annotation-derived rates, and is then scored against those annotations.

That matters to `knee-bs4l`, which revived the calibration family on +0.0129 annotated against +0.0072
proxy. Two readings produce that gap. Either the annotated number is inflated by its own supervision, or
the calibration is doing what it was built to do — moving targets out of report-space toward
annotation-space — in which case the larger annotated gain is DEFINITIONAL and the proxy is the number
understating the mechanism. The second reading is the interesting one, because the board's ruler is
annotation-like (`holdannot_s2026` reads 0.8487 on report labels and 0.8932 on the annotation cells).

`stability` separates them without spending a GPU minute. Five scalars fitted on 72-196 cells each cannot
memorise which of 58 studies has an effusion; what they CAN carry is their own sampling error, correlated
with the same cells they are later scored against. So refit the constants on each half of the annotated
studies and see how far they move. Halves that agree within the shipped intervals say the constants are a
low-variance estimate of a population quantity, the second-order term is small, and the +0.0129 is
ruler-matching rather than leakage. Halves that disagree say the opposite and the number is unquotable
until a split-half REFIT is trained. Either way the answer costs minutes, not an arm.

WHAT IT DOES NOT SETTLE. A stable pair bounds the sampling-error route only. It does not make the
annotated read INDEPENDENT — the constants are still fitted on the ruler they are scored against, so the
number stays a consistency check and never becomes a confirmation.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from core import cli
from core.data.labels.targets import ASSERTS_PRESENCE, MEASURED_AGREEMENT
from core.findings import STUDY, TARGETS

logger = logging.getLogger(__name__)

LABELS = "report_labels_v2.csv"
TOLERANCE = 1e-6
"""Levels are written as decimals by the reader and read back as floats; this compares them as levels."""

THIN = 5
"""Below this many annotated cells a rate is not reported as one. Two cells agreeing is not a rate, and a
pooled constant is not refuted by a number whose own interval spans it."""

HALVES = 2
"""How many disjoint refits `stability` compares. Two, because the question is whether the constants are a
population estimate, and a population estimate is one that survives being computed on someone else."""

RESAMPLES = 2000
"""How many corpora are drawn under the pooled constant to price the observed per-finding spread."""

SEED = 2026
"""The draw is seeded so two readings of the same tables cannot disagree about whether pooling holds."""

DISPERSED = 0.05
"""Below this tail fraction the per-finding spread is more than thin cells alone would produce."""

SPREAD_QUANTILE = 0.95
"""Where the simulated range is quoted, so the printed spread is read against a like-for-like number."""

SETTLED = 2.0
"""How many of its OWN standard errors two half-refits may sit apart before the constant reads as a fit.

A flat distance was the first thing written here and it was a broken ruler — 0.05, which is BELOW the
noise it was meant to judge. Halving a sample multiplies its standard error by sqrt(2), and the gap
between two independent halves carries both, so a gap of about twice the full-sample SE is what a
perfectly stable constant PRODUCES. Measured against a flat 0.05, three of five levels read as moving
when the largest of them was 0.9 sigma. The tolerance has to be the level's own binomial spread or it
answers a question about n rather than about the constant.

Two sigma, because this is a two-sided consistency check on five levels at once and the decision it feeds
is whether one number in the ledger may be quoted, not whether an arm runs."""


class LevelAgreement:
    """Per finding and per reader level, how often the radiologist agreed with the reader's assertion."""

    def __init__(self, measured: tuple[tuple[float, float], ...] = MEASURED_AGREEMENT) -> None:
        self.measured = measured
        self.levels = tuple(level for level, _ in measured)
        self.pooled = dict(measured)

    def rates(self, reader: pd.DataFrame, annotated: pd.DataFrame) -> pd.DataFrame:
        """One row per (finding, level): the annotated positive rate there, and how many cells it rests on.

        Joined on the study identifier rather than positionally, because the reader table covers the whole
        corpus and the annotations cover 58 of it — a positional read would silently score the wrong rows.
        """
        shared = reader.index.intersection(annotated.index)
        rows = []
        for finding in TARGETS:
            if finding not in reader.columns:
                continue
            value = reader.loc[shared, finding]
            gold = annotated.loc[shared, finding]
            for level in self.levels:
                cells = (value - level).abs() < TOLERANCE
                cells &= gold.notna()
                rate = float(gold[cells].mean()) if cells.any() else float("nan")
                rows.append((finding, level, rate, int(cells.sum())))
        return pd.DataFrame(rows, columns=["finding", "level", "rate", "studies"])

    def refit(self, reader: pd.DataFrame, annotated: pd.DataFrame, studies: pd.Index) -> dict[float, tuple[float, int]]:
        """The pooled constant each level would take from `studies` alone, beside the cells it rests on.

        The same object `MEASURED_AGREEMENT` holds — every finding's cells at a level thrown together —
        just computed over a subset, which is what makes two of them comparable. The count comes back with
        it because a rate without its n cannot be compared to another rate.
        """
        shared = reader.index.intersection(annotated.index).intersection(studies)
        columns = [finding for finding in TARGETS if finding in reader.columns]
        constants = {}
        for level in self.levels:
            agreed, total = 0.0, 0
            for finding in columns:
                value = reader.loc[shared, finding]
                gold = annotated.loc[shared, finding]
                cells = ((value - level).abs() < TOLERANCE) & gold.notna()
                agreed += float(gold[cells].sum())
                total += int(cells.sum())
            constants[level] = (agreed / total if total else float("nan"), total)
        return constants

    def halves(self, reader: pd.DataFrame, annotated: pd.DataFrame) -> tuple[pd.Index, ...]:
        """The annotated studies split into disjoint groups, deterministically and without an RNG.

        Interleaved over the SORTED identifiers rather than cut in the middle: a cut inherits whatever
        order the identifiers encode, and study identifiers in this corpus are not shuffled.
        """
        shared = reader.index.intersection(annotated.index).sort_values()
        return tuple(shared[offset::HALVES] for offset in range(HALVES))

    def stability(self, reader: pd.DataFrame, annotated: pd.DataFrame) -> list[str]:
        """Each level's constant refitted on disjoint halves, and how far the two land apart.

        A pair inside `SETTLED` says the shipped constant is a population estimate, so a calibrated arm's
        score against the annotations is not meaningfully reading its own supervision back through them.
        """
        groups = self.halves(reader, annotated)
        refits = [self.refit(reader, annotated, studies) for studies in groups]
        lines = []
        for level in self.levels:
            values = [refit[level][0] for refit in refits]
            counts = [refit[level][1] for refit in refits]
            gap = max(values) - min(values)
            sigma = self.gap_sigma(self.pooled[level], counts)
            ratio = gap / sigma if sigma > 0 else float("nan")
            verdict = "settled" if ratio <= SETTLED else "MOVES"
            beside = " ".join(f"{value:.3f}" for value in values)
            lines.append(
                f"  level {level:.2f}  shipped {self.pooled[level]:.3f}   halves {beside}   "
                f"gap {gap:.3f}  expected {sigma:.3f}  {ratio:.1f} sigma  {verdict}"
            )
        return lines

    @staticmethod
    def gap_sigma(rate: float, counts: list[int]) -> float:
        """How far apart two independent estimates of `rate` land by chance alone, given their cell counts.

        The binomial standard error of a difference, under the null that both halves estimate the same
        constant — so the shipped rate supplies p for both terms rather than each half supplying its own,
        which would let a half that happened to land extreme widen its own tolerance.
        """
        variance = sum(rate * (1.0 - rate) / count for count in counts if count)
        return float(np.sqrt(variance))

    def heterogeneity(self, rates: pd.DataFrame, generator: np.random.Generator) -> list[str]:
        """Whether a level's per-finding spread is MORE than thin cells drawn from one constant produce.

        `spread` and `inverted` print a range and name the rows that cross 0.5, and both were read as
        evidence that the findings disagree. Neither is: ten findings holding 5-27 annotated cells each,
        every one of them actually drawn from the pooled rate, produce a wide range and cross 0.5 by
        sampling alone. Without this comparison the per-finding claim is unfalsifiable, and the ledger
        already spends it as the mechanism behind `calib_span`'s inversion.

        The statistic is the chi-square dispersion of the per-finding positive counts about the pooled
        constant, and the null is drawn rather than tabulated — the cells are far too thin for the
        asymptotic distribution, which is exactly the regime the whole question lives in. The observed
        RANGE is reported beside the range those same draws produce, because the range is the number the
        ledger quoted and it deserves its own null rather than a p-value standing in for it.
        """
        lines = []
        for level in self.levels:
            rows = rates[(rates["level"] == level) & (rates["studies"] > 0)]
            counts = rows["studies"].to_numpy(dtype=np.int64)
            pooled = self.pooled[level]
            if counts.size < HALVES or pooled <= 0.0 or pooled >= 1.0:
                continue
            positives = np.rint(rows["rate"].to_numpy(dtype=np.float64) * counts)
            drawn = generator.binomial(counts, pooled, size=(RESAMPLES, counts.size)).astype(np.float64)
            observed = float(self._dispersion(positives[None, :], counts, pooled)[0])
            tail = float((self._dispersion(drawn, counts, pooled) >= observed).mean())
            thick = counts >= THIN
            reported = ""
            if int(thick.sum()) >= HALVES:
                widths = self._width(drawn[:, thick], counts[thick])
                span = float(self._width(positives[None, thick], counts[thick])[0])
                reported = f"range {span:.3f} vs {float(np.quantile(widths, SPREAD_QUANTILE)):.3f} by chance   "
            verdict = "BEYOND POOLING" if tail < DISPERSED else "consistent with pooling"
            lines.append(
                f"  level {level:.2f}  {counts.size} findings  chi2 {observed:.1f} on {counts.size - 1} df  "
                f"p {tail:.3f}   {reported}{verdict}"
            )
        return lines

    @staticmethod
    def _dispersion(positives: NDArray[np.float64], counts: NDArray[np.int64], rate: float) -> NDArray[np.float64]:
        """Chi-square dispersion of each row of per-finding positive counts about one shared rate."""
        residual = positives - counts * rate
        return np.asarray((residual**2 / (counts * rate * (1.0 - rate))).sum(axis=-1), dtype=np.float64)

    @staticmethod
    def _width(positives: NDArray[np.float64], counts: NDArray[np.int64]) -> NDArray[np.float64]:
        """The range the per-finding rates span in each row, which is the number `spread` prints."""
        observed = positives / counts
        return np.asarray(observed.max(axis=-1) - observed.min(axis=-1), dtype=np.float64)

    def spread(self, rates: pd.DataFrame) -> list[str]:
        """Each level's per-finding rates beside the one constant standing in for all of them."""
        lines = []
        for level in self.levels:
            rows = rates[(rates["level"] == level) & (rates["studies"] >= THIN)]
            if rows.empty:
                continue
            lines.append(
                f"  level {level:.2f}  pooled constant {self.pooled[level]:.3f}   "
                f"per-finding {rows['rate'].min():.3f} to {rows['rate'].max():.3f} "
                f"over {len(rows)} findings with {THIN}+ annotated cells"
            )
        return lines

    def inverted(self, rates: pd.DataFrame) -> list[str]:
        """Findings whose own rate sits on the OTHER side of 0.5 from the constant that replaces it.

        This is the row that changes a ranking rather than a magnitude, and it is the shape `calib_span`
        measured as its largest negative.
        """
        thick = rates[(rates["studies"] >= THIN) & rates["rate"].notna()]
        found = []
        for _, row in thick.iterrows():
            constant = self.pooled[row["level"]]
            if (row["rate"] > ASSERTS_PRESENCE) != (constant > ASSERTS_PRESENCE):
                found.append(
                    f"  {row['finding']:18s} level {row['level']:.2f}  own rate {row['rate']:.3f}  "
                    f"constant {constant:.3f}  ({int(row['studies'])} annotated cells)"
                )
        return found


def main() -> None:
    """Measure each finding's own agreement rate at each reader level, against the pooled constant."""
    command = cli.Command(__doc__, logger)
    command.parser.add_argument("--labels", type=Path, help="the reader table; defaults to report_labels_v2.csv")
    arguments = command.parsed()

    root = command.rooted(arguments)
    reader = pd.read_csv(arguments.labels or root.processed() / "labels" / LABELS).set_index(STUDY)
    annotated = pd.read_csv(root.raw() / "train.csv").set_index(STUDY).dropna(subset=TARGETS, how="all")

    agreement = LevelAgreement()
    rates = agreement.rates(reader, annotated)
    command.report([f"{len(reader.index.intersection(annotated.index))} annotated studies in the reader table"])
    command.report(["per-finding spread against the pooled constant:", *agreement.spread(rates)])
    # The rows that invert rather than soften — a ranking metric feels these and not the others.
    command.report(["findings the pooled constant puts on the WRONG SIDE of 0.5:", *agreement.inverted(rates)])
    command.report(
        [
            "is that spread more than thin cells drawn from ONE constant would produce:",
            *agreement.heterogeneity(rates, np.random.default_rng(SEED)),
        ]
    )
    command.report(
        [
            "constants refitted on disjoint halves — do they estimate a population or these studies:",
            *agreement.stability(reader, annotated),
        ]
    )


if __name__ == "__main__":
    main()
