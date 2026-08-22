"""Can the model tell the two compartments apart at all? `python -m kneemri.evaluation.side_blindness`.

THE ONE THING knee-yrg NEVER MEASURED. That bead established the lateral deficit on the full corpus —
Medial Meniscus beats Lateral Meniscus by +0.0622, Medial OA beats Lateral OA by +0.0472 — and then
refuted four explanations for it in a row: the laterality flip (the gap survives where the DICOM tag and
the geometric rule agree), the slice band (a symmetric cut cannot make an asymmetric deficit), prevalence
(matched counts move the gap by 0.0001), and the in-plane crop (it discards 0.0060 lateral against 0.0173
MEDIAL, so it points the wrong way). Every one of those four asks the same KIND of question — is our
framing of the DATA cutting one compartment harder than the other — and the answer four times was no.

So the bead's own conclusion is that the next attempt has to be a MODEL change, and this is the reading
that says WHICH model change. It stops asking what the data does to the compartments and asks what the
model's OUTPUT does with them.

THE HYPOTHESIS IT TESTS, stated so it can fail. A model whose head sees a pooled, position-free summary
of a study can learn "this knee has a meniscus tear" without ever learning WHICH compartment holds it. On
this corpus medial positives outnumber lateral roughly two to one, so a presence detector that guesses
the majority compartment scores well on the medial label and badly on the lateral one — which is the
observed signature exactly, and it is produced by the POOLING rather than by anything in the pixels. The
four refuted mechanisms cannot distinguish that from "the lateral compartment is genuinely harder to
see", because both predict the same per-label profile.

THE DISCORDANT CONTRAST separates them, and it is the whole content of this module. Take the studies
positive on EXACTLY ONE side of a compartment pair, and score EACH of the pair's two columns at ranking
its own studies above the other side's. Both groups have the finding; they differ only in which
compartment. A side-blind detector emits the same number for both and scores 0.50 by construction — not
approximately, but as an identity, because its output does not depend on the thing being varied. A model
that resolves the compartment scores above 0.50 no matter how hard the finding is, because difficulty
moves both groups together and this contrast divides it out.

BOTH COLUMNS, because one column cannot tell the two failures apart. A pair whose representation has no
side in it fails twice; a pair whose side IS represented but was only learnt on the majority class fails
once, on the minority column, while its partner reads well above chance. Those call for different work —
the first wants position back into the representation, the second wants the minority column supervised
against its partner — and a single-column reading collapses them into one number.

WHY THE CHANCE LEVEL IS TRUSTWORTHY HERE in a way per-label AUC is not. Every other reading of this axis
compares two labels with different prevalences, different annotators' thresholds, and different intrinsic
difficulty, so 0.87 against 0.82 needs an argument before it means anything. This contrast holds the
finding fixed and varies only the side, and its null is 0.50 for a reason that does not depend on any of
those — so a result near 0.50 is a positive claim about the model, not an absence of evidence.

WHAT IT CANNOT SAY. It reads the pooled ENSEMBLE the same way it reads a member, so a 0.50 here indicts
the family, not one architecture; the members share a pooling stage and would have to be read apart to
blame it. And the discordant groups are small — the whole point is that they exclude the studies positive
on both sides — so the interval matters more than the point estimate and is reported beside it.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core.bootstrap import RESAMPLE_SEED, RESAMPLES
from core.findings import TARGETS
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import MIDPOINT, LabelScores

logger = logging.getLogger(__name__)

COMPARTMENT_PAIRS: tuple[tuple[str, str], ...] = (
    ("Medial Meniscus", "Lateral Meniscus"),
    ("Medial OA", "Lateral OA"),
)
"""The two findings this corpus annotates ONCE PER COMPARTMENT, which is what makes the contrast exist.

Named as pairs rather than derived from the label strings. A prefix match on "Medial"/"Lateral" would
also catch MCL — the medial COLLATERAL ligament, which has no lateral counterpart annotated here — and
pair it with nothing or, worse, with whichever lateral label sorted next to it.
"""

CHANCE = 0.5
"""What a detector that cannot resolve the compartment scores on the discordant contrast, exactly."""


@dataclass(frozen=True)
class Discordant:
    """One compartment pair's discordant studies: those positive on exactly one of the two sides."""

    pair: tuple[str, str]
    lateral_only: NDArray[np.bool_]
    medial_only: NDArray[np.bool_]

    @staticmethod
    def of(
        pair: tuple[str, str],
        truth: Float[NDArray[np.float32], "studies targets"],
        columns: dict[str, int],
    ) -> "Discordant":
        """The two exclusive groups, binarised at the same midpoint every other reading here uses."""
        medial = truth[:, columns[pair[0]]] > MIDPOINT
        lateral = truth[:, columns[pair[1]]] > MIDPOINT
        return Discordant(pair, lateral_only=lateral & ~medial, medial_only=medial & ~lateral)

    def contrast(
        self,
        predictions: Float[NDArray[np.float64], "studies targets"],
        columns: dict[str, int],
        scores: LabelScores,
        *,
        lateral: bool,
    ) -> float:
        """How well ONE of the pair's columns separates the studies that belong to it from the others.

        BOTH DIRECTIONS ARE READ, because only the pair of numbers distinguishes the two failures this
        module exists to tell apart. A model emitting ONE signal for the finding and copying it into
        both columns scores 0.50 twice — the column cannot depend on a side it never represented. A
        model that resolves the majority compartment and misses the minority one scores well on the
        medial column and 0.50 on the lateral. Reading only the lateral column cannot separate those,
        and they call for different work: the first is a representation that has no side in it, the
        second is a side that is represented but not learnt on the smaller class.
        """
        selected = self.lateral_only | self.medial_only
        wanted = self.lateral_only if lateral else self.medial_only
        column = predictions[selected, columns[self.pair[1] if lateral else self.pair[0]]].astype(np.float64)
        return scores.area_under_curve(column, wanted[selected].astype(np.int64))


class SideBlindness:
    """The discordant contrast for every compartment pair, with the interval that decides it."""

    def __init__(self, scores: LabelScores, resamples: int = RESAMPLES) -> None:
        self.scores = scores
        self.resamples = resamples

    def lines(
        self,
        pool_names: list[str],
        predictions: Float[NDArray[np.float64], "studies targets"],
        truth: Float[NDArray[np.float32], "studies targets"],
        columns: dict[str, int],
    ) -> list[str]:
        """Both of each pair's columns on its own discordant contrast, with the interval that decides it."""
        report = [f"discordant contrast over {len(truth)} studies, {len(pool_names)} members combined"]
        for pair in COMPARTMENT_PAIRS:
            groups = Discordant.of(pair, truth, columns)
            lateral, medial = int(groups.lateral_only.sum()), int(groups.medial_only.sum())
            if lateral == 0 or medial == 0:
                report.append(f"  {pair[1]:18s} UNREADABLE - {lateral} lateral-only, {medial} medial-only")
                continue
            report.append(f"  {pair[0]} / {pair[1]}: n={lateral} lateral-only vs {medial} medial-only")
            for side in (True, False):
                observed = groups.contrast(predictions, columns, self.scores, lateral=side)
                low, high = self._interval(groups, predictions, columns, lateral=side)
                verdict = "resolves the side" if low > CHANCE else "INDISTINGUISHABLE FROM SIDE-BLIND"
                column = pair[1] if side else pair[0]
                report.append(f"    {column:18s} column {observed:.4f}  [{low:.4f}, {high:.4f}]   {verdict}")
        return report

    def _interval(
        self,
        groups: Discordant,
        predictions: Float[NDArray[np.float64], "studies targets"],
        columns: dict[str, int],
        *,
        lateral: bool,
    ) -> tuple[float, float]:
        """A 95% bootstrap interval, resampling the two groups SEPARATELY.

        Separately because the group sizes are the design, not the data: resampling the pooled selection
        would let a draw contain forty lateral-only studies and six medial-only ones, which is a
        different contrast with a different variance rather than a replicate of this one.
        """
        generator = np.random.default_rng(RESAMPLE_SEED)
        wanted = np.flatnonzero(groups.lateral_only if lateral else groups.medial_only)
        other = np.flatnonzero(groups.medial_only if lateral else groups.lateral_only)
        column = columns[groups.pair[1] if lateral else groups.pair[0]]
        labels = np.concatenate([np.ones(len(wanted), dtype=np.int64), np.zeros(len(other), dtype=np.int64)])
        draws = []
        for _ in range(self.resamples):
            rows = np.concatenate([generator.choice(wanted, len(wanted)), generator.choice(other, len(other))])
            draws.append(self.scores.area_under_curve(predictions[rows, column].astype(np.float64), labels))
        finite = np.array([value for value in draws if np.isfinite(value)])
        return float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def main() -> None:
    """Read the compartment contrast off holdout records, with no card and no inference pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, nargs="+", required=True, help="runs sharing ONE split")
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pool = RecordMix(arguments.records).per_member()
    columns = {target: index for index, target in enumerate(TARGETS)}
    combined = pool.combined(list(range(len(pool))))
    for line in SideBlindness(LabelScores()).lines(pool.names, combined, pool.truth, columns):
        logger.info(line)


if __name__ == "__main__":
    main()
