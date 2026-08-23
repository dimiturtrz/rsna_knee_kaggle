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

IT CANNOT AUDIT A RUN TRAINED WITH A SIDE TERM, measured rather than supposed. `side_dense` (knee-6u12)
fitted `PairedSides` at weight 1.0, which places literally zero gradient on the CONTROL column, and the
control still went dirty in both pairs: 0.4770 / 0.4133 under the matched control against 0.4274 / 0.3498
under the arm. Pushing two columns of a shared trunk apart moves a third that reads the same features, so
the confound detector stops being a detector exactly when a side term is on. Two consequences: a
side-term weight sweep cannot be steered by this contrast, since the sweep bends its own ruler; and
`side_dense` established separately that a column can gain here while LOSING per-label AUC, so this
reading is a diagnostic about the representation and never a proxy for the score.

AND IT IS NOT THE METRIC. The competition ranks study against study WITHIN a label; this contrast ranks
one label against another WITHIN a study. They are orthogonal orderings over the same logits, and buying
the second cost the first outright on the one arm that tried.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray

from core.bootstrap import RESAMPLE_SEED, RESAMPLES
from core.findings import COMPARTMENT_PAIRS, TARGETS
from kneemri.evaluation.record_mix import RecordMix
from kneemri.evaluation.scores import MIDPOINT, LabelScores

logger = logging.getLogger(__name__)

CHANCE = 0.5
"""What a detector that cannot resolve the compartment scores on the discordant contrast, exactly."""

CONTROL = "PF OA"
"""The column read on every pair's contrast as a NEGATIVE control, because it belongs to neither side.

THE CONFOUND THIS EXISTS TO KILL. The contrast assumes its two groups differ only in which compartment
holds the finding. They need not: medial disease is both commoner and typically higher-grade here, so
the medial-only studies may simply be sicker knees. A column that tracks overall burden would then
separate the groups without knowing anything about sides, and the medial column's above-chance reading
would be that artefact rather than the side-awareness it looks like.

Patellofemoral OA is the third compartment. It is annotated on the same axis as the pair, it is present
in both groups, and it is neither of the two sides being varied — so it cannot legitimately resolve this
contrast, and whatever it scores is the confound's size measured directly. Near 0.50 licenses reading
the pair's own columns as side information. Well above it means the pair's numbers are burden, and the
honest conclusion is that NEITHER column resolves the side.
"""


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
        label: str,
        predictions: Float[NDArray[np.float64], "studies targets"],
        columns: dict[str, int],
        scores: LabelScores,
    ) -> float:
        """How well ANY column ranks this pair's lateral-only studies above its medial-only ones.

        ONE fixed direction for every column, so the three readings share a scale and 0.50 means the
        same thing in all of them. A caller that wants the medial column's own success reads it as
        `1 - x`, which is what `oriented` is for.

        ANY column and not just the pair's, because the pair's numbers are only interpretable beside a
        column that cannot legitimately move: see CONTROL. And both of the pair's own columns are read
        because one alone cannot tell apart a representation with no side in it (which fails on both)
        from a side that was only learnt on the majority class (which fails on the minority column
        while its partner reads well above chance) — and those call for different work.
        """
        selected = self.lateral_only | self.medial_only
        column = predictions[selected, columns[label]].astype(np.float64)
        return scores.area_under_curve(column, self.lateral_only[selected].astype(np.int64))

    def oriented(self, label: str, value: float) -> float:
        """`value` turned so that above-chance always means "this column found its own side".

        The medial column succeeds by ranking lateral-only studies LOW, so its raw contrast runs
        backwards; flipping it here rather than at each call site is what keeps one verdict rule
        readable across all three columns.
        """
        return 1.0 - value if label == self.pair[0] else value


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
            for label in (pair[1], pair[0], CONTROL):
                observed = groups.oriented(label, groups.contrast(label, predictions, columns, self.scores))
                low, high = sorted(
                    groups.oriented(label, bound) for bound in self._interval(groups, label, predictions, columns)
                )
                verdict = self._verdict(label, low, high)
                report.append(f"    {label:18s} column {observed:.4f}  [{low:.4f}, {high:.4f}]   {verdict}")
        return report

    @staticmethod
    def _verdict(label: str, low: float, high: float) -> str:
        """What the interval licenses saying — and for the control, the OPPOSITE sentence.

        The control's job is to fail. An interval that straddles chance is the result there, because it
        says the two groups do not differ on a column that cannot know which side they differ on; a
        control that separates them is the one outcome that invalidates the pair's numbers beside it.
        """
        if label == CONTROL:
            return "control clean" if low <= CHANCE <= high else "CONFOUNDED - the groups differ on a neutral column"
        return "resolves the side" if low > CHANCE else "INDISTINGUISHABLE FROM SIDE-BLIND"

    def _interval(
        self,
        groups: Discordant,
        label: str,
        predictions: Float[NDArray[np.float64], "studies targets"],
        columns: dict[str, int],
    ) -> tuple[float, float]:
        """A 95% bootstrap interval, resampling the two groups SEPARATELY.

        Separately because the group sizes are the design, not the data: resampling the pooled selection
        would let a draw contain forty lateral-only studies and six medial-only ones, which is a
        different contrast with a different variance rather than a replicate of this one.
        """
        generator = np.random.default_rng(RESAMPLE_SEED)
        wanted = np.flatnonzero(groups.lateral_only)
        other = np.flatnonzero(groups.medial_only)
        column = columns[label]
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
