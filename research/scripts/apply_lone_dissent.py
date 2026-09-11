"""Write the v2-lone-dissent correction into report_labels_v2, soft (D335/D336).

The selector is `labels/v2_lone_dissent.csv`: the 2,315 cells where the three foreign
report-label tables vote unanimously and our v2 differs. Two independent references —
main's blind adjudication of 12 such cells and the organisers' radiologist gold on 32 —
both put v2's error rate on that set at 75%.

Correction, not masking: FCF (Jiang, AAAI-24) Thm 5 has correction beating filtering at
equal cleaned-noise rate, and `confident_span` measured masking 27.3% of cells at -0.0116.
The written value is the cluster's SOFT mean rather than its hard vote, because v2's targets
are confidence-graded ({0.08, 0.28, 0.68, 0.82, 0.94}) and writing hard 0/1 into exactly the
cells the arm varies would smuggle a target-distribution change into a one-variable arm.
`__conf` and `__verdict` are untouched: a corrected cell stays supervised.

**D363 SPLIT THE OPERATION BY DIRECTION AND ONLY ONE HALF SURVIVES.** The balance56 blind pack
adjudicated 28 cells the correction DELETES (v2=1 -> cluster 0) and 28 it ADDS (v2=0 -> cluster 1).
Deletions are wrong 2/28 = 0.071 [0.020, 0.226]; additions are wrong 22/28 = **0.786** [0.605, 0.898],
and 21 of those 22 are OA columns where the cluster asserts a compartment the report never NAMES. So
`--direction down` writes the deletions alone, leaving v2 untouched where the cluster wants to add: at
corpus scale that keeps ~2,080 cells at ~93% right and drops ~235 additions that were ~21% right.

    python research/scripts/apply_lone_dissent.py [--direction both|down]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

LABELS = Path("D:/data/volumetric/mri/processed/rsna_knee_abnormality/labels")
SOURCE = LABELS / "report_labels_v2.csv"
SELECTOR = LABELS / "v2_lone_dissent.csv"
TARGETS = {
    "both": LABELS / "report_labels_v2_corrected.csv",
    "down": LABELS / "report_labels_v2_delonly.csv",
}
UID = "StudyInstanceUID"
TOLERANCE = 1e-6

log = logging.getLogger(__name__)


def written(direction: str, v2_soft: float, cluster_soft: float) -> bool:
    """Whether a selector row is written under `direction` ('both', or 'down' for deletions only)."""
    return direction == "both" or cluster_soft < v2_soft


def apply_correction(source: pd.DataFrame, selector: pd.DataFrame, direction: str) -> pd.DataFrame:
    """Return `source` with each selected cell in `direction` overwritten by the cluster's soft value."""
    corrected = source.set_index(UID).copy()
    mismatched = 0
    for row in selector.itertuples(index=False):
        before = corrected.at[row.study_uid, row.column]
        if abs(float(before) - float(row.v2_soft)) > TOLERANCE:
            mismatched += 1
        if written(direction, float(row.v2_soft), float(row.cluster_soft)):
            corrected.at[row.study_uid, row.column] = row.cluster_soft
    if mismatched:
        message = (
            f"{mismatched} of {len(selector)} selected cells did not hold the v2_soft value "
            "the selector recorded - the selector and the table disagree about the corpus"
        )
        raise ValueError(message)
    return corrected.reset_index()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", choices=sorted(TARGETS), default="both")
    args = parser.parse_args()
    target = TARGETS[args.direction]

    source = pd.read_csv(SOURCE)
    selector = pd.read_csv(SELECTOR)
    corrected = apply_correction(source, selector, args.direction)

    eligible = sum(
        written(args.direction, float(row.v2_soft), float(row.cluster_soft))
        for row in selector.itertuples(index=False)
    )
    changed = int((corrected.set_index(UID) != source.set_index(UID)).sum().sum())
    log.info("direction %s", args.direction)
    log.info("selector rows %d, written %d", len(selector), eligible)
    log.info("cells changed %d", changed)
    log.info("shape %s -> %s", source.shape, corrected.shape)
    corrected.to_csv(target, index=False)
    log.info("wrote %s", target)


if __name__ == "__main__":
    main()
