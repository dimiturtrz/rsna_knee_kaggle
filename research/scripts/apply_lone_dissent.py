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

    python research/scripts/apply_lone_dissent.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LABELS = Path("D:/data/volumetric/mri/processed/rsna_knee_abnormality/labels")
SOURCE = LABELS / "report_labels_v2.csv"
SELECTOR = LABELS / "v2_lone_dissent.csv"
TARGET = LABELS / "report_labels_v2_corrected.csv"
UID = "StudyInstanceUID"
TOLERANCE = 1e-6

log = logging.getLogger(__name__)


def apply_correction(source: pd.DataFrame, selector: pd.DataFrame) -> pd.DataFrame:
    """Return `source` with each selected cell overwritten by the cluster's soft value."""
    corrected = source.set_index(UID).copy()
    mismatched = 0
    for row in selector.itertuples(index=False):
        before = corrected.at[row.study_uid, row.column]
        if abs(float(before) - float(row.v2_soft)) > TOLERANCE:
            mismatched += 1
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
    source = pd.read_csv(SOURCE)
    selector = pd.read_csv(SELECTOR)
    corrected = apply_correction(source, selector)

    changed = int((corrected.set_index(UID) != source.set_index(UID)).sum().sum())
    log.info("selector rows %d", len(selector))
    log.info("cells changed %d", changed)
    log.info("shape %s -> %s", source.shape, corrected.shape)
    corrected.to_csv(TARGET, index=False)
    log.info("wrote %s", TARGET)


if __name__ == "__main__":
    main()
