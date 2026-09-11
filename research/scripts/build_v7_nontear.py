"""Write the per-column non-tear correction on top of report_labels_v6tears (D352).

The operation is `column-substitution`, third use, and it is deliberately a SECOND table
rather than more cells in v6tears: one arm varies one thing, and folding the non-tear
correction into the tear table would leave neither priceable.

The rule is PER COLUMN, not a pooled margin threshold. D351 measured a soft-margin gradient
on 72 adjudicated OA-family cells and proposed substituting the 365 corpus cells clearing
|cluster_soft - v2_soft| >= 0.6. Re-cut on all 192 non-tear adjudicated cells (M358) that
pooled rule is Simpson's paradox: the low-margin stratum is 0.704, not a coin flip, and the
gradient runs BACKWARD on Contusion (hi 0.780 / lo 0.861 over n=41/79) - the column that
supplies 204 of the 365. The pooled 0.83 was Contusion's 0.833 whole-column base rate leaking
into an OA statistic through a shared threshold. Verified independently by the researcher
(D352: Contusion hi 0.780 / lo 0.861, OA-family hi 0.783 / lo 0.294).

So:
  Contusion   WHOLESALE      556 cells, cluster right ~0.833 over 120 adjudicated
  OA family   MARGIN >= 0.6  147 cells, cluster right ~0.78 over 23 adjudicated
  Effusion (n=9) and Synovitis (n=6)  LEFT - too thin to bank either way
  the other sub-threshold cells       LEFT - cluster 0.704 there, and 0.704 is not bankable

THE RESIDUAL-RISK BUDGET, STATED BEFORE BANKING. At 0.833 on Contusion the expected breakage
is ~93 cells; at 0.78 on the gated OA cells, ~32. Roughly 125 wrong cells bought for roughly
578 right ones. That is the whole budget and it is what a later refutation is measured against.
It is an order of magnitude looser than v6tears' ~14, which is why this is a separate table.

The written value is the cluster's SOFT mean, for the reason `apply_lone_dissent.py` gives:
v2's targets are confidence-graded, and writing hard 0/1 into exactly the cells an arm varies
would smuggle a target-distribution change into a one-variable arm.

    python research/scripts/build_v7_nontear.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LABELS = Path("D:/data/volumetric/mri/processed/rsna_knee_abnormality/labels")
SOURCE = LABELS / "report_labels_v6tears.csv"
SELECTOR = LABELS / "v2_lone_dissent.csv"
TARGET = LABELS / "report_labels_v7nontear.csv"
UID = "StudyInstanceUID"
WHOLESALE = ("Contusion",)
GATED = ("Medial OA", "Lateral OA", "PF OA", "Baker's")
MARGIN_FLOOR = 0.6

log = logging.getLogger(__name__)


def selected(selector: pd.DataFrame) -> pd.DataFrame:
    """Return the selector rows the per-column rule substitutes."""
    margin = (selector["cluster_soft"] - selector["v2_soft"]).abs()
    wholesale = selector["column"].isin(WHOLESALE)
    gated = selector["column"].isin(GATED) & (margin >= MARGIN_FLOOR)
    return selector[wholesale | gated]


def substitute(source: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Return `source` with the cluster's soft value written into the selected cells."""
    substituted = source.set_index(UID).copy()
    for row in rows.itertuples(index=False):
        substituted.at[row.study_uid, row.column] = row.cluster_soft
    return substituted.reset_index()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    source = pd.read_csv(SOURCE)
    selector = pd.read_csv(SELECTOR)
    rows = selected(selector)
    missing = set(rows["study_uid"]) - set(source[UID])
    if missing:
        message = f"{len(missing)} selected studies are absent from {SOURCE.name}"
        raise ValueError(message)

    substituted = substitute(source, rows)
    changed = int((substituted.set_index(UID) != source.set_index(UID)).sum().sum())
    log.info("selector rows %d, substituted %d, studies %d", len(selector), len(rows), rows["study_uid"].nunique())
    log.info("per column %s", rows["column"].value_counts().to_dict())
    log.info("v2=1 over-calls among them %d", int((rows["v2_value"] == 1).sum()))
    log.info("cells changed %d", changed)
    substituted.to_csv(TARGET, index=False)
    log.info("wrote %s", TARGET)


if __name__ == "__main__":
    main()
