"""Write the tear-column wholesale correction into report_labels_v5blend (D350).

The operation is `column-substitution`, second use: the cluster's value replaces v5blend's
on the four TEAR columns' contested cells only, and v5blend's own content stands everywhere
else. It is not a blend - `table_blend.py` is a mean/tiebreak combiner over foreign readers
of the same report, a different operation (D348).

The measurement behind it: 277 of the 469 blind-adjudicated holdout cells are tear-column
dissent cells, and on all 277 v2 is wrong and the cluster is right - accuracy 0.000 against
1.000, over 223 distinct studies (D350). The selector's fold assignment is a hash of the
REPORT TEXT (`core/data/corpus/folds.py:61`), so no label-side variable enters the draw and
the rate is a population statement about contested tear cells rather than a sample artefact.

Two guards, both from D350:
  1. Only the selector's contested cells are touched. v2's UNCONTESTED tear cells are a
     different population, where the first audit put unanimous-cell error at 1.48%.
  2. The 1.000 rests on 223 studies, so a residual cluster error at the ~1% level is
     invisible at that n - about 14 cells across the corpus's tear-dissent studies. That is
     the whole residual-risk budget, stated before the table is banked.

The written value is the cluster's SOFT mean, not its hard vote, for the same reason
`apply_lone_dissent.py` writes soft: v2's targets are confidence-graded, and writing hard
0/1 into exactly the cells an arm varies would smuggle a target-distribution change into a
one-variable arm. `__conf` and `__verdict` are untouched - a corrected cell stays supervised.

    python research/scripts/build_v6_tears.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LABELS = Path("D:/data/volumetric/mri/processed/rsna_knee_abnormality/labels")
SOURCE = LABELS / "report_labels_v5blend.csv"
SELECTOR = LABELS / "v2_lone_dissent.csv"
TARGET = LABELS / "report_labels_v6tears.csv"
UID = "StudyInstanceUID"
TEARS = ("ACL", "MCL", "Medial Meniscus", "Lateral Meniscus")

log = logging.getLogger(__name__)


def substitute(source: pd.DataFrame, selector: pd.DataFrame) -> pd.DataFrame:
    """Return `source` with the cluster's soft value written into the selected tear cells."""
    substituted = source.set_index(UID).copy()
    for row in selector.itertuples(index=False):
        substituted.at[row.study_uid, row.column] = row.cluster_soft
    return substituted.reset_index()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    source = pd.read_csv(SOURCE)
    selector = pd.read_csv(SELECTOR)
    tears = selector[selector["column"].isin(TEARS)]
    missing = set(tears["study_uid"]) - set(source[UID])
    if missing:
        message = f"{len(missing)} selected studies are absent from {SOURCE.name}"
        raise ValueError(message)

    substituted = substitute(source, tears)
    changed = int((substituted.set_index(UID) != source.set_index(UID)).sum().sum())
    log.info("selector rows %d, tear rows %d, studies %d", len(selector), len(tears), tears["study_uid"].nunique())
    log.info("per column %s", tears["column"].value_counts().to_dict())
    log.info("v2=1 over-calls among them %d", int((tears["v2_value"] == 1).sum()))
    log.info("cells changed %d", changed)
    substituted.to_csv(TARGET, index=False)
    log.info("wrote %s", TARGET)


if __name__ == "__main__":
    main()
