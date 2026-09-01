"""FP-vs-TP gate for cleanlab scores on the gold 58.

Claude's question: do the lowest label-quality scores land on radiologist-confirmed false
positives, or do they also flag radiologist-confirmed true positives at the same rate?
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from core import cli
from core.data.labels.reader_agreement import POSITIVE, ReaderAgreement
from kneemri.evaluation.label_errors import (
    DetectorGate,
    GoldCensus,
    LabelErrorDetector,
    ReaderTable,
)
from kneemri.evaluation.published_ruler import PublishedRuler

DEFAULT_LABELS = Path("labels") / "report_labels_v2.csv"
MIDPOINT = 0.5


def main() -> None:
    command = cli.Command(__doc__, logging.getLogger(__name__))
    command.parser.add_argument("--labels", type=Path, default=None)
    arguments = command.parsed()
    root = command.rooted(arguments)
    labels = arguments.labels or root.processed() / DEFAULT_LABELS

    table = pd.read_csv(labels)
    annotated = ReaderAgreement.annotated(root)
    published = root.processed() / "published"

    identifiers, per_seed = PublishedRuler(published).per_seed()
    reader = ReaderTable.of(table, identifiers)
    predicted = np.mean(np.stack(per_seed), axis=0)

    detector = LabelErrorDetector(predicted, reader)
    quality = detector.quality()

    agreement = ReaderAgreement()
    cells = agreement.cells(annotated, table)
    census = GoldCensus.of(cells, agreement.errors(cells))

    row_index = {name: i for i, name in enumerate(identifiers)}
    col_index = {name: i for i, name in enumerate(census.target[:12] if len(census.target) >= 12 else census.target)}
    inside = np.array([study in row_index for study in census.study], dtype=bool)
    rows = np.array([row_index[study] for study in census.study[inside]], dtype=np.int64)
    cols = np.array([col_index[target] for target in census.target[inside]], dtype=np.int64)

    quality = quality[rows, cols]
    cells = cells.iloc[inside].reset_index(drop=True)
    census = census.restricted(inside)

    # True positives: reader said YES and radiologist agrees
    true_positive = (cells["verdict"] == POSITIVE).to_numpy() & (cells["truth"] > MIDPOINT).to_numpy()

    fp_vs_all = DetectorGate(quality, census.false_positive)
    fp_vs_tp = DetectorGate(
        quality[true_positive | census.false_positive],
        census.false_positive[true_positive | census.false_positive],
    )

    n_fp = int(census.false_positive.sum())
    n_tp = int(true_positive.sum())

    lines = [
        f"scoreable cells: {len(quality)}",
        f"false positives (radiologist-confirmed wrong): {n_fp}",
        f"true positives (radiologist-confirmed correct): {n_tp}",
        f"misses: {int(census.missed.sum())}",
        f"true negatives: {len(quality) - n_fp - n_tp - int(census.missed.sum())}",
        "",
        f"FP vs ALL scoreable:  AUC {fp_vs_all.separation():.4f}",
    ]
    for row in fp_vs_all.ranked():
        lines.append(row.line())

    lines += [
        "",
        f"FP vs TRUE POSITIVES only: AUC {fp_vs_tp.separation():.4f}",
        "(control half: does the flagger separate wrong YES from correct YES?)",
    ]
    for row in fp_vs_tp.ranked():
        lines.append(row.line())

    lines += ["", "false-alarm rate on true positives at each budget:"]
    for row in fp_vs_tp.ranked():
        alarms = row.budget - row.caught
        fpr_on_tp = alarms / n_tp if n_tp else float("nan")
        lines.append(f"    budget {row.budget:4d}: {alarms:2d}/{n_tp} true positives flagged = {fpr_on_tp:.3%}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
