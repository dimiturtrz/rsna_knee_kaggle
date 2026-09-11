"""Mask v2 label cells whose compartment is never named in the report.

D365/D366: 711 v2 OA-column positives carry no compartment lexeme anywhere in
their report, and 692 of them (97.3%) sit in the UNANIMOUS stratum -- every
extractor agrees on a compartment the report never stated. That is shared
inference, not disagreement, so no dissent-based operation reaches these cells.

The operation is a MASK, never a rewrite: a cell the report does not support is
a cell we do not know, and writing a 0 there would assert absence with the same
confidence that put the 1 there. Masking removes supervision; it cannot invent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ID_COLUMN = "StudyInstanceUID"


def masked(labels: pd.DataFrame, cells: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Return `labels` with each (study, column) cell in `cells` set to NaN."""
    out = labels.copy()
    position = {uid: i for i, uid in enumerate(out[ID_COLUMN])}
    written = 0
    skipped = 0
    for uid, column in zip(cells["study_id"], cells["column"], strict=True):
        row = position.get(uid)
        if row is None or column not in out.columns:
            skipped += 1
            continue
        out.iloc[row, out.columns.get_loc(column)] = float("nan")
        written += 1
    return out, written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    cells = pd.read_csv(args.cells)
    table, written, skipped = masked(labels, cells)

    before = int(labels.isna().sum().sum())
    after = int(table.isna().sum().sum())
    table.to_csv(args.out, index=False)
    print(f"cells requested {len(cells)} written {written} skipped {skipped}")
    print(f"nan before {before} after {after} delta {after - before}")
    print(f"shape {table.shape} -> {args.out}")


if __name__ == "__main__":
    main()
