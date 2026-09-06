#!/usr/bin/env bash
# reader_end_chain.sh — score the v3 label table the moment the corpus reader finishes. Zero card.
#
# WHY IT IS A SCRIPT AND NOT A LINE OF PROSE. `recal_dense` sat four days with its deciding gold
# number never taken — ledger row written, convergence clean, every gate green — because the reading
# lived in prose and needed a second command (knee-v78o1). A chain that has to be retyped at handoff
# is a chain that gets retyped wrong.
#
# THE ASSERT IS THE POINT, AND IT IS NOT A ROW COUNT.
#
#   1. THERE ARE TWO FILES CALLED report_labels_v3.csv. A stale one sits in the labels directory
#      (mtime 2026-08-30) and ALREADY reads 4407 lines; its `__verdict` columns are byte-identical to
#      v2's while its numeric columns differ on 12 of 12 findings. The live reader writes to
#      `.agent-runtime/scratch/` (`OUT_DIR`, `corpus_reader.py`). Asserting on the NAME passes
#      instantly against a table a week old and reports a finished chain.
#   2. `wc -l` IS NOT A ROW COUNT HERE. Report text embeds newlines inside quoted CSV fields — the live
#      table read 3564 lines at 3525 studies. The assert counts DISTINCT STUDIES.
#   3. THE SIDECARS ARE THE EXIT SIGNAL. The reader writes the census and the finish-reason
#      distribution only after the walk completes, so their existence says the process finished rather
#      than that it reached a row count.
#
# WHAT THE TWO READINGS ARE. `reader_agreement` scores the table against the 58 radiologist-annotated
# studies — the only truth in this project not produced by a reader, and the gate v3 has to pass:
# BETTER than v2's 140 net wrong (108 confident false positives + 32 misses). `ruler_swap` then holds
# the predictions fixed and swaps the truth table underneath, which is the different question: what
# the seat ORDER looks like when the ruler changes.

set -euo pipefail
cd "$(dirname "$0")/.."

TABLE=.agent-runtime/scratch/report_labels_v3.csv
# The corpus is 4407 studies, read from the reader's OWN census sidecar (every one of the twelve
# findings reports total=4407) and from its finish-reason table (4407 rows, all `stop`, no truncation).
# This line said 4406 until 2026-09-06 — a number carried over from prose, not from the corpus — and it
# refused the completed walk with `REFUSED: 4407 distinct studies, expected 4406`.
EXPECTED=4407

.venv/Scripts/python.exe - "$TABLE" "$EXPECTED" <<'PY'
import sys
from pathlib import Path

import pandas as pd

table, expected = Path(sys.argv[1]), int(sys.argv[2])
for sidecar in (table.with_suffix(".census.csv"), table.with_suffix(".finish_reasons.csv")):
    if not sidecar.exists():
        raise SystemExit(f"REFUSED: {sidecar.name} absent — the reader writes it only after the walk")
studies = pd.read_csv(table).iloc[:, 0].astype(str).nunique()
if studies != expected:
    raise SystemExit(f"REFUSED: {studies} distinct studies, expected {expected}")
print(f"OK: {studies} distinct studies, both sidecars written")
PY

.venv/Scripts/python.exe -m core.data.labels.reader_agreement --labels "$TABLE"
.venv/Scripts/python.exe -m kneemri.kaggle.ruler_swap --table "$TABLE"
