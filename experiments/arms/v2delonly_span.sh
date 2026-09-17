#!/usr/bin/env bash
# The lone-dissent correction with its BROKEN HALF REMOVED. One variable from `band_span`: the table.
#
# WHY THIS ARM EXISTS. D363 adjudicated 56 blind cells from the lone-dissent selector, stratified by
# DIRECTION before any verdict existed, and the two halves are not the same operation:
#
#   deletions (v2=1 -> cluster 0)   2/28 wrong = 0.071 [0.020, 0.226]
#   additions (v2=0 -> cluster 1)  22/28 wrong = 0.786 [0.605, 0.898]
#
# Constant-0 floor on those 56 is 0.857, so neither rate may be read as an accuracy. 21 of the 22
# addition errors are OA columns, all on one mechanism: the cluster asserts a compartment the report
# does not NAME ("gonarthrosis" becoming a positive `Lateral OA` cell). That is D319's
# compartment-assignment problem arriving in the label table.
#
# THE TABLE. `report_labels_v2_delonly.csv`, written by
# `research/scripts/apply_lone_dissent.py --direction down`: the 2,080 selector rows where
# `cluster_soft < v2_soft`, soft values as always, `__conf`/`__verdict` untouched. The 235 addition
# rows are deliberately skipped and v2 stands there. Verified at build: 2,080 cells changed.
#
# THE RESIDUAL-RISK BUDGET, BEFORE THE RUN. At the measured rates: ~2,080 deletions at ~93% right
# (~+1,934 net) and zero additions bought. `v2_corrected` bought the same deletions AND ~235
# additions at ~21% right against a v2 that was right ~87% there, so it is carrying about -185 known
# -wrong cells that this table does not. This is the FIRST table in the chain whose error budget was
# measured by direction rather than pooled.
#
# THE PRICE. 12 epochs, ~35 min on the 5090. It corrects 2,080 cells against v7nontear's 703 and
# v6tears' 1,379 - the largest single correction in the family, and the only one with a
# direction-stratified adjudication behind it. Odds it clears 1.5% on the twelve-column mean alone:
# low, as for every table arm. It is worth the card because it is the only way to price the
# one-directional operation against a model, and because every other table in the chain
# (`v2_corrected`, `v6tears`, `v5blend`, `v7nontear`) is now known to carry the broken direction.
#
# THE RULER, AND THE SAME TRAP. `--eval-labels` stays pinned to v2, so the ledger number is
# comparable to `band_span` 0.8565, `v6tears_span` and `v7nontear_span` 0.8160. A member trained on a
# table whose errors it was taught NOT to reproduce reads a FLOOR on the v2 ruler by construction.
# The read is the corrected T1 rescore off this run's holdout npz.
#
# PREDICTION, written before the run. On the v2 ruler: BELOW `band_span`, because the ruler still
# holds the ~2,080 over-calls this table removes - a v2-ruler drop is expected and is not evidence
# against the table. Against `v7nontear_span` 0.8160 on that same ruler: ABOVE it, because this table
# removes an error class without buying the addition class v7nontear bought. On the corrected T1
# ruler: at or above every other member in the chain. T1 below `v7nontear_span` refutes the
# one-directional reading and says the deletions are not the half that carries - PLUMBING first,
# assert `labels` names v2_delonly and `eval-labels` names v2 before writing any number down.
#
# reading: D:/models/rsna_knee_abnormality/v2delonly_span_s2026*.holdout.npz
# bead: knee-gcjig
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

DELONLY="$PROCESSED/labels/report_labels_v2_delonly.csv"
require "$DELONLY"
require "$LABELS"

run_train v2delonly_span \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 \
  --labels "$DELONLY" --eval-labels "$LABELS" --out "$MODELS/v2delonly_span_s2026.pt"
