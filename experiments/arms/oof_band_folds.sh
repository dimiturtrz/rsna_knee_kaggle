#!/usr/bin/env bash
# oof_band_folds.sh — the control's geometry, on a table whose middle rung carries the model's own order.
#
# WHAT MOVES AND WHAT DOES NOT. Exactly one token differs from `oof_folds.sh`: `--labels` points at
# `report_labels_v6band50.csv` instead of `report_labels_v2.csv`. Same encoder, same 224 px, same 8
# epochs, same slice geometry, same seed, same five folds. That is the whole design — a label-table A/B
# where the control is already on disk (`oof_fold[0-4]_s2026`), so this arm buys a PAIRED difference on
# the same 58 gold studies rather than a level to be compared against a remembered number.
#
# WHAT THE TABLE IS. `core/data/labels/band_resolution.py`, share 0.5, raw scale. Within each finding,
# the cells sitting on the reader's middle rung (raw 0.68, calibrated 0.3776 — the level the radiologist
# agrees with 37.8% of the time) are ranked by that finding's out-of-fold score from `oof_fold[0-4]`, and
# re-spread symmetrically about the rung by up to half the nearer neighbour's distance. 6461 cells,
# 12.22% of the table, move; every other cell is byte-identical.
#
# WHY THE MIDDLE RUNG AND WHY NOW. D301 censused it over all 4406 corpus rows: nine of the twelve columns
# hold between 415 and 568 band cells against a uniform 538, seven within 6% of uniform. Effusion is 3x
# uniform, Synovitis and Fracture a quarter. So the mechanism's per-column weight is nearly FLAT under a
# scorer whose weights are exactly flat (D298: the board is `(1/12) Σ AUC_i`, verbatim from the
# evaluation page) — the best shape a label-side lever can have here, and the `2/12 × gain` ceiling that
# was the reason to doubt it is ruled out on 4406 rows with no truth involved. D300 measured that the
# model can order those cells at all: within-band AUC 0.7490 on the 98 gold band cells, z +4.14 against
# the 0.5409 null the statistic actually has, and 0.568 precision against a 0.378 base rate in the top
# 37 — a 1.50x lift.
#
# THE CENSUS LINE THAT LOOKS LIKE A BUG AND IS NOT. Every column reports the identical
# `mean 0.6800 range [0.6100, 0.7500]`. Correct by construction: the percentile map is uniform on [0, 1]
# within each column, so at one share every column's re-spread is the same affine image of it. Only the
# ASSIGNMENT of cells to values differs per column, and that assignment is the entire treatment.
#
# THE SYMMETRY IS DELIBERATE AND IT COSTS THE WIDER SIDE'S ROOM. The reader's neighbours are not
# equidistant (raw: 0.40 below, 0.14 above), so a spread reaching both would drag the band's mean
# 0.68 -> 0.615 alongside its order. A gain would then be unreadable: pixels ordering the rung, or 0.68
# merely being too high — and D300's calibration says 0.68 IS too high, which is exactly what makes the
# two indistinguishable. Reaching the nearer distance both ways moves ONE thing.
#
# PRE-REGISTRATION, written before the run. Decided on the PAIRED gold@58 under `rankpooled` (the
# registered internal ruler, paired sd 0.0096) against `oof_fold[0-4]_s2026`:
#   * >= +0.020 — the rung carries orderable signal the flat target was throwing away; re-derive the
#     table at full recipe strength and take the share dial to 0.75.
#   * <= -0.020 — self-training wrote the model's own errors back into the table; the middle-rung route
#     closes and the 0.6879 rung is not attempted either.
#   * between — NULL AT THIS RESOLUTION, and that is the honest word. The gold ruler's fold-to-fold sd is
#     0.0550 and n is 58; it cannot see a point and a half. A null here retires the mechanism as a
#     CHEAP lever and does not refute it.
# The board-comparable row is the flat `macro` (D298 §3(a)) and it is quoted beside the decision, never
# instead of it — the two read 0.028 apart on one system.
#
# WHAT IT IS, SAID PLAINLY. Self-training with the reader's bracket kept. The order comes from a model
# fitted on this corpus's own report labels, so a systematic reader error the model learned is one this
# table now asserts more confidently. Three things bound it and none removes it: the model never saw the
# study whose cell it orders, a cell cannot cross a rung the reader did not already bracket, and `--share`
# prices the whole treatment on one dial whose floor is the identity.
#
# COST: five runs, ~25 min each at 224/8, ~2.1 h of card.
#
# deliberate: --fold 0..4 (all five, because the paired control is the five-fold stitch)
# deliberate: --image-size 224 (matched to the control; a member recipe would break the pairing)
# bead: knee-27kuj
# reading: oof_band_fold*_s2026.holdout.npz
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

BAND_LABELS="$PROCESSED/labels/report_labels_v6band50.csv"
require "$BAND_LABELS"

for fold in 0 1 2 3 4; do
  run_train "oof_band_fold${fold}" \
    $RECIPE --fold "$fold" --epochs 8 --slice-pooling max --whole-stack $THROUGHPUT \
    --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 --image-size 224 \
    --labels "$BAND_LABELS" --out "$MODELS/oof_band_fold${fold}_s2026.pt"
done
