#!/usr/bin/env bash
# recal_band_dense.sh — is the v3 supervision table's +0.0105 a property of the RE-MAP, or of the 224 px
# corner it was measured in? One arm answers it, and the ledger names this as follow-up (2) of knee-8wa0.
#
# WHAT LANDED. `recal_holdannot` swapped `holdannot`'s supervision table for `report_labels_v3.csv`, whose
# six reader levels are replaced by their measured positive rates on the 58 radiologist-read studies
# (`kneemri.evaluation.level_calibration`, fitted under knee-7oep). Read on both rulers as knee-9198
# requires, GOLD first because the proxy's own truth IS the report table and is biased against label-side
# movement:
#
#   ruler                                     holdannot   recal_holdannot   delta
#   GOLD (supervision_gap, 58 studies)          0.8932        0.9037        +0.0105
#   proxy (ruler_check on v2, 909 studies)      0.8506        0.8515        +0.0009
#   proxy, confident-only (71.0% of cells)      0.8740        0.8754        +0.0014
#
# Gold up past the 0.0062 floor, proxy not down: the arm's own LANDS branch. The reader over-calls in the
# middle of its scale — the 0.68 level is 62% negative and was trained as 0.68 positive on 12% of cells —
# and re-pointing the gradient at the measured rate moves the model toward the image-read process for free.
#
# WHY A SECOND CORNER IS THE RIGHT NEXT SPEND, AND A ROSTER REBUILD IS NOT. The gold read is n=58, which is
# a level and not a selector (knee-kddo): +0.0105 clears the floor but not by much, and one arm at one
# recipe corner cannot separate the re-map from the corner. The ledger's follow-up order is explicit —
# v3 becomes the default table (free), THEN a dense-corner replication, and only THEN a roster rebuild,
# which is many arms and must not be bought on a single n=58 read. This arm is that replication and it is
# the cheapest one available: `band_dense_hold` already exists as a matched control read on BOTH rulers,
# so nothing extra goes on the card to obtain the comparison.
#
# THE SINGLE VARIABLE IS THE TABLE. Every other flag is byte-identical to `band_dense_hold.sh`.
#
# `--eval-labels` IS PINNED TO v2, AND THAT IS THE LESSON OF THE ARM ABOVE, NOT A PREFERENCE.
# `recal_holdannot` passed `--labels v3` with `--eval-labels` unset, so its holdout truth was binarised at
# 0.5 against the RE-MAPPED values: the ruler moved with the arm and the log ended `best holdout mean AUC
# 0.8964`, +0.0477 over its control, which sat there for an hour looking like the largest single-arm gain
# this project has ever measured. `run_comparison` refuses the pair outright — "the two runs were scored by
# different label tables over their 909 shared studies" — which is the guard working. Any label-VALUE arm
# pins the eval table.
#
# control  band_dense_hold = 0.8573 proxy @ e7 (curve 0.7452 / 0.7779 / 0.8315 / 0.8403 / 0.8482 / 0.8546 /
#          0.8541 / 0.8573 / 0.8546 / 0.8543 / 0.8536 / 0.8535), and gold via `supervision_gap`.
# cost     ~40 min — `band_dense_hold`'s own runtime, unchanged.
#
# PRE-REGISTERED, on GOLD first, for the same reason the arm above was:
#   REPLICATES — gold up by 0.0062 or more over `band_dense_hold`'s gold, proxy not down by the floor.
#     The re-map is a property of the TABLE and not of the 224 corner. That licenses follow-up (3), the
#     roster rebuild, which is many arms and is not licensed by anything today.
#   CORNER — gold within 0.0062 either way. The +0.0105 was the 224 px / 12 slice corner, not the re-map.
#     v3 stays the default table (it costs nothing) and the label lane stops being priced as a member lane.
#   REFUTES — gold DOWN by 0.0062 or more. At the dense corner the re-map moves training away from the
#     image-read process, which would make the 224 result corner-specific in the opposite direction and
#     send the follow-up per-finding rather than pooled.
#
# READING IT. Proxy: `python -m kneemri.evaluation.ruler_check` against v2, NEVER the training log. Gold:
# `python -m kneemri.evaluation.supervision_gap --labels "$LABELS" --records
# $PROCESSED/recal_band_dense_s2026.pt`. Both, per knee-9198.
#
# PRICE: ~40 min of card for a mechanism the label lane has measured positive twice (blendlabels +0.0034,
# recal_holdannot +0.0105 gold). P(clears the 0.0062 gold floor) ~ 0.5 — a replication of a single n=58
# read at a different corner, with the corner itself the named suspect. It is priced as an INSTRUMENT: it
# does not produce a seatable member (the control is 0.8573 proxy and this arm is a table swap, not a
# quality lever), and what it buys is whether follow-up (3) may ever be armed.
#
# bead: knee-7uoq
# It is follow-up (2) of knee-8wa0; the two-ruler rule is knee-9198, the level fit is knee-7oep.
#
# deliberate: --labels       (report_labels_v3.csv — the single variable)
# deliberate: --eval-labels  (v2, so the proxy does not move with the arm; see above)
# deliberate: --hold-annotations (the gold ruler REFUSES the checkpoint without it, and it closes the leak
#   from fitting the level map on those same 58 studies)
# deliberate: every other flag — they are band_dense_hold's and may not be touched in this arm.
#
# NO --vram-gb: DeviceBudget derives the cap from --slices — 20 slices reserve 28.26 GB.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

RECAL="$PROCESSED/labels/report_labels_v3.csv"
require "$RECAL"
require "$LABELS"

run_train recal_band_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --hold-annotations \
  --labels "$RECAL" --eval-labels "$LABELS" --out "$PROCESSED/recal_band_dense_s2026.pt"
