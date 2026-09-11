#!/usr/bin/env bash
# band_dense's recipe with ONE token changed: the supervision table.
#
# WHY. Measured 2026-09-11 on gold57 (the 58 organiser-labelled studies in raw/train.csv), statistic
# cell disagreement at midpoint on both sides:
#
#   flight0234 report_labels_v4hybrid  0.1813
#   lixin73    report_labels_gpt56sol  0.1813
#   steven     llm_labels_v4_blend     0.1827
#   steven     llm_labels_full         0.1988
#   OURS       report_labels_v2        0.2047
#
# Paired study-clustered bootstrap, 20,000 resamples, 57 studies: ours - majority = +0.0219,
# 95% CI [+0.0044, +0.0395]. Our error is one-sided, FP 108 against FN 32, worst on Contusion (we call
# 36 positive where gold says 18). experiments/lib.sh:72 puts 286 arm invocations on the losing table.
# Unlike report_labels_v3 this is not circular: v3 was FITTED on these 58 and is v2 RE-CUT (dropping the
# 0.68 rung reproduces its 0.1696 exactly); the foreign tables were built by other people and none of
# them pasted gold — the seven tables that DID read exactly 0.0000 on all 58 and are excluded.
#
# THE RULER IS RIGGED AGAINST THIS ARM AND THAT IS THE POINT. Our 909-study holdout labels are derived
# from report_labels_v2, so a member trained on a truer table is scored on a v2-derived ruler and reads
# worse the truer it gets. A NEGATIVE OFFLINE NUMBER IS NOT A REFUTATION HERE. This is the reason the
# axis has stayed closed: every offline test of a label swap this project can run is a test of
# agreement-with-v2 wearing a score's clothes.
#
# control  band_dense_s2026 = 0.8578 @ e8 (best single member the project owns), same recipe, v2 labels.
# cost     one arm at band_dense's step time, 12 epochs, NO new cache — the decode is keyed on geometry
#          and the labels are not part of the cache key, so this reuses 336px_20sl_0.1-0.9band.
#
# PREDICTION, written before the run.
#   (a) within -0.010 of 0.8578 -> the table survives a ruler built against it; ship the member and read
#       the BOARD, which is image-read radiologist consensus (README:3844-3847) and the only instrument
#       independent of both tables.
#   (b) below -0.020 -> the gold-space advantage does not survive training and the axis closes on the
#       pixels rather than on the ruler.
#   (c) ABOVE 0.8578 -> the strongest possible read: a table wins on a ruler made of its rival.
# The board delta against an otherwise identical entry is the number that decides it, and it costs one
# slot. Nothing here licenses a seating.
# reading: foreign_labels_dense_s2026.holdout.npz
# bead: knee-y8t8i
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

FOREIGN_LABELS="D:/data/volumetric/mri/raw/knee_label_tables/lixin73_rsna-knee-llm-report-labels-sol56/report_labels_gpt56sol.csv"
require "$FOREIGN_LABELS"

run_train foreign_labels_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$FOREIGN_LABELS" --out "$MODELS/foreign_labels_dense_s2026.pt"
