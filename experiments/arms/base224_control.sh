#!/usr/bin/env bash
# bead: knee-vrvus
# reading: base224_control_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   The CONTROL for mergedk_corpus, and nothing else: the identical line on our own 4,407 studies.
#
# WHY IT COSTS A RUN RATHER THAN A LOOKUP
#   No number on the ledger shares this recipe at 224 px. `blendhold` 0.8575 and `base224` are 224 px
#   members from other recipes, and `merged3_corpus` 0.8521 is 336 px. Reading mergedk against any of
#   them would confound the corpus with the geometry or with three other flags, which is the ambiguity
#   this project has paid for twice. One variable means one control, run here.
#
# deliberate: --image-size 224 — the mergedk cache is 224 px because a 336 px merge needs ~43 GB of
# deliberate: new array and /d holds 23 GB (knee-rijuv). The control matches the arm, not the ledger.
#
# CONTROL, STATISTIC, POPULATION, CUT
#   This arm IS the control. Its own number is a per-column ROC-AUC, unweighted 12-column macro,
#   population holdout909, cut midpoint 0.5 — the same three every row it will be compared to carries.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train base224_control \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --image-size 224 --band-low 0.2 --band-high 0.8 \
  --scheme canonical \
  --labels "$LABELS" --eval-labels "$LABELS" --out "$MODELS/base224_control_s2026.pt"
