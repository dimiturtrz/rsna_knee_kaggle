#!/usr/bin/env bash
# final_ws_wf — the WARM twin of `final_ws_df`: the same seat, the same teacher, the same distill
# weight, started from the champion checkpoint instead of from scratch, on a complete four-epoch cycle.
#
# WHAT IT DECIDES, AND IT IS NOT THIS SEAT. `final_ws_df` is the from-scratch control and it is already
# on disk, so this pair prices the WARM START itself at the cost of one run. If the two land inside the
# 0.0062 floor, every retrain from here — tranche one, the merged successor, gen-2 — runs at four epochs
# instead of twelve. Twenty-five minutes to license 3x on all of it is the best ratio on the card today
# (mastermind 05:41Z, against the owner's 05:40Z ask for faster demonstration).
#
# --epochs 4 IS A COMPLETE ONE_CYCLE, NOT A TRUNCATION. `convergence_check`'s TRUNCATED class is a
# twelve-epoch recipe killed early, whose anneal never ran; this recipe ASKS for four and anneals inside
# them. That distinction is the whole reason the comparison is legible.
#
# THE PRIOR IS AGAINST IT, AND THAT IS WHY IT IS WORTH THE CARD. Every distillation arm on the ledger
# peaks LATE — `foreign_distill` e9 of 12, `foreign_distill_mix07` e10-e11 of 12 — and the +0.0107 that
# licenses `retrain_six_distill` is a twelve-epoch term. The one priced warm start this project owns,
# `warm_calib_dense` 0.8562 against `calib_dense_hold` 0.8634, lost by 0.0072. So this is an INSTRUMENT:
# it retires a suspicion or it licenses an axis, and it is not sold as an improvement.
#
# THE READING IS THE WITNESS, NOT A HOLDOUT AUC. Both arms are `--fold -1` and hold nothing out, so no
# ledger row can price either. The comparison is the canonical-holdout witness on the two checkpoints.
# reading: final_ws_wf_s2026.holdout.npz
set -u
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

TEACHER="$PROCESSED/teacher_oof5.npz"
require "$TEACHER"
require "$MODELS/final_ws_cf.pt"

FIT="--fold -1 --ema"
DENSE="--slices 20 --band-low 0.1 --band-high 0.9"
CALIB="--calibrated-targets"
DISTILL="--distill-weight 0.7 --teacher-targets $TEACHER"
WARM="--init $MODELS/final_ws_cf.pt --init-scope all"

run_train final_ws_wf \
  $RECIPE $FIT --epochs 4 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE $CALIB $DISTILL $WARM \
  --labels "$LABELS" --out "$MODELS/final_ws_wf.pt"
