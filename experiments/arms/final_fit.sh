#!/usr/bin/env bash
# final_fit.sh [epochs] — fit ONE model on EVERY study, which is what we submit now that fold and seed
# ensembling are out.
#
# WHAT CHANGED AND WHY. Our leaderboard entries have been rank-averaged fold members: five at 0.888,
# then two whole-stack ones. That is ensembling-by-k-fold, and it is banned here for a good reason -
# it buys variance reduction rather than signal, and it costs one full run per member. It also has a
# quieter cost that matters more: every member is fitted on 80% of the corpus, so the ensemble never
# contains a model that has seen the fifth fold. This arm trades N models on 80% for ONE model on 100%.
#
# THE THING THAT MAKES IT DIFFERENT FROM EVERY OTHER ARM: with --fold -1 no study is held out, so
# there is NO SCORE. Nothing here can be compared to 0.8457, the run cannot rank its own epochs, and
# train.py keeps the LAST epoch by construction rather than the best. This arm produces a submission,
# not a measurement - and it must therefore only ever be run on a recipe that a fold-0 arm has already
# settled.
#
# WHICH IS WHY THE EPOCH COUNT IS AN ARGUMENT AND NOT A DEFAULT DECISION. It is imported from the arm
# that settled the recipe: whole-stack peaked at epoch 6 of a 12-epoch one-cycle on both folds tried.
# Seven is the honest reading of that, and it is a mild EXTRAPOLATION rather than a measurement: a
# 7-epoch one-cycle is not the schedule whose epoch 6 scored 0.8457, because one-cycle is defined over
# total_steps and a shorter budget anneals faster. knee-1je measured that direction as favourable
# (12 epochs beat 24, 0.8457 vs 0.8413) but not at this length.
#
# EMA IS ON, and this is the case it was argued for: with no holdout there is no best-epoch selection
# to fall back on, so the answer would otherwise depend entirely on where the last step landed.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
EPOCHS="${1:-7}"
require "$LABELS"

run_train "final_fit_e${EPOCHS}" \
  --fold -1 --epochs "$EPOCHS" --one-cycle --weight-floor --rank-weight 1.0 \
  --slice-pooling max --whole-stack --ema \
  --recompute-activations --vram-gb 22 \
  --labels "$LABELS" --out "$PROCESSED/final_ws_e${EPOCHS}.pt"
