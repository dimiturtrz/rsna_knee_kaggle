#!/usr/bin/env bash
# An exponential moving average of the weights, scored and saved instead of the live ones.
#
# WHAT IT REPLACES. Averaging several runs - folds, seeds - is the usual way to stop an answer
# depending on where training happened to stop, and it is banned here: it buys variance reduction, not
# signal, and costs one full run per member. An EMA buys the same thing from ONE run, for one extra
# copy of the weights and no extra GPU time. That is the trade this arm is testing.
#
# THE NUMBER IT TARGETS. Three seeds of the same recipe spread 0.0062 on this ruler - the noise floor
# every verdict in the ledger is read against. That spread is the model landing on a different point
# of the same basin depending on the last few hundred steps, which is exactly what an average over
# recent weights damps. If it works, it should shrink the gap between the best epoch and the last one.
#
# WHY IT IS NOT OBVIOUSLY FREE. We already keep the BEST epoch by holdout score, which is itself a
# selection over the trajectory - so an EMA is partly buying something we have. It should therefore
# help LEAST here and MOST in the final fit, where there is no holdout to rank epochs by and the last
# epoch is kept by construction. That is the case this arm is really scouting.
#
# control  wsfold0_e6 = 0.8457 - same recipe, live weights.
# cost     one extra copy of the weights in VRAM; no extra step time.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train weight_average \
  $RECIPE --epochs 12 \
  --slice-pooling max --whole-stack --ema \
  --recompute-activations \
  --labels "$LABELS" --out "$PROCESSED/ema_ws_s2026.pt"
