#!/usr/bin/env bash
# The encoder rate under the CURRENT recipe, with resolution held at the control's 336 px.
#
# WHY IT IS THE OBVIOUS ARM NOW. On 2026-08-18 the confound in our best member finally came apart:
# 392 px at the control's 8e-6 scored 0.8459 against wsfold0_e6's 0.8457 — resolution alone is
# +0.0002, nothing. Everything res392 gained (0.8523, +0.0066) therefore sits on the other variable it
# changed, the encoder rate 8e-6 -> 4e-5. That makes the rate the largest unpriced effect we have, and
# it has never been measured under whole-stack: rate_isolate's +0.0051 was taken in the WINDOWED era,
# before a step saw every slice, and a rate interacts with how much each step sees.
#
# ONE VARIABLE. 336 px is the control's own cache and 12 epochs its own schedule; only --backbone-lr
# moves. If this lands near +0.006 the rate is a lever worth sweeping, and 448 px caches stop being
# the story they were bought to be (knee-d2fc). If it lands flat, the gain is specific to reading at
# 392 px WITH the faster rate, which is an interaction and a different arm again.
#
# control  wsfold0_e6 = 0.8457, the same recipe at 8e-6.
# risk     none to the recipe; the output is its own checkpoint and can only join the pool on merit.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train rate_wholestack \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 \
  --labels "$LABELS" --out "$PROCESSED/lr4e-5_ws_s2026.pt"
