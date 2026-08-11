#!/usr/bin/env bash
# Slice count, deconfounded - REFUTED, and it took the excuse with it.
#
# Reading 20 slices instead of 12 once scored -0.0016, and that was explained away: at a fixed group
# of 3, more slices means each group spans less tissue, so the arm changed coverage AND thickness.
# Max pooling encodes each slice alone, which makes a group's physical thickness independent of the
# slice count. Under it the explanation predicts 20 slices should WIN.
#
# control  rank24_s2026            0.8272
# Q  12 slices, max pooling        0.8310   +0.0038  (isolates the pooling change)
# R  20 slices, max pooling        0.8309   -0.0001 against Q - the prediction failed
#
# So more slices inside the band buy nothing, and the thickness story was wrong rather than unlucky.
# The pooling change survives on its own and is carried by compose_small / compose_base.
source "$(dirname "$0")/../lib.sh"
require "$LABELS"

run_train maxpool_12 \
  --fold 0 --epochs 24 --one-cycle --weight-floor --rank-weight 1.0 \
  --slice-pooling max --labels "$LABELS" --out "$PROCESSED/max12_s2026.pt"

run_train maxpool_20 \
  --fold 0 --epochs 24 --one-cycle --weight-floor --rank-weight 1.0 \
  --slice-pooling max --slices 20 --labels "$LABELS" --out "$PROCESSED/max20_s2026.pt"
