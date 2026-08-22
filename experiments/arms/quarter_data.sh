#!/usr/bin/env bash
# quarter_data.sh — the third point, because one interval cannot locate the knee.
# bead: knee-krnd
#
# WHAT THIS ARM IS. `half_data` measured 50% of the training studies at 0.8344 against `band_dense`'s
# 0.8578 at 100% — a raw −0.0234, ~0.019 net of the registered schedule confound, and the only axis this
# project has ever measured above the 1.5% bar. That is ONE interval. This arm adds a second, at 25%,
# so the slope is FITTED rather than assumed.
#
# WHY IT MATTERS MORE THAN AN INSTRUMENT USUALLY DOES. Every price now on the board is a log-linear
# extrapolation from that single interval, and each of them is currently deciding what gets built:
# MRNet at ~+0.0095 (knee-ombx), a full doubling of the corpus at ~+0.020, the `final_fit` 80→100% step
# at ~+0.0064 (knee-5895). If the curve is already flattening at our operating point, all three are
# optimistic and the data axis is worth materially less than the one point suggests. If it is not, the
# external-data schedule stands as written. Nothing else on the board can distinguish those.
#
# THE THREE READINGS, written before the number exists.
#
#   25→50 MATERIALLY STEEPER THAN 50→100: the curve is flattening, we sit past the steep part, and every
#   extrapolation above is an overestimate. MRNet falls further under the bar and the external-data
#   argument has to be re-made on supervised pretraining alone rather than on volume.
#
#   THE TWO INTERVALS COMPARABLE: log-linear holds across two doublings, which is as much support as a
#   three-point fit can give it. The prices stand and knee-ombx's schedule is priced correctly.
#
#   25% WITHIN THE FLOOR OF 50%: the corpus saturates far below where we sit, which would make the
#   `half_data` reading itself the surprising one and send the whole axis back for a cause — the first
#   suspect being the schedule confound doing more work than `schedule_length` prices it at.
#
# THE SAME CONFOUND APPLIES AND IS CHARGED THE SAME WAY. Quarter the studies is quarter the optimiser
# steps, and `--one-cycle` spans `total_steps`, so this arm compresses the cycle twice as far as
# `half_data` did. `schedule_length` prices step compression at 0.0028 for 0.67× the steps; at 0.25× that
# scales to roughly 0.0056–0.0096, and the bias is DOWNWARD. The three-point fit must be read on the
# CONFOUND-CORRECTED points, not the raw ones — which is precisely why the correction has to be stated
# per point rather than waved at once.
#
# PRICE: ~22 minutes of card. `half_data` ran 204 s an epoch on 50% of the studies, so 25% is ~110 s and
# twelve epochs is ~22 min. P(clears 1.5%) = 0, DECLARED — the arm is handicapped by construction and
# cannot produce a member. It is an INSTRUMENT and is justified as one: it decides whether the only axis
# measured above the bar is priced right before any expensive hour is spent on it.
#
# control  band_dense = 0.8578 (best epoch 8); half_data = 0.8344 (best epoch 6). Both in this README.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train quarter_data \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --train-fraction 0.25 \
  --labels "$LABELS" --out "$PROCESSED/quarter_data_s2026.pt"
