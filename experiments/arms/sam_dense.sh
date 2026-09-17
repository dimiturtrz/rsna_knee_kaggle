#!/usr/bin/env bash
# sam_dense.sh — descend from the worst point of a ball, on the recipe the corner's best member ran.
#
# WHY THIS AND NOT ANOTHER KNOB. The weight-space axis closed this morning: the uniform soup is negative
# and degrades monotonically with K (-0.0044 per doubling), and the sibling soup landed ON that law to
# 0.0004, which killed the greedy escalation on arithmetic before it cost 90 minutes of card. What that
# leaves is the axis the owner keeps naming — the corner reads ~0.865 held out while the board reads
# 0.944, so the thing worth buying is GENERALISATION, not another point of fit. SAM is the cheapest
# training-time technique with a published effect size on that axis.
#
# THE SINGLE VARIABLE IS --sharpness 0.05 AND NOTHING ELSE. Control is calib_dense_hold_s2026 = 0.8634
# @holdout909 midpoint, which ran this exact flag list. No second control arm is needed and no precision
# confound exists: the stamped recipe reads "bf16": "True", so there is no GradScaler on this path at all
# and `Sharpness` has no scale factor to `unscale_`.
#
# PREDICTION, WRITTEN BEFORE THE RUN. If flatness is what separates our corner from the board, the mean
# lands at or above 0.8684 and the gain is SPREAD across labels — a sharpness penalty has no per-finding
# preference and cannot concentrate. A gain on one or two columns is the mechanism REFUTED while the mean
# reads as a win, which is what the per-label read exists to catch.
#
# THE REFUTATION CONDITION. Below 0.8604 says the radius is too large for a 12-epoch one-cycle at a
# backbone rate of 4e-5, OR that flatness is not the axis — and those two are told apart by the per-label
# row, not by the mean. Between 0.8604 and 0.8684 is inside the board's own noise on a solo read and is
# DIRECTIONAL ONLY: keep the flag, never plan around it.
#
# THE HARNESS SUSPECT THIS ARM NAMED IS VOID. It read `Sharpness.climbed` returning False on a
# non-finite gradient norm as a live fp16/GradScaler path; the stamp says bf16, so that route does not
# exist here. The suspect was written from this comment rather than from the stamp. It does not touch the
# verdict, which cleared the harness by direct evidence: the ascent witness fired live at radius 0.05.
#
# cost     279 s/epoch measured, DOUBLED by the second forward and backward, x 12 epochs = ~1.9 h.
#          Zero submission slots.
#
# bead: knee-zqouw
# reading: sam_dense_s2026.holdout.npz
#
# NO --vram-gb: DeviceBudget derives the cap from --slices — 20 slices reserve 28.26 GB.
# control  calib_dense_hold_s2026 = 0.8634 (ledger, holdout909 midpoint, identical recipe)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train sam_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --sharpness 0.05 \
  --labels "$LABELS" --out "$MODELS/sam_dense_s2026.pt"
