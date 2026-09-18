#!/usr/bin/env bash
# headlr_peak18.sh — headlr_ship's line on a budget that ends where the recipe peaks.
#
# bead: knee-yatne
# reading: headlr_peak18_s2026.holdout.npz
#
# CONTROL. headlr_ship_s2026, 0.9059 ledger75 / 0.8702 midpoint, peak e17 of 24 — the highest
# offline75 member on disk. Single knob against it: --epochs 24 -> 18.
#
# MECHANISM. That control peaks at e17 and DECAYS to 0.8676 by e23, so a 24-epoch one_cycle spends
# six epochs past where its own anneal pays. Two questions in one arm: whether a completed 18-epoch
# anneal beats an interrupted 24-epoch one, and whether the 0.0027 a --fold -1 refit gives away to
# last-epoch selection (fitting.py:283-292 — with no holdout the rank key is the epoch INDEX, so the
# kept checkpoint is always the last) can be returned to the shipping line. 0.0027 is the CEILING on
# the second half alone; the anneal shape changes with the budget, so this is not the 24-epoch run
# stopped at 17.
#
# WHY THIS MEMBER AND NOT THE FAMILY. The penalty is member-specific: labelwise_ext_drop40_long and
# labelwise_dense_drop40_long peak exactly at e23, labelwise_depth24_drop40_long at e18 (-0.0006).
# Only headlr_ship decays materially, which is why no earlier refit exposed it.
#
# deliberate: --vram-gb 22 — owner VRAM cap 50-70%; run_comparison marks it NOT_AN_AXIS, so the
#             comparison against the control's 28 stays single-knob.
# deliberate: holdout fit, NOT --fold -1 — the point is a ledger75 number read against 0.9059
#             without spending a board slot.
# CARD. 18 epochs at the control's measured 221 s, ~1.1 h. Nothing else of ours on the card.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train headlr_peak18 \
  $RECIPE --epochs 18 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 24 --batch 10 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --head-lr 2e-4 \
  --labels "$LABELS" --out "$MODELS/headlr_peak18_s2026.pt"
