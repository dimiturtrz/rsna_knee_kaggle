#!/usr/bin/env bash
# dropstack_dense.sh -- stochastic depth, ramped linearly over the trunk's blocks, at the dense corner.
#
# bead: knee-b9hnw
# reading: dropstack_dense_s2026.holdout.npz
#
# WHY THIS AND WHY NOW. A flag-coverage diff over `options.py` against every arm on disk returned EMPTY:
# every option flag this project owns has already been passed by some arm. So the remaining lever is not a
# knob combination, it is a MECHANISM the trainer does not have. This is the second of two on the same axis
# (generalisation at fixed data and fixed architecture); `sam_dense` is the first, and this one is a third
# of its price.
#
# THE MECHANISM. Each residual block's branch is zeroed for a WHOLE SAMPLE with probability p, survivors
# divided by (1-p) so the block is identity in expectation. That is an implicit ensemble over 2^depth
# sub-networks sharing weights, and it is the regulariser that carried the published ViT fine-tuning
# recipes -- the one thing those recipes have that this trunk has never run.
#
# WHY A MODULE AND NOT `from_pretrained(drop_path_rate=...)`, WHICH IS ONE LINE. Two reasons, both measured
# rather than assumed. First, `transformers` builds `Dinov2DropPath(config.drop_path_rate)` at EVERY layer
# with the SAME rate; the recipe whose effect sizes are published RAMPS the rate linearly from zero at the
# first block. The early blocks carry the features every later block reads, and dropping them at the full
# rate is exactly why a flat schedule underperforms its own ramped control. Second, `ENCODERS` is a frozen
# module-level registry shared with the kernel, so a per-run number cannot live in it without putting a
# training concern into the seam that serving also crosses.
#
# EINSTEIN GATE, RUN BEFORE THIS WAS BUILT. `TimmTrunk.drop_path` defaults to 0.0 and no flag has ever
# reached it. `foreign_encoders.py:297` does pass `drop_path=0.1`, but that is the FOREIGN raptor encoder,
# never our trunk. No submission and no ledger row in this project has ever been measured with stochastic
# depth on. This is not a repeat.
#
# ONE VARIABLE. Byte-identical to `calib_dense_hold` (ledger 0.8634 @holdout909 midpoint) except for
# `--stochastic-depth 0.1`. Same recipe, epochs, backbone rate, slices, band, calibrated targets, held
# annotations, label table and seed. The control is measured and is not re-run.
#
# PRE-REGISTERED BANDS, against calib_dense_hold 0.8634:
#   >= 0.8684  carries. The escalation is a RATE sweep (0.05 / 0.2) and then the same rate at the span
#              corner -- not a seed and not a fold.
#   0.8604-0.8683  directional only. Free and mechanism-sound, so the member is kept as a diversity seat,
#              but nothing is planned around it and no confirmation seed is spent.
#   < 0.8604  the rate is too high for a 12-epoch fine-tune of a pretrained trunk (the published schedules
#              run hundreds of epochs from scratch). That REFUTES this rate at this budget, NOT the
#              mechanism -- the honest follow-up is 0.05, once, or nothing.
#
# AND THE REFUTATION THE MEAN CAN HIDE: a gain concentrated on one or two labels is not this mechanism.
# Stochastic depth acts on the trunk, which every label reads through, so it must move the row broadly.
# Read the PER-LABEL row before the mean.
#
# HARNESS SUSPECT, NAMED IN ADVANCE AND ANSWERABLE. `StochasticDepth.applied` walks by ATTRIBUTE NAME, so a
# trunk family spelling its residual policy differently receives nothing -- and an arm whose rate reached
# zero blocks is an ordinary run reporting under this arm's name, with an identical checkpoint, an
# identical score and an identical ledger row. That is why the walk RAISES on zero blocks rather than
# warning, and why the count is stamped into the recipe as `drop_path_blocks`. Assert it from the LOG
# before any verdict: the printed recipe must carry "drop_path_rate": 0.1 AND a nonzero "drop_path_blocks"
# (12 for dinov2-small).
#
# PRICE: ~56 min (279 s/epoch MEASURED at 20 slices x 12 epochs), no submission slot, and ZERO extra
#        compute over the control -- no second forward, no parameter, no buffer. A third of sam_dense.
# EXPECTED GAIN: P(clears 0.005) ~ 0.25, P(clears 1.5%) ~ 0.06. The published effect sizes are from
#        long from-scratch schedules; a 12-epoch fine-tune is the regime where this is smallest.
# VRAM: unchanged from the control. Dropping a branch removes activations, never adds them.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train dropstack_dense \
  $RECIPE --epochs 12 --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --stochastic-depth 0.1 \
  --labels "$LABELS" --out "$MODELS/dropstack_dense_s2026.pt"
