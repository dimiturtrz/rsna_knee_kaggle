#!/usr/bin/env bash
# peak_rate.sh [peak] — the peak that DELIVERS what the old constant delivered.
#
# MEASURED 2026-08-13, summing the learning rate over every step at our settings (pct_start 0.15,
# 12 epochs x 291 steps):
#     constant 8e-6           total 0.02794   mean 8.0e-6
#     OneCycle at peak 8e-6   total 0.01405   mean 4.0e-6
# OneCycle delivers 50.3% of the movement. So moving from a constant 8e-6 to OneCycle(max_lr=8e-6)
# HALVED the total backbone learning while calling it the same rate, and every arm quoted by its peak
# inherits that - the "5x rate" arm at 4e-5 delivers a mean 2e-5, which is 2.5x the old constant.
#
# THE ARM: peak 1.6e-5, which integrates to the constant 8e-6 that the pre-OneCycle recipe used. If the
# control has been quietly under-training its encoder since OneCycle landed, this is where that shows.
#
# WHY IT IS NOT A RATE SWEEP, which is banned here as self-tuning: the value is DERIVED, not searched.
# One number, from an integral, with a prediction attached.
#
# THE FALSIFIABLE PART: if the control is under-trained, this gains and the gain is largest where the
# encoder matters most - the focal findings the DINOv3 work showed are carried by fine local structure
# (menisci, Baker's, contusion) rather than the joint-wide ones (effusion, synovitis, OA). If instead
# 8e-6 was already the right PEAK and the halving was a feature, this loses, and the interpretation is
# that the anneal - not the integral - is what OneCycle sells.
#
# control  wsfold0_e6 = 0.8457, identical recipe at peak 8e-6.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
PEAK="${1:-1.6e-5}"
require "$LABELS"

run_train "peak_rate_${PEAK}" \
  $RECIPE --epochs 12 $PATIENCE \
  --slice-pooling max --whole-stack --backbone-lr "$PEAK" \
  $THROUGHPUT \
  --labels "$LABELS" --out "$MODELS/peak_rate_${PEAK}_s2026.pt"
