#!/usr/bin/env bash
# mixup224_strong.sh — mixup224's recipe at `--mixup 0.4`. The SECOND point of the dose pair; it has no
# independent thesis and must never be read alone.
#
# bead: knee-x580
#
# WHY 0.4 AND NOT 1.0. Beta(a, a) at a = 1.0 draws the weight uniformly and blends half the batch near
# 50/50; on 1,763 training studies that trades away more signal than it buys, and it would not be a
# second point on a curve, it would be a different regime. 0.1-0.4 is the customary range for a
# classifier this size; 0.2 and 0.4 are the two ends of it, which is the widest separation available
# inside the regime the mechanism is argued for. The draw is folded to the dominant half, so every row
# stays recognisably its own study at both concentrations.
#
# WHAT THIS POINT BUYS THAT A RE-RUN OF 0.2 WOULD NOT. A second seed at 0.2 measures the noise floor,
# which is already measured at 0.0062 and does not need re-measuring. A second CONCENTRATION measures
# whether the knob has a direction. Those are different questions, and only the second one can turn two
# sub-floor numbers into evidence: noise has no reason to respect the knob, so an ordering that follows
# the dose is signal the individual points do not carry.
#
# HOW IT CAN FAIL AND STILL BE WORTH THE NINETEEN MINUTES. If 0.4 lands clearly BELOW 0.2 and below the
# control, the family is not flat — it is peaked somewhere at or under 0.2, which names 0.1 as the next
# point rather than closing the axis. If both land under the control, the axis closes tonight on two
# points instead of hanging on one, and the remaining suspects move to supervision volume (knee-ombx)
# and representation (knee-jk3). Either reading retires a suspicion, which is what an instrument arm is
# for; neither is sold here as an improvement.
#
# control  native224 = 0.8502 — the SAME control as mixup224, which is the whole point of the pair.
#          Twelve epochs, 12 slices, 0.20-0.80 band, whole-stack, 4e-5, 224 px, same labels, same
#          909-study holdout.
# cost     ~19 min, 8124 MiB. Runs after mixup224 rather than beside it: two 8 GB arms fit the card on
#          memory, but they would share SMs and neither epoch time would be a fact about its recipe.
#
# HARNESS CHECK. The printed recipe must say "image_size": "224" and "mixup": 0.4 — the 0.4 in
# particular, because this arm and mixup224 differ in exactly one character and a copy-paste that left
# 0.2 in place would produce a plausible, meaningless second point.
#
# PREDICTION. Read the pair's clauses in mixup224.sh; this arm adds only the ordering requirement.
#
# deliberate: --mixup 0.4 (the pair's second concentration)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train mixup224_strong \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --mixup 0.4 \
  --labels "$LABELS" --out "$PROCESSED/mixup224_strong_s2026.pt"
