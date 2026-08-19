#!/usr/bin/env bash
# unfreeze_depth.sh [blocks] — how many of the encoder's twelve blocks should adapt?
#
# THE SPEED IS LARGE AND MEASURED: one epoch at four blocks ran 1075 img/s against six blocks' 817,
# +32%, eager both ways. That stacks with compile's +48% toward roughly twice the experiments per hour,
# which is worth a full arm to settle. This is a THROUGHPUT arm, not a score knob: knee-jk3 says stop
# running levers inside the decode family, and this one is not in it — it buys wall-clock, and the score
# is the price being checked rather than the thing being bought.
#
# ALREADY RUN AND REFUTED — 2026-08-19, `unfreeze4_s2026.pt`, ledger row `unfreeze_4`: 0.8320, which is
# -0.0137 against wsfold0_e6, 2.2x the noise floor, converged (peaked at epoch 6 and flat for five after)
# and negative on eleven of twelve labels. This arm is NOT queued and must not be re-armed casually: it
# would overwrite that checkpoint and re-buy a refutation the ledger already holds. The re-basing below
# was written on 2026-08-20 in the belief that the arm had never run — a name check on THIS FILE's name
# returned nothing because the ledger indexes the run by its OUTPUT name, `unfreeze`. The re-based
# control is still the right control; what was wrong was the premise that anything needed running.
#
# The one thing the refutation does NOT cover: it was measured at the stale operating point (the default
# rate, band 0.20-0.80). Shallow adaptation could in principle behave differently at 4e-5 / 0.10-0.90.
# That is a QUESTION, not a plan — 2.2x the floor with eleven of twelve labels negative is a decisive
# kill, and knee-jk3 says stop running knobs. Re-run only if something else makes encoder depth live.
#
# THE SCORE IS NOT SETTLED, and the throughput arm cannot settle it. Its epoch-0 holdout was 0.7539
# against 0.7600 - a 0.0061 gap sitting exactly ON the 0.0062 noise floor, read from the single weakest
# measurement in the ledger. A shallower adaptation could catch up (the head has more epochs to use a
# fixed representation) or fall further behind (fewer blocks to specialise to MRI, which is a long way
# from the natural images the encoder was trained on). Both are plausible; one epoch distinguishes them
# not at all.
#
# WHY IT MIGHT COST NOTHING: our own ledger says capacity is not the constraint here — base over small
# was +0.0011 against the same floor — and unfreezing fewer blocks is a capacity reduction of the same
# kind. WHY IT MIGHT COST SOMETHING: the frozen blocks are the GENERIC ones, and knee MRI is far enough
# from ImageNet that the specialisation may need to reach deeper than four blocks.
#
# CONTROL, RE-BASED 2026-08-20: band_dense = 0.8578, the SAME recipe at the settled operating point
# (4e-5, 20 slices, band 0.10-0.90, 12 epochs) with the default six blocks unfrozen. The comment here
# used to name wsfold0_e6 = 0.8457, taken at the default rate and the 0.20-0.80 band. calib_span had
# just been written up for that exact mistake — a lever measured +0.0038 at a stale operating point and
# NEGATIVE on the recipe we ship — and knee-aixw holds the general form of it. Measuring a speed lever
# against a control from before the rate and span settled would have bought the same retraction twice.
#
# PREDICTION. Throughput rises by about 30%, reproducing the ratio the epoch-0 measurement showed at a
# different operating point. Holdout mean falls by LESS than the 0.0062 floor, because capacity is not
# what limits this model. If the mean does fall, the loss concentrates in FOCAL labels (ACL, the two
# menisci, the three OA rows) rather than REGIONAL ones, because fine-scale discrimination is what the
# deeper blocks would be specialising and coarse context is what the frozen generic ones already carry.
#
# REFUTATION. Two conditions, either one of which retires shallow adaptation as a speed lever:
#   - throughput does NOT rise by at least 20%, in which case the arm's whole premise is void and the
#     score is irrelevant — there is nothing being bought to pay for;
#   - the mean falls MORE than 0.0062 below band_dense, in which case four blocks cannot carry knee MRI
#     and the 32% is not purchasable at any price this project is willing to pay.
# A mean inside the floor WITH the throughput confirmed makes four blocks the new default for every arm
# whose question is not about encoder depth, which is what would make this arm worth its own slot.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
BLOCKS="${1:-4}"
require "$LABELS"

run_train "unfreeze_${BLOCKS}" \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --unfreeze-last "$BLOCKS" \
  --labels "$LABELS" --out "$PROCESSED/unfreeze${BLOCKS}_s2026.pt"
