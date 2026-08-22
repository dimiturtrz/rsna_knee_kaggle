#!/usr/bin/env bash
# vjepa_member's second half, warm-started from its own epoch-5 encoder.
#
# WHY THIS EXISTS RATHER THAN A THIRD FULL RUN. vjepa_member died twice with exit 127 and no traceback
# (knee-sx3a) — once at epoch 6 of 12, once at step 121 of epoch 0. Both deaths are unexplained and
# neither is a verdict on the trunk. What they did NOT destroy is the work: `save_partial` writes every
# epoch, so `vjepa_member_s2026.partial` holds epoch 5 at holdout 0.8313, six epochs of encoder
# adaptation already paid for. `FittedEncoder` (knee-3fo9) converts that partial into the file
# `--encoder-weights` reads, and the conversion is proven rather than assumed: loaded into a stock
# vjepa2l it changes 96 of 587 tensors — exactly the six opened blocks at sixteen tensors each, with the
# frozen remainder bit-identical. So this arm resumes an adaptation instead of repeating one.
#
# WHAT IS AND IS NOT MATCHED. Every flag below is vjepa_member's, unchanged, except the two that make
# this the second half: `--encoder-weights` and `--epochs 6`. What does NOT carry over is the head
# (fitted for these targets, cheap to relearn) and the OneCycle schedule, which restarts. So this is not
# a resume in the optimiser's sense — it is a fresh six-epoch fit on an encoder that has already seen six
# epochs. Stated plainly because it cuts the generous way: total encoder adaptation is 12 epochs, same as
# the original schedule, but taken as two cycles rather than one.
#
# THE BAR IS UNCHANGED, and deliberately so. vjepa_member pre-registered 0.845 as "the depth axis carries
# something the frozen probe could not see", 0.83-0.845 as the tie that retires volume pretraining at
# this data scale, and below 0.82 as a reason to suspect the harness first. A member is priced by its
# holdout mean whatever path produced it, so those three bands read here as written. The one asymmetry
# to hold in mind: a schedule restart on an already-adapted trunk is if anything favourable, so a MISS
# here is the stronger of the two verdicts and a PASS the weaker.
#
# cost  ~1.6 h for 6 epochs at ~16 min an epoch, against ~3.2 h for a fresh twelve. Halving the price of
#       the read is the whole reason the bridge was built before this was relaunched.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
# bead: knee-rpkw
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train vjepa_warm \
  $RECIPE --epochs 6 --variant vjepa2l --pool mean --slice-pooling channels \
  --whole-stack $THROUGHPUT --slices 20 --band-low 0.1 --band-high 0.9 --train-size 256 --batch 6 \
  --backbone-lr 4e-5 \
  --encoder-weights "$PROCESSED/vjepa_member_e5.encoder.pt" \
  --labels "$LABELS" --out "$PROCESSED/vjepa_warm_s2026.pt"
