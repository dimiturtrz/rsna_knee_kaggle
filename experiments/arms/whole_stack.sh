#!/usr/bin/env bash
# Read the whole cached stack per step instead of one three-slice window.
#
# A training step currently sees 6 slots x 3 consecutive slices = 18 images, out of 72 cached and
# ~165-190 acquired. A finding outside the sampled window is invisible to that step. Whole-stack
# pooling removes the sampled-group limitation without changing what a group SPANS - which is the
# confound that killed the slice-count arm - and it makes training match a deployment that already
# averages ten overlapping windows.
#
# Two measurements say the stack holds more than one window shows: overlap TTA is worth +0.0156 at
# inference, and the occlusion depth profile ranges 0.7976 to 0.8332 across windows.
#
# PRIOR ART, and the reason this arm outranks the resolution one: MRNet (Stanford, PLOS Medicine 2018)
# combines per-slice features by an element-wise max over ALL slices of a series. Maxing over the whole
# stack is the canonical operation in this literature; our three-slice window is the deviation.
#
# control  compose_small / arm Q (max pooling, WINDOWED). Single variable: the read.
#
# EPOCH BUDGET, measured 2026-08-11 rather than inherited. The 24-epoch version peaked at 0.8413
# around epoch 6 and then gave it back - 0.8298 by epoch 10 with the loss still falling. Showing the
# model 4x the imaging per step means it converges in roughly a third of the epochs, so 24 was not
# merely 80 wasted minutes: one-cycle is defined over total_steps, so a budget the model does not need
# stretches the anneal across it. 12 epochs, with the schedule defined over 12.
#
# VRAM BUDGET, measured 2026-08-11 rather than guessed. A step puts 12 studies x 6 slots x 12 slices =
# 864 images through the encoder. Uncapped and without checkpointing it took all 32 GB and ran at
# 145 W of 575 with NO epoch in fifteen minutes: Windows pages GPU memory over PCIe instead of raising
# OOM, so a run that does not fit crawls while reporting 100% utilisation. With checkpointing and an
# 18 GB cap it holds 19.2 GB, 99% utilisation and 476-514 W - roughly 3.4x the throughput in 40% less
# memory, because the cap is what keeps it inside real VRAM. It also leaves ~12 GB for the other
# project on this card. Neither flag is optional here.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train whole_stack \
  --fold 0 --epochs 12 --one-cycle --weight-floor --rank-weight 1.0 \
  --slice-pooling max --whole-stack --recompute-activations --vram-gb 20 \
  --labels "$LABELS" --out "$PROCESSED/wholestack_s2026.pt"
