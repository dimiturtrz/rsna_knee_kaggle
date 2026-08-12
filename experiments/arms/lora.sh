#!/usr/bin/env bash
# lora.sh [rank] — adapt every block along a rank-r subspace, with the pretrained weights FROZEN.
#
# THE QUESTION. Our recipe rewrites the last six blocks of an encoder trained on 1.7B images using
# gradients from 3,498 studies labelled out of radiology reports. Whatever those blocks learn about
# knees, they also forget some of what made the encoder worth starting from — and our holdout cannot
# see it, because the same fine-tuned model produces the score. LoRA removes the trade rather than
# balancing it: the base weight stays bit-identical and the entire update lives in an added rank-r
# term, so what a small noisy dataset can do to that representation is bounded by the rank.
#
# MEASURED BEFORE RUNNING, on both encoders: 72 layers adapted, the model is BIT-IDENTICAL to the
# pretrained one at initialisation (up is zero-init), 1.44M trainable parameters against DINOv2-small's
# 22.7M (6.4%), and a saved member rebuilds from its recorded rank and reloads exactly.
#
# TRAINABLE PARAMETERS ARE 7x FEWER THAN THE CONTROL'S, which is the honest risk: our control opens
# 10.9M parameters over six blocks. If the task needs that much movement, rank 8 will underfit and the
# curve will sit below the control at every epoch — a clean refutation rather than an ambiguous one.
# The opposite outcome is the interesting one: matching the control while touching nothing pretrained
# means the fine-tuning was spending capacity to recover what it had just destroyed.
#
# THE LEARNING RATE IS 20x THE CONTROL'S. Adapters start at zero and are the only thing moving, so the
# 8e-6 that suits a pretrained block is far too small for a matrix that begins as nothing; 2e-4 is the
# standard starting point for LoRA and is what this arm uses. That is a REAL confound with the control
# and it cannot be avoided — a frozen-base run and a fine-tuned run cannot share a backbone rate.
#
# control  wsfold0_e6 = 0.8457, same recipe, six blocks fine-tuned at 8e-6.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
RANK="${1:-8}"
require "$LABELS"

run_train "lora_r${RANK}" \
  --fold 0 --epochs 12 --one-cycle --weight-floor --rank-weight 1.0 \
  --slice-pooling max --whole-stack --lora-rank "$RANK" --backbone-lr 2e-4 \
  --recompute-activations --compile-encoder --bf16 --vram-gb 26 \
  --labels "$LABELS" --out "$PROCESSED/lora_r${RANK}_s2026.pt"
