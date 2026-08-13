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
# THE RATE IS 5e-4, WHICH IS A PEAK AND NOT A CONSTANT. OneCycle warms up to it over the first 15% of
# steps and anneals to near zero, so the number chosen is the top of the curve, not the rate the run
# spends its time at.
#
# 60x THE CONTROL'S 8e-6, and that is the right order of magnitude rather than an aggressive one: the
# adapters are zero-initialised, so they are new parameters being learned from scratch, not pretrained
# weights being nudged. A fine-tuning peak would leave them approximately where they started. It is
# also a REAL confound against the control that cannot be removed — a frozen-base run and a fine-tuned
# run cannot share a backbone rate, because the parameter being rated is not the same parameter.
#
# NOT SWEPT, deliberately. Self-tuning a learning rate spends card time on our own hyperparameters
# instead of on the data, so this starts from a value the literature agrees on and moves on.
#
# control  wsfold0_e6 = 0.8457, same recipe, six blocks fine-tuned at 8e-6.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
RANK="${1:-8}"

# THE ENCODER IS AN ARGUMENT, and on DINOv3 this stops being the same experiment. Every v3 run so far
# inherited DINOv2's adaptation recipe - six blocks rewritten at 8e-6 - and the interpretation committed
# on 2026-08-13 ruled out patch grid (441 patches against 196 bought 0.0023) and capacity (v3-S+ scored
# 0.0045 WORSE than v3-S) while leaving ADAPTATION untested. This is that test.
#
# WHY v3 SPECIFICALLY MIGHT BE RECIPE-DAMAGED WHERE v2 IS NOT: v3 carries LayerScale, a per-channel
# learned gain initialised small, and places position with RoPE. A backbone rate chosen for a network
# with neither is not obviously the right step for one with both, and an oversized step on those gammas
# would degrade exactly the fine local structure the per-label table says v3 lost - Baker's -0.115,
# lateral meniscus -0.077, against effusion -0.014. LoRA freezes all of it: gammas, RoPE, base weights
# bit-identical, the entire update confined to a rank-r subspace.
#
# THE OUTCOME IS READABLE EITHER WAY. Near 0.84 and the encoder was never the problem - our fine-tuning
# was destroying it, which is a finding that applies back to DINOv2. Still near 0.80 and the
# representation verdict hardens and the scope caveat in that document closes.
VARIANT="${3:-small}"
SIZE="${4:-336}"
# The cache only has to be at least as large as the read; 448 is needed only above 336.
CACHE=336
[ "$SIZE" -le 336 ] || CACHE=448

# The gated weights need a token, and only the v3 variants do. Read from the credential file into the
# environment, never echoed and never written into a log - the trainer needs it to reach the hub once.
case "$VARIANT" in
  v3*)
    HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9_-]+' /d/life/creds/machine/hfpat.txt | head -1)"
    export HF_TOKEN
    [ -n "$HF_TOKEN" ] || { echo "MISSING: no Hugging Face token, and DINOv3 is a gated repository"; exit 1; }
    ;;
esac

# IT OOM'D THE FIRST TIME, and the reason is the point of the method rather than a bug in it. Adapting
# every block means every block needs its activations kept for the backward pass; fine-tuning the top
# six keeps six. Measured beside it: unfreezing all twelve costs +29% over six at batch 2, and adapters
# add their own down/up intermediates on top of that.
#
# TWO CHANGES, neither of which touches the objective. `expandable_segments` reclaims the 2.41 GB the
# allocator was holding reserved-but-unallocated when it died — pure fragmentation, visible in the OOM
# message itself. And the cap goes to 30 GB of 31.8: the 26 GB ceiling was set to leave room for the
# sibling project, and nothing else is on the card tonight.
#
# WHAT IS NOT CHANGED: the batch. Batch 12 is what every number in the ledger was measured at, and the
# ranking term scores 66 within-batch pairs at 12 against 15 at 6 — shrinking it to fit would make this
# a different experiment that happens to also use LoRA.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RATE="${2:-5e-4}"
require "$LABELS"

# THE ENCODER AND READ SIZE ARE IN THE NAME. dinov3.sh once wrote one name whatever size it read, so
# launching a second probe began overwriting the first run's log and was about to take its checkpoint.
# A run that can destroy the run it is being compared against is not an experiment.
NAME="lora_r${RANK}_${VARIANT}_${SIZE}"

run_train "$NAME" \
  --fold 0 --epochs 12 --one-cycle --weight-floor --rank-weight 1.0 \
  --slice-pooling max --whole-stack --lora-rank "$RANK" --backbone-lr "$RATE" \
  --variant "$VARIANT" --image-size "$CACHE" --train-size "$SIZE" \
  --recompute-activations --compile-encoder --bf16 --vram-gb 30 \
  --labels "$LABELS" --out "$PROCESSED/${NAME}_s2026.pt"
