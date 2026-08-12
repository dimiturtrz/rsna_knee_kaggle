#!/usr/bin/env bash
# dinov3.sh [variant] — swap the encoder for DINOv3, everything else held fixed.
#
# WHAT DINOv3 CHANGES. Same idea as v2 — a vision transformer trained with no labels, self-distilled
# from its own moving average — scaled from 142M to 1.7B pretraining images and distilled from a 7B
# teacher. The substantive addition is GRAM ANCHORING: in long training runs a DINOv2's PATCH-level
# features quietly degrade while its global feature keeps improving, and v3 anchors the patch-to-patch
# similarity structure so local detail stays sharp.
#
# WHY THAT SHOULD MATTER HERE SPECIFICALLY. Our findings are small and local — MCL is a thin ligament
# at the medial margin, a meniscal tear is 1-3 mm — and our head reads pooled PATCH tokens. Dense
# feature quality is the axis v3 improves and the axis this task lives on. Published benchmarking also
# has v3 beating medical-specific foundation models (BiomedCLIP, CT-Net) on several medical tasks,
# which is what deflates the "use a medical encoder instead" plan (knee-zkz).
#
# 336 px, NOT 392. Patch 16 tiles cleanly only at multiples of 16, and 392 is not one — 392/16 = 24.5.
# The rungs are 336 (441 tokens), 384 (576), 448 (784). This runs at 336 against the 336 control so the
# ONLY difference is the encoder; 384 is the natural follow-up and costs exactly what DINOv2 costs at
# 336, which is a free resolution rung if this lands.
#
# control  wsfold0_e6 = 0.8457 — DINOv2-small, whole-stack, max pooling, 336 px, same fold.
#
# THE HONEST PRIOR: capacity has never been the constraint here (base over small was +0.0011 on this
# ruler), and a better single model is the WEAKER lever anyway — our best single scored 0.868 on the
# board against 0.894 for a diverse ensemble. Expect a member that fails differently rather than a
# jump. That is still worth having: differently-failing members are the one thing measured to pay.
#
# SUBMITTABLE. Kaggle has no official metaresearch/dinov3 - its only DINOv3 is Keras's, in a KerasHub
# preset format `AutoModel.from_pretrained` cannot read - so the transformers snapshot is published as
# the PRIVATE dataset `dimiturnt/dinov3-hf`, one directory per variant. The DINOv3 licence permits
# making the materials available under its own terms with a copy of the Agreement alongside; the HF
# snapshot ships LICENSE.md and it was uploaded with the weights, so that holds by construction.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
VARIANT="${1:-v3small}"
SIZE="${2:-336}"

# THE ENCODER RATE IS AN ARGUMENT because the recipe was tuned on DINOv2 and need not transfer. 8e-6
# suits a block whose weights are already close to what the task wants; DINOv3 places position with
# RoPE, carries LayerScale, and was distilled from a 7B teacher rather than trained directly - nothing
# says its blocks want the same step size, and read size has already been eliminated as the difference.
RATE="${3:-8e-6}"

# READ SIZE IS THE SUSPECT, AND THE TWO ENCODERS DISAGREE ABOUT WHAT IS NATIVE.
#   DINOv2-small  config image_size 518, patch 14 -> our 336 px read is INSIDE its training range
#   DINOv3-S/16   config image_size 224, patch 16 -> our 336 px read is 1.5x BEYOND its native grid
# v3 places position with RoPE (rope_theta 100, pos_embed_rescale 2.0), which extrapolates far better
# than v2's learned embeddings would, but extrapolation is still not the same as having been trained
# there - and the high-resolution adaptation stage in the paper is applied to the 7B teacher.
#
# TWO MECHANISMS PULL OPPOSITE WAYS and only a measurement separates them:
#   (a) GRID: at 336 v2 cuts 24x24 = 576 patches and v3 cuts 21x21 = 441, so equal PIXELS hands v3 a
#       23% coarser map of anatomy whose findings are 1-3 mm. This wants a LARGER read (384 -> 24x24).
#   (b) EXTRAPOLATION: 336 is already 1.5x v3's native 224, and 384 is 1.7x. This wants a SMALLER read.
# 224 is the cheap end to test first: it is v3's own training resolution, and a quarter of the tokens
# of 336, so an epoch is a fraction of the cost of the answer it buys.
#
# The cache only has to be at least as large as the read; 448 is needed only above 336.
CACHE=336
[ "$SIZE" -le 336 ] || CACHE=448
require "$LABELS"

# The gated weights need a token. Read from the credential file into the environment, never echoed and
# never written into a log — the trainer only needs it to reach the hub once, at construction.
HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9_-]+' /d/life/creds/machine/hfpat.txt | head -1)"
export HF_TOKEN
[ -n "$HF_TOKEN" ] || { echo "MISSING: no Hugging Face token, and DINOv3 is a gated repository"; exit 1; }

# THE READ SIZE IS IN THE NAME, and it was not: this arm wrote `dinov3_v3small` whatever size it read,
# so launching the 224 probe began overwriting the 336 run's log and was about to overwrite its
# checkpoint. The 336 member survives as dinov3_v3small_336_s2026.pt only because it was copied aside
# while the 224 run was already going. A run that can destroy the run it is being compared against is
# not an experiment.
run_train "dinov3_${VARIANT}_${SIZE}_lr${RATE}" \
  --fold 0 --epochs 12 --one-cycle --weight-floor --rank-weight 1.0 \
  --slice-pooling max --whole-stack --variant "$VARIANT" \
  --image-size "$CACHE" --train-size "$SIZE" --backbone-lr "$RATE" \
  --recompute-activations --compile-encoder --bf16 --vram-gb 26 \
  --labels "$LABELS" --out "$PROCESSED/dinov3_${VARIANT}_${SIZE}_lr${RATE}_s2026.pt"
