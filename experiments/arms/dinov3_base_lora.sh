#!/usr/bin/env bash
# dinov3_base_lora.sh — DINOv3 at BASE, adapted rather than fine-tuned.
#
# EVERY v3 NUMBER WE HOLD WAS TAKEN ON THE WRONG VARIANT UNDER THE WRONG RECIPE, which is a poor basis
# for the verdict currently written down. Checked rather than assumed:
#     DINOv2-S/14   image_size 518  patch 14  hidden 384  layers 12  registers 0
#     DINOv3-S/16   image_size 224  patch 16  hidden 384  layers 12  registers 4
# SAME WIDTH, SAME DEPTH. v3-S/16 is not a larger network than v2-S/14 - it is the same size distilled
# from a 7B teacher with a coarser patch, so it sees 23% fewer tokens at equal pixels. The published
# DINOv3 advance sits in the 7B teacher and the LARGE distilled students, making S/16 the variant where
# an upgrade is least expected.
#
# AND THE RECIPE COST IT 0.0160, measured: v3-S at 224 reaches 0.8217 under LoRA against 0.8057
# fine-tuned. So the family was judged on its weakest member while being actively damaged.
#
# S+ DOES NOT ANSWER THIS. It scored 0.8012 against v3-S's 0.8080 and I used that to bury the base
# hypothesis (knee-d2s) - wrongly, because S+ is a WIDER SMALL, not the next rung.
#
# 224 px, v3's own native grid, where the LoRA result was measured. Base is 768-wide against small's
# 384, so the cap stays at 26 and the batch is unchanged.
#
# control  dinov3_v3small_224 under LoRA = 0.8217, and DINOv2's 0.8457 is the bar that matters.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

HF_TOKEN="$(grep -oE 'hf_[A-Za-z0-9_-]+' /d/life/creds/machine/hfpat.txt | head -1)"
export HF_TOKEN
[ -n "$HF_TOKEN" ] || { echo "MISSING: no Hugging Face token, and DINOv3 is a gated repository"; exit 1; }

run_train "dinov3_base_lora" \
  --fold 0 --epochs 12 --one-cycle --weight-floor --rank-weight 1.0 --patience 3 \
  --slice-pooling max --whole-stack --lora-rank 8 --backbone-lr 5e-4 \
  --variant v3base --image-size 336 --train-size 224 \
  --recompute-activations --compile-encoder --bf16 --vram-gb 26 \
  --labels "$LABELS" --out "$PROCESSED/dinov3_base_lora_s2026.pt"
