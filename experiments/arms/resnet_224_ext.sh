#!/usr/bin/env bash
# resnet_224_ext.sh — resnet_224_short's trunk under the recipe that lifted the ViT from 0.880 to 0.905.
#
# bead: knee-k01q0
# reading: resnet_224_ext_s2026.holdout.npz
#
# At 224 px on the same 909 holdout the CNN already beats the ViT: resnet_224_short 0.8842 @ledger75
# against band224 0.8801. The ViT then gained +0.025 (labelwise_ext_drop40_long 0.9045) from a head
# and label recipe the CNN never received: labelwise pooling, the extended scheme, calibrated targets,
# held annotations and 0.40 slot dropout. This arm gives the CNN that recipe and keeps its own trunk
# settings (full-trunk finetune at one rate, 224 px, 2.5D through-plane input, twelve epochs), so the
# single variable against resnet_224_short is the recipe.
#
# PRE-REGISTERED, ledger75 on the 909, against resnet_224_short 0.8842:
#   >= 0.8990  the recipe transfers; a CNN solo is worth a whole-corpus refit and an efficiency entry
#   0.8870-0.8989  partial transfer, below the ViT; read which of the five terms carried it
#   <= 0.8869  the recipe does not transfer to a CNN trunk
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train resnet_224_ext \
  $RECIPE --epochs 12 --variant resnet50d --image-size 224 $THROUGHPUT \
  --through-plane --slice-pooling labelwise --slices 20 --band-low 0.1 --band-high 0.9 \
  --whole-stack --unfreeze-last 16 --backbone-lr 2e-4 --head-lr 2e-4 --weight-decay 0.05 \
  --scheme extended --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --batch 4 --accumulate 2 \
  --labels "$LABELS" --out "$MODELS/resnet_224_ext_s2026.pt"
