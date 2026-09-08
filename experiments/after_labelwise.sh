#!/usr/bin/env bash
# Wait for labelwise_extended to release the card, then re-run the in-domain MAE with the loss terms
# in the same units (knee-c3qfe).
#
# WHY THIS IS NOT A REPEATED EXPERIMENT. `pretrain_matched_e8` ran this exact configuration and its
# pixel term never left the value a decoder scores for predicting the patch mean (0.9859 -> 0.9682
# against per-patch standardised targets, where 1.0 IS that trivial solution). The cause is in
# `SpectralMae.forward`: the decoder's standardised output was merged with RAW visible patches before
# the spectral term read the image, so the two terms wanted the same tensor in two different units and
# the spectral term — a log1p magnitude MSE over the whole 336x336 plane, priced at every mask boundary
# — carried the larger gradient and won. One variable moves here, `restore`, which is the inverse of the
# standardisation `pixel_loss` already applies. Mask ratio, high-pass radius, spectral weight, rate,
# schedule and corpus are byte-identical to that run, which is what makes the pixel curve a reading.
#
# THE MECHANISM CHECK IS PRE-REGISTERED AT EPOCH 2 AND IT IS MINE TO MAKE, not this script's: pixel must
# fall below 0.90. That is far outside anything the broken objective produced (0.9704 after one epoch,
# 0.9682 after eight) and far above where a fitting masked autoencoder settles. If it is still pinned
# near 1.0, kill the run there — the fix is not the whole story and the next suspects are the single
# nn.Linear(384, 196) decoder and a 0.6 mask ratio on an encoder that reads the masked positions. This
# script therefore ends at the pretrain; `encoder_drift` and `arms/mae_unitfix.sh` come after the branch.
#
# THE GATE IS THE MARKER IN THE CHECKPOINT'S OWN LOG. `_arm_log` names the file for `--out` minus `.pt`
# (labelwise_extended_s2026.full.log, under $MODELS/experiments and NOT under logs/), while `_mark`
# writes the run_train NAME (ARM DONE labelwise_extended) — two different variables, and gating on the
# wrong spelling holds the card idle while queue_check reads OK.
set -u
cd "$(dirname "$0")/.."

MODELS="D:/models/rsna_knee_abnormality"
GATE="$MODELS/experiments/labelwise_extended_s2026.full.log"
OUT="$MODELS/adapt_small_e8_unitfix.pt"

echo "waiting for 'ARM DONE labelwise_extended' in $GATE at $(date -u +%FT%TZ)"
until grep -q "ARM DONE labelwise_extended" "$GATE" 2>/dev/null; do sleep 60; done
echo "card released at $(date -u +%FT%TZ)"

.venv-arm/Scripts/python.exe -m kneemri.training.pretrain \
  --epochs 8 --slices-per-step 96 --lr 1e-4 --seed 2026 \
  --out "$OUT" > logs/pretrain_unitfix.log 2>&1
echo "QUEUE pretrain_unitfix exited $? $(date -u +%FT%TZ)"
