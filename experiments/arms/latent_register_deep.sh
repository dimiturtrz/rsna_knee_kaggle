#!/usr/bin/env bash
# latent_register_deep.sh — the mask ratio the register trunk was never actually refused at.
#
# reading: latent_register_deep_s2026.holdout.npz
# deliberate: v3small latent at mask 0.60, against latent_register_dense's 0.30, everything else held
# bead: knee-jwf2c
#
# WHY THIS RUNG EXISTS. 0.30 was never chosen; it was inherited from the dinov2 ladder, where it was the rung
# that refused. On v3small it did not refuse — it came within 0.07 of the floor, and that near-miss is what
# kept the ratio where it was. Read off the weights on 2026-09-08 (`kneemri.training.encoder_profile`, built
# because the arm's own code froze at launch and could not print the split), block 11 holds 0.88 of its mass
# on the REGISTERS and 0.00 on the class token, with the patch distribution behind the sink still spread at
# 4.46 nats. The block was never close to losing its discrimination. The gate was reading how much mass had
# left the patches and calling that a collapse, which on a register trunk are unrelated quantities.
#
# The ruler was replaced (commit 2d440c19, `AttentionHealth.axis`) and re-validated against every checkpoint
# the old one refused — a total class-token sink, a single-patch lock, and a walk to uniform — all three of
# which it still refuses. So the cap on this axis was a measurement artefact, and this arm spends the room.
#
# WHY 0.60 AND NOT 0.45. MAE trains at 0.75 and the latent-MIM family sits high for a mechanical reason: a
# light mask lets a patch be reconstructed from its own neighbours, so the objective never has to build the
# long-range structure that is the whole point of adapting. 0.45 would be a half-step justified by nothing
# but caution about a bound that has just been shown not to exist. If 0.60 refuses, the refusal is now
# informative — it would be the first real reading of this trunk's limit rather than a re-reading of ours.
#
# COST AND WHAT IT HAS TO BEAT. ~40 min of pretrain (8 epochs at ~290 s, 446 keys) plus the 12-epoch dense
# fine-tune, ~2.7 h total. It is priced against latent_register_dense's holdout and nothing else: identical
# trunk, recipe, rate, seed, slices and band, one flag apart. A knob-sized win is directional and stays a
# knob; what would make this worth the card is the mask ratio turning out to be a LADDER on register trunks
# where it was a cliff on dinov2, because that re-opens the whole masked-adaptation family with room in it.
#
# PRE-REGISTERED, before the run:
#   1. Survives, and the fine-tune beats latent_register_dense. The ratio is a live axis on this trunk; the
#      next arm is 0.75, not another trunk.
#   2. Survives, and the fine-tune is flat or worse. Then adaptation strength is not what bounds this member
#      and the axis closes at 0.30 — a real closure, since the gate is no longer the thing doing the bounding.
#   3. Refuses at epoch 0 on the corrected ruler. Then 0.30-0.60 holds the trunk's actual limit, the reading
#      costs seven minutes, and the fine-tune below never starts.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

TAG="adapt_latent_v3small_m60_e8"
OUT="$MODELS/$TAG.pt"

run_pretrain "$TAG" \
  --variant v3small --objective latent --epochs 8 --slices-per-step 96 --lr 1e-4 --seed 2026 \
  --mask-ratio 0.60 --out "$OUT"

require "$OUT"

run_train latent_register_deep \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant v3small --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --encoder-weights "$OUT" \
  --labels "$LABELS" --out "$MODELS/latent_register_deep_s2026.pt"
