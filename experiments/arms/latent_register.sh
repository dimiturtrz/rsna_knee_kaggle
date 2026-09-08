#!/usr/bin/env bash
# latent_register.sh — the same masked objective, the same mask, the same rate, on a trunk that HAS REGISTERS.
#
# reading: latent_register_dense_s2026.holdout.npz
# deliberate: variant v3small (4 register tokens) at mask 0.30 lr 1e-4 latent — a rung that REFUSED on dinov2-small
# bead: knee-lm37v
#
# THE OBSERVATION. Epoch 1 of the mask ladder put 64% of every patch query's attention mass on the PREFIX at
# block 11 (entropy 2.88 against a stock ~5.4, prefix mass 0.16 -> 0.64). That is an attention sink, and it is
# the pathology Darcet et al. named in *Vision Transformers Need Registers* (ICLR 2024): a ViT with nowhere to
# park global information grows high-norm artefact tokens that swallow the routing.
#
# THE THING NOBODY HAD CHECKED. `pretrain.py` takes `--variant` and defaults to SMALL, and SMALL resolves to
# `facebook/dinov2-small`: patch 14, family dinov2, **prefix of one class token and no registers at all**.
# Every run in this family — pixel, spectral, latent-EMA, all three mask rungs — adapted that trunk. So the
# sink had exactly one token to form on, and it formed there. Meanwhile `v3small` has sat in ENCODERS the
# whole time carrying `num_register_tokens: 4`, and D256 measured the one PUBLISHED clean continuation of this
# kind (OrthoFoundation, 1.25M knee images) on dinov3-vitl16 — a register model — holding its routing to a max
# |delta| of 0.423 nats. The published fix for our exact failure shape is a checkpoint we already own and had
# never pointed the objective at.
#
# WHY MASK 0.30 AND NOT 0.15. Because 0.30 is the rung that REFUSED on dinov2-small at epoch 0, at b10 +1.53 /
# keep 0.24 against a floor of 0.25. Holding objective, mask, rate, schedule and seed fixed and changing only
# the trunk makes this an A/B with a known negative arm rather than a fresh exploration.
#
# THE CONFOUND, STATED BEFORE THE RESULT. dinov2-small and dinov3-vits16 differ in more than registers: patch
# 16 against 14, LVD-1689M against LVD-142M, and a different distillation recipe. A survival here does NOT
# isolate registers as the cause — it isolates the TRUNK. Isolating registers would need a v3 trunk with its
# registers ablated, which is a second arm and only worth writing if this one survives.
#
# COST. Patch 16 at 336 px is 21x21 = 441 patches + 1 cls + 4 registers = 446 keys against dinov2's 577, so an
# epoch should come in under the mask ladder's 431 s. The gate returns its verdict at the END OF EPOCH 0, so
# the refusal branch costs ~7 minutes and the survival branch spends the remaining seven epochs earning it.
#
# PRE-REGISTERED:
#   1. SURVIVES EPOCH 0 at mask 0.30. Then the trunk is the variable and the objective was never the problem;
#      the whole masked-adaptation family re-opens on v3-class trunks and this arm carries it to the fine-tune.
#      Watch the prefix mass specifically: if the sink is present but PARKED ON THE REGISTERS, that is the
#      Darcet mechanism working as designed and the gate should be taught to say so.
#   2. REFUSES WITH THE SAME SHAPE (b10 drifting toward uniform, keep ~0.24). Then registers and trunk are out,
#      three knobs and the trunk are all retired, and what is left is the objective itself against our data.
#   3. REFUSES WITH A DIFFERENT SHAPE. Report it and do not generalise — a new failure on a new trunk is one
#      observation, not an axis.
#
# THE FINE-TUNE CONTROL IS NOT band_dense. That 0.8578 is a dinov2-small number and cannot price a v3small
# encoder. The comparison this arm owes is a v3small run at the identical supervised recipe with the flag off;
# find it in the ledger before quoting anything, and if it does not exist the fine-tune here reads against
# nothing and must say so.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

TAG="adapt_latent_v3small_m30_e8"
OUT="$MODELS/$TAG.pt"

run_pretrain "$TAG" \
  --variant v3small --objective latent --epochs 8 --slices-per-step 96 --lr 1e-4 --seed 2026 \
  --mask-ratio 0.30 --out "$OUT"

require "$OUT"

run_train latent_register_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant v3small --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --encoder-weights "$OUT" \
  --labels "$LABELS" --out "$MODELS/latent_register_dense_s2026.pt"
