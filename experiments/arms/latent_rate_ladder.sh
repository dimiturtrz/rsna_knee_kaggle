#!/usr/bin/env bash
# latent_rate_ladder.sh — is the LEARNING RATE what destroys routing? A ladder, 7 minutes a rung.
#
# reading: latent_rate_dense_s2026.holdout.npz
# deliberate: lr 3e-5 then 1e-5, mask 0.30, latent target, everything else byte-identical to the mask ladder
# bead: knee-lm37v
#
# WHAT THE MASK LADDER SETTLED. Three rungs, one epoch each, everything else byte-identical:
#
#   mask 0.6   b10 +1.55  keep 0.23  REFUSED
#   mask 0.30  b10 +1.53  keep 0.24  REFUSED
#   mask 0.15  b10 +1.35  keep 0.33  SURVIVED epoch 0
#
# The ratio is a dial, but a sharply NON-LINEAR one: the first halving moved the failing block 0.02 nats
# and the second moved it 0.18. So it buys margin only at the very bottom of its range, where a masked
# objective has least task left. The TARGET is separately exonerated (`latent_adapt`: pixel+spectral and
# latent-EMA die identically under identical masking, teacher spread flat at 0.3836-0.3881 while the loss
# fell 0.387 -> 0.0611), which leaves the rate as the one knob of the three nobody has walked.
#
# WHY RATE IS NEXT — AND THE ARGUMENT IS WEAKER THAN IT WAS WHEN THIS ARM WAS WRITTEN. "Rate was already
# excluded" rested on spectral@1e-5 also collapsing, but in the OPPOSITE DIRECTION: away from uniform, into an
# attention sink (0.000 / 0.035 / 0.023 nats), while every run at 1e-4 collapses TOWARD uniform at the top
# blocks. Two failure modes at two rates read as a dial nobody had walked.
#
# THE MASK LADDER'S EPOCH 1 SHOWS BOTH MODES IN ONE RUN AT ONE RATE:
#
#   epoch 0  vs stock ... +0.33 +1.35 -0.07 | keep ... 0.80 0.33 0.99    block 10 fails, TOWARD uniform
#   epoch 1  vs stock ... +0.03 +0.31 -2.53 | keep ... 0.98 0.85 0.53    block 10 heals, block 11 fails, INTO a sink
#
# Block 11's entropy is 2.88 against a stock ~5.4 and its prefix mass went 0.16 -> 0.64: two thirds of the
# attention on the prefix. The collapse did not stop at mask 0.15; it MIGRATED one block up and changed sign.
# So the two shapes are not two rate regimes. They are two faces of one thing — the last blocks lose their
# learned routing, and the direction they fall in is not the rate's to decide. This ladder is therefore no
# longer "the one knob nobody has walked"; it is a cheap 14-minute check on whether MAGNITUDE still modulates
# the size of a destruction whose SHAPE is now known to be rate-independent. Run it, but read outcome 1 below
# as the weak claim it now is.
#
# THE LADDER. The gate returns a verdict in ONE epoch, and at mask 0.30 an epoch is 431 s at 560 slices/s,
# so a rung costs ~7 minutes. It STOPS at the first rung that survives epoch 0 and spends the remaining
# seven epochs plus the 12-epoch fine-tune on that one, so the card never proves a collapse twice.
#
# PRE-REGISTERED:
#   1. DRIFT AT BLOCK 10 SCALES WITH THE RATE (3e-5 lands near +0.5, 1e-5 near +0.15). Then it is gradient
#      MAGNITUDE, the mechanism is ordinary catastrophic drift rather than anything about masking, and the
#      fix is rate plus a longer schedule — a fixable recipe, not a closed axis.
#   2. DRIFT FLAT ACROSS A 10x RATE CHANGE (both rungs near +1.5). Then rate is out too, all three knobs
#      this family owns are retired, and the suspect becomes the trunk/objective interaction itself. The
#      successor is self-distillation WITHOUT a mask — OrthoFoundation's own recipe, which D256 measured
#      keeping its routing on this exact architecture — and that is a different objective, separately argued.
#   3. A RUNG SURVIVES ALL EIGHT EPOCHS. Fine-tune it and read it against the control. >= 0.8578 opens the
#      adaptation path; flat says routing survived and bought nothing, and the cap is in the data.
#
# control  band_dense = 0.8578 (best epoch 8), byte-identical fine-tune with the flag off.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

SURVIVOR=""
for RATE in 3e-5 1e-5; do
  TAG="adapt_latent_lr${RATE/-/}_e8"
  OUT="$MODELS/$TAG.pt"
  if "$REPO/.venv-arm/Scripts/python.exe" -m kneemri.training.pretrain \
      --objective latent --epochs 8 --slices-per-step 96 --lr "$RATE" --seed 2026 \
      --mask-ratio 0.30 --out "$OUT" 2>&1 | tee "$MODELS/experiments/$TAG.full.log" \
      && [ -f "$OUT" ]; then
    SURVIVOR="$OUT"
    break
  fi
  echo "rung lr $RATE was refused by the attention gate; next rung"
done

require "$SURVIVOR"

run_train latent_rate_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --encoder-weights "$SURVIVOR" \
  --labels "$LABELS" --out "$MODELS/latent_rate_dense_s2026.pt"
