#!/usr/bin/env bash
# bead: knee-grkrv
# reading: basinsoup_soft_k*_s2026.pt
#
# WHAT THIS ARM IS
#   `basinsoup_r2.sh` byte for byte -- same parent `basinsoup_dmlb_k4_s2026`, same K=3/K=4 shape, same
#   VRAM shape, same child count -- with the FORK alone moved to `--epochs 2 --backbone-lr 1e-5`. A
#   matched contrast against round 2 on ONE variable. The product is ONE checkpoint, ONE forward pass.
#
# THE QUESTION, and why round 2 could not ask it
#   Round 2 read `<= 0.9092` and closed the family as SATURATED. That reading is superseded: it never
#   priced its children against the parent, so it could not separate the two terms that make up the net.
#   Priced 2026-09-21 (ledger75, 909 studies, population eb3ba199):
#
#     round 1   children mean 0.9062   parent dml_b 0.9078            soup 0.9092
#     round 2   children mean 0.9067   parent basinsoup_dmlb 0.9092   soup 0.9091
#
#     SOUP PREMIUM over children mean   +0.0030 (r1)   +0.0024 (r2)
#     FORK DEGRADATION vs parent        -0.0016 (r1)   -0.0025 (r2)
#     net                               +0.0014        -0.0001
#
#   The premium is CONSTANT and DIVERGENCE-INDEPENDENT -- child spread went 0.0001 -> 0.0034, 34x, and
#   the premium moved 0.0006, inside noise. Averaging never stopped working. DEGRADATION is the term
#   that changed, it grows with parent strength, and it is the only one a recipe can move.
#
# deliberate: --epochs 2 --backbone-lr 1e-5 — the ONE moved variable, ~4x less weight movement than the
#             fork round 2 held fixed. Round 2 called those values "fixed by measurement, not a knob
#             with a sweep in it", and they were measured positive on a NON-souped parent where
#             degradation was 0.0016. Nothing measured them on a souped one.
# deliberate: --seed 701/702/703, fresh, so these are no earlier trio re-run. With the trunk supplied by
#             --init the init half of `seed` is inert; it buys data order and augmentation draws only.
# deliberate: K=4 includes the parent, on round 2's own argument — it is the basin's centre and the
#             anchor that pulls the average back. K=3 minus K=4 IS the divergence reading.
# deliberate: --vram-gb 22 --batch 4 --accumulate 3 — the owner's 50-70% cap of 32.6 GB, unchanged.
#
# READABLE EVEN ON A NULL: the CHILD MEAN. Above 0.9067 says degradation shrank and the mechanism holds
#   whatever the soup nets; at or below says the fork rate does not set degradation and the two-term
#   model is wrong rather than the recipe.
#
# ZERO SLOTS. Fold 0, priced offline on ledger75, never on the board.
#
# CARD. 2 epochs x ~350 s x 3 children = ~35 min, plus ~10 min to backfill and price the two soups.
#   ~0.8 h total. Nothing else of ours may share the card.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

PARENT="$MODELS/basinsoup_dmlb_k4_s2026.pt"
require "$PARENT"

for SEED in 701 702 703; do
  run_train "basinsoup_soft_c$SEED" \
    $RECIPE --epochs 2 --slice-pooling labelwise --whole-stack $THROUGHPUT \
    --backbone-lr 1e-5 --slices 24 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
    --calibrated-targets --hold-annotations --ema \
    --batch 4 --accumulate 3 --slot-dropout 0.40 \
    --init "$PARENT" --init-scope all --seed "$SEED" \
    --labels "$LABELS" --out "$MODELS/basinsoup_soft_c$SEED.pt"
done

ARMPY="uv run --no-sync --project $REPO python"

$ARMPY -m kneemri.models.weight_soup \
  --members "$MODELS/basinsoup_soft_c701.pt" "$MODELS/basinsoup_soft_c702.pt" \
  "$MODELS/basinsoup_soft_c703.pt" \
  --out "$MODELS/basinsoup_soft_k3_s2026.pt"

$ARMPY -m kneemri.models.weight_soup \
  --members "$PARENT" "$MODELS/basinsoup_soft_c701.pt" "$MODELS/basinsoup_soft_c702.pt" \
  "$MODELS/basinsoup_soft_c703.pt" \
  --out "$MODELS/basinsoup_soft_k4_s2026.pt"

$ARMPY -m kneemri.inference.holdout_backfill --device cuda \
  --checkpoints "$MODELS/basinsoup_soft_k3_s2026.pt" "$MODELS/basinsoup_soft_k4_s2026.pt"

$ARMPY -m kneemri.evaluation.ledger75 --holdouts "$MODELS" \
  --only basinsoup_soft_k3_s2026 basinsoup_soft_k4_s2026 \
         basinsoup_soft_c701 basinsoup_soft_c702 basinsoup_soft_c703 \
         basinsoup_dmlb_k4_s2026 basinsoup_r2_k4_s2026
