#!/usr/bin/env bash
# bead: knee-ted6u
# reading: basinsoup2_k*_s2026.pt
#
# WHAT THIS ARM IS
#   basinsoup.sh again with ONE term changed: --epochs 8 instead of 4. Same parent, same
#   --backbone-lr 2e-5, same everything else. The product is a SINGLE checkpoint with a single
#   forward pass, not an ensemble.
#
# WHY THIS TERM AND NOT A KNOB SWEEP
#   basinsoup read +0.0016 (parent 0.9073 -> 0.9089), the first positive weight souping has produced
#   here. Its own measurement names what bounded it: K=3 and K=4 both read 0.9089 -- adding the PARENT
#   to the average changed nothing to four decimals, so the children were still sitting on the parent's
#   point and the average had almost nothing to average. Child divergence, not K, is the constraint.
#
#   Cycle LENGTH is the term that buys divergence without the excursion basinsoup argued against. A
#   one-cycle spends more wall-time near the peak before it anneals, so the child travels further and
#   still lands in a flat point at the end. Raising the peak LR would instead reproduce, inside this
#   arm, the independent-runs setting the two earlier closes already refuted -- so the LR is deliberately
#   untouched, and this arm changes exactly one thing against a run whose result is already on disk.
#
# PRE-REGISTERED, on ledger75 @ holdout909, parent = 0.9073, basinsoup_k3 = 0.9089
#   >= 0.9117       carries the bar -- the offline75 a solo needs to beat the pinned 0.946 champion.
#   0.9089..0.9117  divergence helps and is insufficient. Keep the better soup, no slot.
#   < 0.9089        cycle length is not the term either. Souping closes on its CONSTRAINT.
#
# THE FALSIFIER IS BUILT IN. If K=3 and K=4 again agree to four decimals, the children did not move,
#   and that is the reading whatever the mean does -- a soup whose K term is inert is a parent.
#
# deliberate: --epochs 8, the ONLY diff against basinsoup.sh. Everything else is that arm's child line
#             byte for byte, including the nine flags neither line mentions and which therefore resolve
#             to the same `TrainConfig` defaults the parent's stamp records -- so the twelve architecture
#             keys and three split keys `weight_soup` refuses on are identical across parent, children
#             and both soups, by construction rather than by coincidence.
# deliberate: --seed 201/202/203, fresh seeds so these three children are not the earlier three re-run.
#             With the trunk supplied by --init the init half of `seed` is inert, so it buys data order
#             and augmentation draws only -- the intended divergence term.
# deliberate: --vram-gb 22 -- the owner's 50-70% band, the proven sibling value.
# deliberate: fold 0 and not -1, so K=4 including the parent is not a FULL_DATA_FOLD leak.
#
# CARD. 8 epochs x ~280 s x 3 children = ~1.9 h, plus the two soups and their 909-study backfill.
#   Nothing else of ours may share the card (`card_check`).
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

PARENT="$MODELS/labelwise_ext_drop40_long_ema_s2026.pt"
require "$PARENT"

for SEED in 201 202 203; do
  run_train "basinsoup2_c$SEED" \
    $RECIPE --epochs 8 --slice-pooling labelwise --whole-stack $THROUGHPUT \
    --backbone-lr 2e-5 --slices 20 --vram-gb 22 --band-low 0.1 --band-high 0.9 --scheme extended \
    --calibrated-targets --hold-annotations --ema \
    --batch 4 --accumulate 3 --slot-dropout 0.40 \
    --init "$PARENT" --init-scope all --seed "$SEED" \
    --labels "$LABELS" --out "$MODELS/basinsoup2_c$SEED.pt"
done

# THE ARM PRICES ITS OWN PRODUCT, and `--price` is part of that now: `weight_soup` leaves a soup with no
#   fingerprint and a NaN holdout, `holdout_backfill` writes the first and `ledger75 --price` the second,
#   because pricing a record is evaluation's arithmetic and inference may not import the layer that
#   scores it (knee-emyky). Without both a soup is scoreable and unshippable.
ARMPY="uv run --no-sync --project $REPO python"

$ARMPY -m kneemri.models.weight_soup \
  --members "$MODELS/basinsoup2_c201.pt" "$MODELS/basinsoup2_c202.pt" "$MODELS/basinsoup2_c203.pt" \
  --out "$MODELS/basinsoup2_k3_s2026.pt"

# K=4 includes the PARENT, and its agreement with K=3 is the DIVERGENCE READING, not a spare candidate.
$ARMPY -m kneemri.models.weight_soup \
  --members "$PARENT" "$MODELS/basinsoup2_c201.pt" "$MODELS/basinsoup2_c202.pt" "$MODELS/basinsoup2_c203.pt" \
  --out "$MODELS/basinsoup2_k4_s2026.pt"

$ARMPY -m kneemri.inference.holdout_backfill --device cuda \
  --checkpoints "$MODELS/basinsoup2_k3_s2026.pt" "$MODELS/basinsoup2_k4_s2026.pt"

$ARMPY -m kneemri.evaluation.ledger75 --holdouts "$MODELS" --price \
  --only basinsoup2_k3_s2026 basinsoup2_k4_s2026 basinsoup2_c201 basinsoup2_c202 basinsoup2_c203 \
         basinsoup_k3_s2026 labelwise_ext_drop40_long_ema_s2026
