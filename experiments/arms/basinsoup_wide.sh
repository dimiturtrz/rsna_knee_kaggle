#!/usr/bin/env bash
# bead: knee-gxu5r
# reading: basinsoup_dmlb_k[67]_s2026.pt
#
# WHAT THIS ARM IS
#   Round 1 (`basinsoup_dmlb.sh`) extended by three more children off the SAME parent with the SAME
#   recipe. Only the member count moves. The product of each soup is ONE checkpoint, ONE forward pass.
#
# THE QUESTION, and why it is the last one on this axis
#   knee-grkrv closed souping as a LADDER on a mechanism: leaving a soup is a STEP, not a slope, so any
#   resumed gradient forfeits the averaged point in full and a round's net is structurally ~0. Three
#   rounds, two fork recipes:
#
#     round 1   children 0.9062   parent dml_b (PLAIN FIT) 0.9078   soup 0.9092   net +0.0014
#     round 2   children 0.9067   parent basinsoup (SOUP)   0.9092   soup 0.9091   net -0.0001
#     round 3   children 0.9059   parent basinsoup (SOUP)   0.9092   soup 0.9081   net -0.0011
#
#   The rule that survives is AVERAGING MUST BE THE LAST OPERATION, and the one-shot off a PLAIN FIT is
#   the one form that pays. That operation has exactly ONE free parameter never swept: K.
#
# deliberate: --epochs 4 --backbone-lr 2e-5 --init dml_b_s2026 --init-scope all — round 1's recipe
#             UNCHANGED. This arm moves K and nothing else; a recipe change would cross two variables
#             and make the K reading unattributable.
# deliberate: seeds 404/405/406, continuing round 1's 401/402/403 series. With the trunk supplied by
#             --init the init half of `seed` is inert; it buys data order and augmentation draws only,
#             which is exactly the noise the average is supposed to cancel.
# deliberate: BOTH K=6 (six children, no parent) and K=7 (parent + six). K=4 included the parent, so
#             K=7 is the like-for-like extension; K=6 separates "more children" from "the parent's pull".
# deliberate: --vram-gb 22 --batch 4 --accumulate 3 — the owner's 50-70% cap of 32.6 GB, unchanged.
#
# WHY IT IS AN INSTRUMENT, NOT AN IMPROVEMENT
#   Expected offline gain ~+0.001, under every floor (0.0053 paired, 0.0062 seed). It is worth 0.6 h
#   because it DECIDES a bigger spend: on the board, shipping soup-of-ff reads K=3 0.945 and K=4 0.946,
#   and extending to K=8 costs ~1.2 h of card AND A SLOT. One board tick is inside the ~0.004 2-sd
#   scatter and cannot license that. Fold-0 can, for free — _ff members fit --fold -1 and can never be
#   priced offline at all (knee-s8kim).
#
# PRE-REGISTERED on ledger75 against basinsoup_dmlb_k4_s2026 = 0.9092:
#   >= 0.9102      the premium SCALES with K. Licenses soup-of-ff K=8, the best-priced slot available.
#   0.9093-0.9101  weak or decaying scale. Real, not worth 1.2 h and a slot. Record and stop.
#   <= 0.9092      the premium is FLAT in K. K-extension dead, soup-of-ff K=4 final, and weight space
#                  closes with NO free parameter left rather than with one untested.
#
# DIAGNOSTIC READABLE EVEN ON A NULL: the new children's mean. c401/402/403 read 0.9061/0.9062/0.9062.
#   If 404/405/406 miss that band the seeds are not exchangeable and the K reading is VOID rather than
#   negative — a harness verdict, not a mechanism one.
#
# ZERO SLOTS. Fold 0, priced on ledger75, never on the board.
#
# CARD. 4 epochs x ~175 s x 3 children = ~35 min, plus ~10 min to backfill and price two soups. ~0.8 h.
#   Nothing else of ours may share the card.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

PARENT="$MODELS/dml_b_s2026.pt"
require "$PARENT"
require "$MODELS/basinsoup_dmlb_c401.pt"
require "$MODELS/basinsoup_dmlb_c402.pt"
require "$MODELS/basinsoup_dmlb_c403.pt"

for SEED in 404 405 406; do
  run_train "basinsoup_dmlb_c$SEED" \
    $RECIPE --epochs 4 --slice-pooling labelwise --whole-stack $THROUGHPUT \
    --backbone-lr 2e-5 --slices 24 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
    --calibrated-targets --hold-annotations --ema \
    --batch 4 --accumulate 3 --slot-dropout 0.40 \
    --init "$PARENT" --init-scope all --seed "$SEED" \
    --labels "$LABELS" --out "$MODELS/basinsoup_dmlb_c$SEED.pt"
done

ARMPY="uv run --no-sync --project $REPO python"

CHILDREN=""
for SEED in 401 402 403 404 405 406; do
  CHILDREN="$CHILDREN $MODELS/basinsoup_dmlb_c$SEED.pt"
done

$ARMPY -m kneemri.models.weight_soup \
  --members $CHILDREN \
  --out "$MODELS/basinsoup_dmlb_k6_s2026.pt"

$ARMPY -m kneemri.models.weight_soup \
  --members "$PARENT" $CHILDREN \
  --out "$MODELS/basinsoup_dmlb_k7_s2026.pt"

$ARMPY -m kneemri.inference.holdout_backfill --device cuda \
  --checkpoints "$MODELS/basinsoup_dmlb_k6_s2026.pt" "$MODELS/basinsoup_dmlb_k7_s2026.pt"

$ARMPY -m kneemri.evaluation.ledger75 --holdouts "$MODELS" \
  --only basinsoup_dmlb_k7_s2026 basinsoup_dmlb_k6_s2026 basinsoup_dmlb_k4_s2026 \
         basinsoup_dmlb_c404 basinsoup_dmlb_c405 basinsoup_dmlb_c406 \
         basinsoup_dmlb_c401 basinsoup_dmlb_c402 basinsoup_dmlb_c403 \
         dml_b_s2026
