#!/usr/bin/env bash
# bead: knee-p9mp5
# reading: basinsoup_r2_k*_s2026.pt
#
# WHAT THIS ARM IS
#   Round TWO of the basin-soup operation: three 4-epoch seed-children forked off a checkpoint that is
#   ITSELF a soup (`basinsoup_dmlb_k4_s2026`, 0.9092 ledger75, the highest fold-0 number this project
#   owns), and the K=3 and K=4 uniform averages of them. The product is ONE checkpoint with ONE forward
#   pass -- not an ensemble, not a seat, not a pair.
#
# THE QUESTION, which no earlier arm asked
#   Souping has reproduced three times with the same sign and order (+0.0016 basinsoup, +0.0014
#   basinsoup_dmlb, +0.001 on the board via basinsoupff_v1). Every one of those rounds forked children
#   off a parent that had never been souped. Whether the operation COMPOUNDS is unasked, and it is the
#   only mechanism on the shelf with a path to a solo that BEATS the pinned champion rather than tying
#   it: the solo board-transfer (knee-0dfys, slope +1.071, intercept -0.0307, resid sd 0.0019) puts
#   board 0.946 at offline75 0.9117, so one further +0.0014 is short at 0.9106 and two clear it.
#
# THE NAMED RISK
#   knee-0dfys names the binding constraint as child DIVERGENCE, not K -- K=3 and K=4 have now read
#   identical to four decimals on two bases, the direct measurement that the children never left the
#   parent's point. A soup IS the centre of its basin by construction, so children forked off a centre
#   may diverge even less, and the average of three points that did not move is the point itself. That
#   is the mechanism by which this saturates rather than compounds.
#
# PRE-REGISTERED, on ledger75 against the 0.9092 parent and the 0.0053 paired floor
#   >= 0.9106      COMPOUNDS at full size. Round 3 is licensed, and an _ff round-2 becomes a slot with a
#                  mechanism behind it rather than a reseating.
#   0.9093-0.9105  partial: the op still pays but is decaying. Read the K=3/K=4 gap and the child spread
#                  to say whether divergence or the op itself is the term that shrank.
#   <= 0.9092      SATURATES. Souping is a ONE-SHOT operation, not an iterable one, and the family closes
#                  on its own stated bar. No further soup arm is licensed.
#
# ZERO SLOTS. Fold 0, so this is priced offline on ledger75 and never on the board.
#
# deliberate: every flag is `basinsoup_dmlb`'s line byte for byte EXCEPT the parent and the seeds, so the
#             twelve ARCHITECTURE and three SPLIT keys `weight_soup` refuses on are identical across
#             parent and children by construction. The parent is a soup of `canonical` members and is
#             therefore itself `canonical`; declaring `--scheme` would fail the signature after an hour
#             of card.
# deliberate: --epochs 4 and --backbone-lr 2e-5, the values basinsoup measured positive and basinsoup2
#             measured the far side of (K=3 0.9075 / K=4 0.9080 at 8 epochs, both below the 4-epoch
#             pair). The cycle length is fixed by measurement, not a knob with a sweep in it.
# deliberate: --seed 501/502/503, fresh, so these are not any earlier trio re-run. With the trunk
#             supplied by --init the init half of `seed` is inert and it buys data order and augmentation
#             draws only -- the one term that differs between the three children.
# deliberate: --vram-gb 22, the owner's 50-70% cap of 32.6 GB, with --batch 4 --accumulate 3 for it.
# deliberate: K=4 includes the round-1 soup itself, on the same argument the round-1 arm made for its
#             parent: it is the basin's centre and the anchor that pulls the average back if the children
#             have drifted. K=3 minus K=4 IS the divergence reading, and it is the number that says which
#             branch the shortfall belongs to if this lands short.
#
# CARD. 4 epochs x ~350 s x 3 children = ~1.2 h, plus ~10 min to backfill and price the two soups.
#   ~1.4 h total. Nothing else of ours may share the card.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

PARENT="$MODELS/basinsoup_dmlb_k4_s2026.pt"
require "$PARENT"

for SEED in 501 502 503; do
  run_train "basinsoup_r2_c$SEED" \
    $RECIPE --epochs 4 --slice-pooling labelwise --whole-stack $THROUGHPUT \
    --backbone-lr 2e-5 --slices 24 --vram-gb 22 --band-low 0.1 --band-high 0.9 \
    --calibrated-targets --hold-annotations --ema \
    --batch 4 --accumulate 3 --slot-dropout 0.40 \
    --init "$PARENT" --init-scope all --seed "$SEED" \
    --labels "$LABELS" --out "$MODELS/basinsoup_r2_c$SEED.pt"
done

ARMPY="uv run --no-sync --project $REPO python"

$ARMPY -m kneemri.models.weight_soup \
  --members "$MODELS/basinsoup_r2_c501.pt" "$MODELS/basinsoup_r2_c502.pt" \
  "$MODELS/basinsoup_r2_c503.pt" \
  --out "$MODELS/basinsoup_r2_k3_s2026.pt"

$ARMPY -m kneemri.models.weight_soup \
  --members "$PARENT" "$MODELS/basinsoup_r2_c501.pt" "$MODELS/basinsoup_r2_c502.pt" \
  "$MODELS/basinsoup_r2_c503.pt" \
  --out "$MODELS/basinsoup_r2_k4_s2026.pt"

$ARMPY -m kneemri.inference.holdout_backfill --device cuda \
  --checkpoints "$MODELS/basinsoup_r2_k3_s2026.pt" "$MODELS/basinsoup_r2_k4_s2026.pt"

$ARMPY -m kneemri.evaluation.ledger75 --holdouts "$MODELS" \
  --only basinsoup_r2_k3_s2026 basinsoup_r2_k4_s2026 \
         basinsoup_r2_c501 basinsoup_r2_c502 basinsoup_r2_c503 basinsoup_dmlb_k4_s2026
