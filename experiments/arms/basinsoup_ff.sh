#!/usr/bin/env bash
# bead: knee-mc1ls
# reading: basinsoup_ff_k*_s2026.pt
#
# WHAT THIS ARM IS
#   `basinsoup.sh` moved onto the FULL-DATA line: three 4-epoch seed-children forked off the shipping
#   solo `labelwise_ext_drop40_long_ff.pt` at `--fold -1`, then the K=3 and K=4 uniform weight averages.
#   The product is a SINGLE checkpoint with a SINGLE forward pass, not an ensemble and not a seat.
#
# WHY THIS OPERATION AND NOT A NEW RECIPE
#   basinsoup is the only weight-space operation this project has ever measured positive: +0.0016
#   ledger75, parent 0.9073 -> basinsoup_k3 0.9089. `basinsoup2` then established WHY it pays and why it
#   cannot be pushed: every child of a converged EMA parent lands BELOW it, so the average can only
#   interpolate between one good point and worse ones, and +0.0016 is the reading of the least-travelled
#   children available. Doubling the cycle bought real divergence (K=3 0.9075 against K=4 0.9080, where
#   basinsoup had the two equal to four decimals) and the product FELL. So 4 epochs at 2e-5 is the
#   measured optimum of this op rather than a knob with a sweep in it, and it is copied here unchanged.
#
# WHY THE PARENT IS THE SHIPPING ENTRY
#   `labelwise_ext_drop40_long_ff.pt` IS `extffsolo_v1`, board 0.945. Every other route priced this week
#   proposes a NEW member of a cluster whose _ff refits all board 0.943-0.945 (knee-xjxdn: the top four
#   recipes sit inside the 0.0053 paired floor and none has produced 0.946). This arm proposes no new
#   member: it applies a measured +delta ON TOP of the 0.945 entry already held.
#
# PRE-REGISTERED ON THE BOARD, because fold -1 leaves no offline ruler to read
#   >= 0.947   the op transfers; a new champion and the best solo this project owns.
#   0.946      a tie at the board's own 0.001 resolution. Keep the standing entry.
#   <= 0.945   weight souping does not transfer to the full-data fit, and the family closes on its
#              FOURTH and last explanation.
#
# deliberate: --fold -1 on the child line, AFTER `$RECIPE` (which sets `--fold 0`), so the later flag
#             wins. All three children and both soups are therefore full-data, which is the ONLY
#             configuration `weight_soup --ship-only` accepts: a mixed pot still fails the signature,
#             since `fold` lives in `SPLIT`.
# deliberate: NO `holdout_backfill` and NO `ledger75` tail, unlike both earlier basinsoup arms. A
#             full-data fit has seen every study, so a holdout record for it would be a number about
#             training data. Leaving those two commands out is the reading, not an omission.
# deliberate: --epochs 4 and --backbone-lr 2e-5, the values basinsoup measured positive and basinsoup2
#             measured the far side of. Neither is touched.
# deliberate: --seed 301/302/303, fresh, so these are not either earlier trio re-run. With the trunk
#             supplied by --init the init half of `seed` is inert and it buys data order and
#             augmentation draws only -- the one term that differs between the three children.
# deliberate: --init "$PARENT" --init-scope all, the flag that makes this a basin fork rather than the
#             independent-runs setting already refuted twice. `all` because the parent's head is exactly
#             the head the children want: same corpus, same nine slots.
# deliberate: every other flag is the parent's line byte for byte, so the twelve ARCHITECTURE keys and
#             three SPLIT keys `weight_soup` refuses on are identical across parent and children by
#             construction -- the nine that neither line mentions fall through to the same `TrainConfig`
#             defaults, which is where a refusal after an hour of card would otherwise hide.
# deliberate: --vram-gb 22, the owner's 50-70% band.
#
# CARD. 4 epochs x ~350 s x 3 children = ~1.2 h. Full data is ~25% more studies an epoch than fold 0,
#   which is where the 280 s the earlier arms measured goes. Nothing else of ours may share the card.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

PARENT="$MODELS/labelwise_ext_drop40_long_ff.pt"
require "$PARENT"

for SEED in 301 302 303; do
  run_train "basinsoup_ff_c$SEED" \
    $RECIPE --fold -1 --epochs 4 --slice-pooling labelwise --whole-stack $THROUGHPUT \
    --backbone-lr 2e-5 --slices 20 --vram-gb 22 --band-low 0.1 --band-high 0.9 --scheme extended \
    --calibrated-targets --hold-annotations --ema \
    --batch 4 --accumulate 3 --slot-dropout 0.40 \
    --init "$PARENT" --init-scope all --seed "$SEED" \
    --labels "$LABELS" --out "$MODELS/basinsoup_ff_c$SEED.pt"
done

ARMPY="uv run --no-sync --project $REPO python"

$ARMPY -m kneemri.models.weight_soup --ship-only \
  --members "$MODELS/basinsoup_ff_c301.pt" "$MODELS/basinsoup_ff_c302.pt" "$MODELS/basinsoup_ff_c303.pt" \
  --out "$MODELS/basinsoup_ff_k3_s2026.pt"

# K=4 includes the PARENT, the basin's centre by construction and the only member with a long fit behind
#   it. On the fold-0 line the K=3/K=4 agreement was itself the divergence reading; here it is the
#   anchored candidate, and the two are the arm's two shippable products.
$ARMPY -m kneemri.models.weight_soup --ship-only \
  --members "$PARENT" "$MODELS/basinsoup_ff_c301.pt" "$MODELS/basinsoup_ff_c302.pt" \
  "$MODELS/basinsoup_ff_c303.pt" \
  --out "$MODELS/basinsoup_ff_k4_s2026.pt"
