#!/usr/bin/env bash
# bead: knee-0dfys
# reading: basinsoup_k*_s2026.pt
#
# WHAT THIS ARM IS
#   Three seed-children forked off ONE already-corpus-fine-tuned parent, so their weight-space average
#   has a shared basin to average inside. The product is a SINGLE checkpoint with a single forward pass,
#   not an ensemble.
#
# WHY THIS AND NOT A LARGER K
#   Weight souping is closed here on two explanations and must not be re-opened by a bigger K, another
#   pair or a longer schedule. Its close names the ONLY way back in: "Re-open only with a mechanism that
#   puts the members in a SHARED basin (a common fine-tune trunk, not a common init), which no arm in
#   this ledger has ever produced." Every member ever souped here shared only the `facebook/dinov2-small`
#   ImageNet init — twenty recipes, twenty basins — which is precisely the setting a UNIFORM soup is
#   published to fail in. So the two closes are the literature reproducing, not a refutation of souping.
#   `--init <parent> --init-scope all` is the mechanism the close asked for and the flag already exists.
#
# THE HONEST COUNTER, STATED FIRST
#   `weight_average.sh` averaged ONE run across ITS OWN epochs and read +0.0003 — a near-null for
#   parameter averaging in this repo. This is not that: an EMA averages consecutive, highly-correlated
#   iterates, while seed-forked children diverge in data order and augmentation draws over their own
#   epochs. If that difference is not real, this arm reproduces the K-law (-0.0044 per doubling, so
#   ~0.9003 at K=3) and weight souping closes on its THIRD explanation, permanently.
#
# PRE-REGISTERED, on ledger75 @ holdout909, parent = 0.9073
#   >= 0.9117       carries the bar — the offline75 a solo needs to beat the pinned 0.946 champion.
#   0.9073..0.9117  real but insufficient alone. Directional, no slot.
#   < 0.9073        the shared basin does not save souping. The axis closes for good.
#
# THE BAR IS RE-DERIVED, NOT QUOTED (2026-09-20T14:04Z). `board_transfer` has no `solo` scope, so this
#   fit exists only as prose in a board row — the exact shape of the constant this repo has now quoted
#   wrong five times. Refitted off `comp/board.jsonl` directly: the ten single-member entries carrying
#   both numbers give OLS slope +1.071, intercept -0.0307, residual sd 0.0019 (population) / 0.0021
#   (sample), which reproduces the recorded fit. Inverting it puts board 0.946 at offline75 0.9117 and
#   board 0.955 at 0.9201 — so the bar carried into this arm as 0.9128 was 0.0011 TOO HIGH, half a
#   residual sd, enough to read a marginal win as merely directional. The parent's own 0.9073 projects
#   to board 0.9413, five ticks under the champion; the soup must find +0.0044 offline75 to clear it.
#
# deliberate: PARENT = labelwise_ext_drop40_long_ema_s2026 (0.9073), and NOT the rank-1 member.
#             `dml_b_s2026` reads 0.9078, ahead by 0.0005 — a TENTH of the 0.0053 paired floor, so the
#             offline ruler cannot order the two at all and "fork off the best member" does not pick
#             between them. What does pick is cost of the fork. `dml_b`'s recipe carries
#             `--distill-weight 0.3 --teacher-targets`, so every child needs a teacher-readings file
#             rebuilt for its split, and the one on disk covers the fold-0 TRAINING split only —
#             `TeacherTargets.rows` raises on any study it does not cover. That is a second
#             card-bound pass, a leak surface (`teacher_readings` refuses `--fold -1` as an argument
#             for exactly this reason) and a third term varying between parent and children, when the
#             soup's whole premise is that ONLY the seed differs. The ext parent needs none of it.
# deliberate: --init "$PARENT" --init-scope all — the one flag that makes this a basin fork rather than
#             the independent-runs setting already refuted twice. `all` and not `trunk`: the parent's
#             head is exactly the head the children want, same corpus and same six slots.
# deliberate: --seed 101/102/103 — the ONLY term that differs between the three children. `seed` says
#             "model init and data order ONLY" (config_tree.py:80), and with the trunk supplied by
#             --init the init half is inert, so seed here buys exactly data order and augmentation draws.
# deliberate: --epochs 4 — a short full one-cycle re-anneal per child. Long enough to decorrelate the
#             children, short enough that each lands back in a flat point rather than wandering out of
#             the parent's basin, which is the condition the average needs.
# deliberate: --backbone-lr 2e-5, HALF the parent's 4e-5 — and this is the arm's one real design choice,
#             fixed by the mechanism rather than tuned. A one-cycle at the parent's own peak LR is a
#             FULL-amplitude excursion from a converged point: it would throw the children out of the
#             basin and reintroduce, inside this arm, the exact independent-runs condition the K-law
#             already measured. Reduced-amplitude cycling from a converged point is the standard
#             snapshot/SWA construction for staying local while still decorrelating. LR is not in
#             `weight_soup`'s ARCHITECTURE tuple, so halving it does not make the average a different
#             network. The failure this risks is the opposite one — too local, children ~ parent,
#             average ~ parent, a null — and that null is branch three, not a wasted card.
# deliberate: every other flag is the parent's line byte for byte, so the architecture tuple
#             `weight_soup` refuses on is identical across parent and all three children.
#             VERIFIED AGAINST THE PARENT'S OWN STAMP, not against the arm file that claims to have
#             produced it (2026-09-20T14:30Z). `weight_soup` refuses on twelve ARCHITECTURE keys plus
#             three SPLIT keys, and NINE of those fifteen are set by neither line — `$RECIPE` is only
#             `--fold 0 --one-cycle --weight-floor --rank-weight 1.0`, so variant, pool, slot_pooling,
#             image_size, unfreeze_last, encoder_layer, lora_rank, folds and split_seed all fall through
#             to `TrainConfig` defaults. A claim of "byte for byte" says nothing about a key neither line
#             mentions, which is where a refusal after 1.1 h of card would have hidden. The parent's
#             stamp reads small / cls_mean / attend / 336 / 20 / whole_stack / 6 / 0.1-0.9 / -1 / 0, and
#             folds 5 / split_seed 2026 / fold 0 — and the child line omits exactly the same nine flags,
#             so both resolve identically by construction rather than by coincidence. The only diffs are
#             --epochs, --backbone-lr, --vram-gb, --init, --init-scope and --seed, none of which is in
#             either tuple. fold is 0 and not -1, so K=4 including the parent is not a FULL_DATA_FOLD
#             leak. Same `$LABELS` on both lines, so the same scored-against table.
# deliberate: --vram-gb 22 — the owner's 50-70% band, the proven sibling value.
#
# CARD. ~4 epochs x ~280 s x 3 children = ~1.1 h, plus the soup and its 909-study pass. Nothing else of
#   ours may share the card (`card_check`).
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

PARENT="$MODELS/labelwise_ext_drop40_long_ema_s2026.pt"
require "$PARENT"

for SEED in 101 102 103; do
  run_train "basinsoup_c$SEED" \
    $RECIPE --epochs 4 --slice-pooling labelwise --whole-stack $THROUGHPUT \
    --backbone-lr 2e-5 --slices 20 --vram-gb 22 --band-low 0.1 --band-high 0.9 --scheme extended \
    --calibrated-targets --hold-annotations --ema \
    --batch 4 --accumulate 3 --slot-dropout 0.40 \
    --init "$PARENT" --init-scope all --seed "$SEED" \
    --labels "$LABELS" --out "$MODELS/basinsoup_c$SEED.pt"
done

# THE ARM PRICES ITS OWN PRODUCT. Three children are not the reading — the SOUP is, and `weight_soup`
#   clears the `holdout` block it inherits, so a souped checkpoint arrives with no per-study record and
#   `ledger75` cannot see it. `holdout_backfill` is the pass that gives it one. Leaving those three
#   commands to a later session is exactly the shape of the nineteen-unpriced-arms failure.
ARMPY="uv run --no-sync --project $REPO python"

$ARMPY -m kneemri.models.weight_soup \
  --members "$MODELS/basinsoup_c101.pt" "$MODELS/basinsoup_c102.pt" "$MODELS/basinsoup_c103.pt" \
  --out "$MODELS/basinsoup_k3_s2026.pt"

# K=4 includes the PARENT itself, which K=3 does not. The parent is the basin's centre by construction
#   and the only member here with a 24-epoch fit behind it, so if the children have drifted the parent
#   is the anchor that pulls the average back; if they have not, K=4 and K=3 agree and that agreement
#   is itself the measurement. Both are one checkpoint with one forward pass.
$ARMPY -m kneemri.models.weight_soup \
  --members "$PARENT" "$MODELS/basinsoup_c101.pt" "$MODELS/basinsoup_c102.pt" "$MODELS/basinsoup_c103.pt" \
  --out "$MODELS/basinsoup_k4_s2026.pt"

$ARMPY -m kneemri.inference.holdout_backfill --device cuda \
  --checkpoints "$MODELS/basinsoup_k3_s2026.pt" "$MODELS/basinsoup_k4_s2026.pt"

$ARMPY -m kneemri.evaluation.ledger75 --holdouts "$MODELS" \
  --only basinsoup_k3_s2026 basinsoup_k4_s2026 basinsoup_c101 basinsoup_c102 basinsoup_c103 \
         labelwise_ext_drop40_long_ema_s2026
