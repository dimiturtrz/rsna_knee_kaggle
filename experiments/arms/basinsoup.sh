#!/usr/bin/env bash
# bead: knee-0dfys
# reading: basinsoup_c*.pt
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
#   >= 0.9128       carries the bar — the offline75 a solo needs to beat the pinned 0.946 champion.
#   0.9073..0.9128  real but insufficient alone. Directional, no slot.
#   < 0.9073        the shared basin does not save souping. The axis closes for good.
#
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
