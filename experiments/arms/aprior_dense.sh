#!/usr/bin/env bash
# aprior_dense.sh — supply the plane-by-finding mapping the head currently has to DISCOVER, and see
# whether scarce supervision spends better when it is not spent rediscovering anatomy.
#
# WHY THIS AXIS, AND WHY NOW. Four independent lines converge on the same short list
# (interpretations/kneemri/2026-08-20_what_is_left_after_decode_and_data.md): the pooling OPERATOR is
# closed (attend_dense −0.0009), the pooled WIDTH is closed (focal_dense −0.0044), the four data levers
# knee-jk3 names are each individually priced, and the offline→board chain has gone flat. What every
# one of those arms held fixed is the SUPERVISION, and the ledger's own conclusion at the bottom of the
# focal_dense row says so in as many words. Of the three remaining axes two are held pending a
# decision; this is the third thing on that list that needs neither a new label nor a new corpus.
#
# WHAT THE FLAG ACTUALLY DOES, read rather than assumed. `SlotHead` gives every diagnosis its own
# softmax over the six acquisition slots and initialises `self.prior` at ZEROS, so which sequence shows
# which finding is something 4,407 study-level label vectors — with no localisation in them at all —
# have to teach it. `--anatomical-prior` initialises that same parameter from the 12x6 table in
# `core/data/corpus/anatomy.py` instead, generated from two facts per finding (which planes show the
# structure, whether it needs fluid weighting) at PRIOR_STRENGTH 2.0. It stays an `nn.Parameter`, so
# the data can overrule it anywhere it disagrees; every entry is non-negative, so it is a preference
# over slots and never a mask. Parameter count, shapes, compute, VRAM and wall-clock are unchanged —
# this arm costs exactly what its control cost, and the only thing that differs is where 72 numbers
# start.
#
# IT IS ALSO ONE OF THE FIVE NEVER-RUN SWITCHES. The 2026-08-18 `--switches` diff found five flags
# landed-checked, stamped and never once executed on the card. `--pool cls_mean_focal` was the first of
# them to be spent (focal_dense, −0.0044) and this is the second. `switch_coverage` already smokes it
# to a written checkpoint at 60 studies / 1 epoch, and `landed.py` asserts it constructed, so what is
# unknown here is the SCIENCE and not the plumbing.
#
# TWELVE EPOCHS, NOT TWENTY-FOUR, and that is the cost tier rather than a compromise. `band_dense` is
# the settled 12-epoch control at 0.8578 (best epoch 8, 361 s an epoch) and `band_dense_long` shows the
# same recipe turning down after epoch 6, so length is not what this question needs. One-cycle makes
# `--epochs` a schedule and not a truncation, so the control has to be matched on it: 12 against 12.
# If the prior moves the mean, the 24-epoch re-run is the follow-up and it is cheap to justify then.
#
# control  band_dense = 0.8578 (best epoch 8 of 12, 4e-5, 20 slices, band 0.10-0.90, max pooling,
#          whole stack, slot_pooling attend — verified in the recorded recipe, anatomical_prior False).
#          Single variable: --anatomical-prior. Same schedule, same operating point, same cache.
# seat     comp/champion.json greedy4_v1 = 0.8679 over the 909-study fold-0 holdout.
# cost     about 1.2 h; the flag adds no compute.
#
# PRE-REGISTERED, three outcomes:
#   PREDICTION — the peak mean clears band_dense's 0.8578 by more than the 0.0062 holdout floor, and
#     the gain concentrates in the findings whose prior ROW IS PEAKED rather than flat: MCL (coronal
#     alone), PF OA and Synovitis (axial-weighted), Medial and Lateral OA (T1-weighted). Those are the
#     rows where the table says something the label vectors cannot, so they are where a warm start can
#     pay. ACL and both menisci have near-flat rows across sagittal and coronal and should barely move.
#   REFUTATION — the peak mean is at or below 0.8578. Then the head had already recovered the mapping
#     from study-level supervision alone, "supervision never says where" is wrong about THIS head, and
#     the localisation argument has to buy a real localisation signal (knee-mxj) rather than a warm
#     start. That is a genuine narrowing and it costs 1.2 h to get.
#   THIRD OUTCOME — the mean is flat but the per-label pattern moves along the peaked/flat split above.
#     Then the prior does what it claims and the ceiling is bounded elsewhere; the member stops being a
#     mean lever and becomes a DIVERSITY generator on the deep24 mechanism — errors shaped by a
#     different attention geometry than the other sixteen — and is priced by seat value against
#     greedy4_v1 rather than by its solo score.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train aprior_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --anatomical-prior \
  --labels "$LABELS" --out "$PROCESSED/aprior_dense_s2026.pt"
