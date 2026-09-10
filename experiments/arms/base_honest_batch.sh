#!/usr/bin/env bash
# base_honest_batch.sh — does trunk CAPACITY pay once the batch handicap is mostly taken off it?
#
# THE HOLE THIS ARM FILLS, and it is a hole in the whole capacity lane rather than in one row.
# Every wider-trunk number this project owns was measured through a batch substitution, and the
# substitution costs more than most of the gains being read:
#
#   band_dense       small   336 px / 20 sl   batch 12                                   0.8578
#   base_dense       base    336 px           batch 2 / accumulate 3                     0.8512
#   v3base_dense     base    336 px           batch 2 / accumulate 3                     0.8455
#   ortho_stack      ViT-L   336 px / 20 sl   batch 1 / accumulate 12 + pair-memory 66   0.8417
#   ortho_dense      ViT-L   224 px / 12 sl   batch 12                                   0.8452
#
# `band_pairmem` prices the batch-1 substitution at **-0.0092** on the small trunk at this geometry.
# So `base_dense` - `band_dense` = -0.0066 and `ortho_stack` - `band_dense` = -0.0161 are both read
# through a confound of the same order as the difference itself. Not one row above is a clean capacity
# measurement, and the lane has been called closed on them twice.
#
# WHY THE CLEAN CELL WAS NEVER RUN: IT DOES NOT FIT. Priced 2026-09-10 with `vram_need` at the shipped
# geometry (336 px / 20 slices / batch 12 / recompute on - gradient checkpointing is already wired at
# `encoder_surgery.py:443` and is IN these numbers) against a 32,607 MiB card:
#
#   small  28,934 MiB   <- what `band_dense` actually holds. 3.7 GB of headroom, and that is ALL of it.
#   base   58,700 MiB
#   large  81,400 MiB
#
# The winning geometry has exactly ONE rung of the capacity ladder on it and we are standing on it.
# That is not "capacity is untested" - it is capacity being structurally unreachable at 336/20/batch 12
# on this card, which is a different sentence and belongs in the ledger as one.
#
# WHAT THIS ARM DOES ABOUT IT. It buys the wider trunk with BATCH rather than with geometry, and stops
# well short of the substitution that poisoned the rows above. base at 336/20 prices at 25,207 MiB at
# batch 5 (29,992 at 6, 34,777 at 7 - over). Every other flag is `band_dense`'s, on `band_dense`'s own
# `336px_20sl_0.1-0.9band` array: no new cache, no geometry trade, no accumulation, no pair memory.
#
# WHY BATCH 5 AND NOT 6. 6 fits at 29,992 with 2.6 GB spare, less headroom than `band_dense` itself has.
# The line is not the number to aim at: `mae_unitfix_s2026_run1` held 32,024 MiB of this card, reported
# 51 MiB free from its first step, ran epoch 0 at 510 img/s and collapsed to 57 img/s the moment epoch
# 0's holdout eval allocated on top - WDDM pages over PCIe instead of failing, and it never recovered.
# 7.4 GB of headroom is the deliberate price of not repeating that.
#
# WHAT BATCH 5 STILL COSTS, stated rather than waved at. The ranking loss draws pairs within the batch:
# 66 pairs at 12, 10 at 5, 1 at 2. So this arm does not remove the confound, it shrinks it by an order
# of magnitude and - this is the part that makes it readable - it lands a THIRD point on the batch axis
# of the base trunk, where `base_dense` already holds batch 2. `base_honest_batch` - `base_dense` is a
# pure batch read at fixed trunk and fixed geometry, which nothing in this project owns yet.
#
# PRE-REGISTERED, and the branches are about CAPACITY, not about batch:
#
#   s >= 0.8578 (band_dense)   Capacity pays once the batch is honest, and the four rows above were
#               reading their own handicap. The trunk axis reopens with a mechanism and a direction,
#               it licenses the ladder's next rung, and it is the first single-model candidate above
#               our best in eleven arms.
#   0.8512-0.8577   Capacity is worth something and the small trunk is worth more at this n on this
#               card. Read the gap over `base_dense` first: a rise from 0.8512 sizes the batch effect
#               on the base trunk, and a FLAT against it says the -0.0092 correction does not transfer
#               across trunk widths - which would retire the handicap as an explanation for the whole
#               wider-trunk table, including `ortho_stack`.
#   s < 0.8512   The wider trunk is worse than its own handicapped run, so the batch was never what
#               held it back. Capacity is retired as the last open suspect at n=4,407 and the ledger's
#               "capacity or supervision" narrows to one.
#
# PRICE. COST: base is ~86M against small's ~21M at an identical 20 x 441 = 8,820 patch tokens, and
# batch 5 costs step efficiency on top, so this lands near 2-2.5x `band_dense` per epoch; 12 epochs is
# ~3-4 h of card. GAIN: the top branch puts a single member at or above 0.8578 where the target needs
# ~0.874, and it is the only unrun cell that can move the trunk axis at all - the corpus lane is closed
# by D310, the seating axis closed across families by `cnnseat_v1` (0.944, dead flat), the pooling
# family closed on width and operator alike, and the data slope is concave with the next doubling
# priced at +0.009. P(top branch) ~ 0.25. Even the bottom branch is worth the hours as an INSTRUMENT:
# it retires the last suspect the ledger names, and no other arm on the board can.
#
# WHAT THIS IS NOT: not a seating, not a re-lensing, not a knob on the settled recipe. It is a member
# attempt with an instrument's floor.
#
# bead: knee-omnzl
# reading: base_honest_batch_s2026.holdout.npz
# deliberate: --variant base (the whole point)
# deliberate: --batch 5 with NO --accumulate and NO --pair-memory (the confound this arm shrinks)
# deliberate: --slices 20 --band-low 0.1 --band-high 0.9 --image-size default 336 (band_dense's, held)
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train base_honest_batch \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant base --batch 5 \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$MODELS/base_honest_batch_s2026.pt"
