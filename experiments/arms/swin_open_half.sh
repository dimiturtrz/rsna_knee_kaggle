#!/usr/bin/env bash
# swin_open_half.sh — the non-DINOv2 lane has never had a recipe of its own, and `--unfreeze-last` is
# where the transplant does the most damage. This arm opens the same FRACTION of the trunk the DINOv2
# recipe opens, rather than the same COUNT.
#
# THE DEFECT. `unfreeze_last 6` is a COUNT, and it was fitted on a trunk that is 12 blocks deep. Every
# non-DINOv2 arm this project has run inherited the number without the denominator:
#
#   trunk           blocks   `unfreeze-last 6` opens   best solo
#   dinov2-small       12            50%                0.8964
#   vitin21k           12            50%                0.7847  (5 of 12 epochs — a floor)
#   swinbase           24            25%                0.8331
#   vjepa2l            24            25%                0.8343  (still climbing — a floor)
#   cnxbase            36            17%                0.8186
#
# Read off `Encoder.blocks`, which is the trunk's REAL depth for a hierarchical trunk (`cnxbase` states
# `depth` 12 and runs 36). Among the runs that converged the ordering is monotone in opened fraction.
# That is confounded with architecture and is NOT offered as the explanation — it is the reason this arm
# is the cheapest of the four candidate explanations to eliminate. See
# `interpretations/kneemri/2026-08-30_the-non-dino-lane-is-quality-bound.md`.
#
# WHY THE FRACTION IS THE QUANTITY. A block's job is set by where it sits in the trunk, not by its index
# from the end. The last six blocks of a 12-block ViT are its whole second half — the part that has
# stopped building general features and started composing them. The last six of a 36-block ConvNeXt are
# the tail of one stage: the trunk is adapted almost nowhere, and the head is asked to make up the
# difference from features fitted to ImageNet classes.
#
# WHY `--layer-decay 0.7` COMES WITH IT, AND WHY THAT IS ONE MECHANISM AND NOT TWO FLAGS. Opening twice
# as many blocks at one flat rate is how a deeper opening goes wrong: the early blocks carry the general
# structure and take the same step as the late ones. Layer-wise decay is the standard answer and it is
# already measured here — `band_dense_llrd` pairs `--unfreeze-last 12 --layer-decay 0.7` on DINOv2, where
# the TOP block keeps this recipe's own rate and each block below takes 0.7 of the one above. The pair is
# the mechanism; either half alone is a different arm.
#
# THE RATE IS HELD AT THE CONTROL'S 1e-4 ON PURPOSE. `swin_supervised` already established 1e-4 on this
# trunk (against 4e-5 in `swin_calib_dense`, which leads at every epoch by ~0.011 but moves two other
# flags and so is directional only). Holding it is what makes this a MATCHED control: `swin_supervised`
# and this arm differ in opened depth and its decay, and in nothing else. A rate sweep on top would
# answer neither question.
#
# control  swin_supervised = 0.8331 — same trunk, same 384 px read, same backbone rate, same 12 epochs,
#          same `$RECIPE`. The cleanest matched control this lane has ever had.
# cost     ~90 min at six open blocks; twelve open blocks doubles what must be retained for the backward
#          and the twelve include stage three at stride 16, so price it at ~2.5-3 h.
#
# PRE-REGISTERED. Priced as an INSTRUMENT, not as a seat play. A seat at swin's ~0.83 correlation needs
# +0.017 to +0.020 solo, and P(one recipe flag pair delivers that) is low. What it buys is the licence to
# re-read nine runs and three quoted floors.
#   LANDS — solo at or above 0.8531. The transplant WAS the deficit, and it clears the seat bar as well.
#     Re-read the whole lane: run the fraction-matched opening on cnxbase (6 -> 18) and vjepa2l (6 -> 12)
#     before anything else in this lane, and treat every non-DINOv2 number in the ledger as re-opened.
#   MOVES — solo 0.8431 to 0.8530. The recipe matters and the count is wrong, but not by enough to seat.
#     Same follow-up arms, priced as instruments rather than as members.
#   FLAT — solo 0.8231 to 0.8430, within 0.010 either way of the control. Opened fraction is not the
#     deficit. The three unconverged runs are then the only recipe suspicion left, and they are cheap.
#   REFUTES — solo at or below 0.8230. The DINOv2 count beats the DINOv2 fraction on this trunk, which
#     closes the recipe explanation and leaves architecture and pretraining. The lane's remaining play is
#     then feature distillation (knee-8009) and nothing else.
#
# bead: knee-23wa
#
# deliberate: --unfreeze-last (12 of swinbase's 24 blocks — the whole point of the arm)
# deliberate: --layer-decay   (0.7, the value band_dense_llrd measured; it travels WITH the opening)
# deliberate: --variant       (swinbase, the trunk that already carries the lane's diversity)
# deliberate: --train-size    (384, pinned by the window tables — see swin_supervised.sh, do not sweep)
# deliberate: --backbone-lr   (1e-4, HELD at the control's value so the arm stays matched)
# deliberate: --batch
# deliberate: --accumulate
# deliberate: --vram-gb       (9, not the control's 8 — twice the open blocks must be retained; the value
#                              is what `arm_check`'s own reserve says this recipe needs, not a guess)
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train swin_open_half \
  $RECIPE --epochs 12 --batch 1 --accumulate 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant swinbase --backbone-lr 1e-4 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 384 \
  --unfreeze-last 12 --layer-decay 0.7 \
  --vram-gb 9 \
  --labels "$LABELS" --out "$PROCESSED/swin_open_half_s2026.pt"
