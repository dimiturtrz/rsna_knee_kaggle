#!/usr/bin/env bash
# bead: knee-x580
# band_dense's recipe with --mixup 0.2 — the first LABEL-SPACE regulariser this project has run.
#
# WHY THIS AXIS. Three independent runs now show the same signature. band_dense_long ran 24 epochs as a
# single variable and peaked at epoch 6 (0.8587), sliding to 0.8405 by epoch 23 while its training loss
# halved 0.85 -> 0.4536: a 0.0182 peak-to-finish decline against a 0.0062 floor. half_data and
# quarter_data price data volume as the only axis on this board that has ever cleared the bar, and the
# fitted slope is concave, so the next doubling of the corpus is worth ~+0.009 and every remaining
# doubling ~+0.017. band_span_hold, finished tonight, peaked at epoch 7 (0.8493) and fell to 0.8464 by
# epoch 11 while its loss fell 0.6834 -> 0.5700 — sub-floor at -0.0029 and directional only, but the
# same shape a third time. A classifier that memorises at twelve epochs has no use for more slices,
# more pixels, more members or more epochs, which is why eleven measured decode-family axes all read
# inside the floor.
#
# WHY NOT THE REGULARISER WE ALREADY HAD. aug_long turned the intensity block on as a single variable
# over the 24-epoch schedule and read -0.0076. Its finding is the informative kind and it is the reason
# THIS arm is not a repeat: the regulariser demonstrably FIRED (train loss higher at every epoch past 4)
# and the peak-to-finish decline was UNCHANGED at -0.0169 against the control's -0.0182. Appearance
# augmentation leaves the pair (this study, this label) intact, so a memorised answer still satisfies
# it. A convex combination moves the TARGET with the pixels, which makes the memorised answer wrong
# rather than merely harder to recognise. Different mechanism, not a stronger dose of the dead one.
#
# WHY 0.2 AND NOT 1.0. Beta(a, a) at a = 1.0 draws the weight uniformly and blends half the batch near
# 50/50; on 1,763 training studies that trades away more signal than it buys. 0.1-0.4 is the customary
# range for a classifier this size and 0.2 sits in it. The draw is folded to the dominant half, so every
# row stays recognisably its own study. If this arm reads flat, 0.4 is the follow-up, not a different
# axis — a null at one concentration is not a null for the family.
#
# control  band_dense = 0.8578 at epoch 8 of 12, ALREADY MEASURED, so this arm buys the pair for the
#          cost of one half. Same cache (336px_20sl_0.1-0.9band), same 12 epochs, same schedule, same
#          --backbone-lr 4e-5. The single variable is --mixup.
# cost     ~40 min of card, one arm, no cache build. Blending is one scalar multiply over tensors the
#          loop already holds; step time should be band_dense's to the second, and if it is not, the
#          seam is wrong rather than the method expensive.
#
# CONFOUND, NAMED BEFORE THE RUN. --report-weight is off here and must stay off: ReportSupervision is
# handed the batch's POSITIONS rather than a table of rows, so it alone reads un-blended under --mixup.
# The label, ranking and teacher terms all read blended rows. See kneemri/training/mixup.py.
#
# PREDICTION, written before the run.
#   CONFIRMED if the peak clears 0.8578 by more than the 0.0062 floor (>= 0.8640) OR the peak epoch
#   moves LATE (8 -> 10 or later) with the mean at or above the control. The second clause matters as
#   much as the first: a regulariser that buys the same score while the curve stops declining is
#   licensing a LONGER schedule, and the 24-epoch cell is where the 0.0182 was measured.
#   REFUTED if the peak lands below 0.8578 by more than the floor, or if the peak epoch moves EARLIER —
#   an earlier peak under a regulariser means the blend destroyed signal rather than memorisation, and
#   that would be a fact about this corpus (small, soft-labelled, twelve correlated findings) and not
#   about mixup.
#   THE THIRD OUTCOME, and the one aug_long landed in: the train loss rises (the blend fired) and the
#   peak-to-finish decline is UNCHANGED. That would say the memorisation is not in the study-to-label
#   pair at all, and would move the remaining suspects to supervision volume — i.e. to knee-ombx, which
#   is owner-blocked, and would be worth knowing for exactly that reason.
# WHY PIXEL SPACE AND NOT THE EMBEDDING, decided before the run rather than after it fails. A convex
# combination of two knees is not a knee: this arm blends anatomy that no scanner could produce, and the
# findings this task turns on are focal and few-voxel. Manifold mixup — the same convex combination taken
# at a hidden representation instead — is the standard answer to exactly that objection, is a
# generalisation of this arm rather than a rival to it, and is reported effective on prostate MR (+2-4
# Dice) and on whole-brain segmentation under ROAM's random-layer variant. It is ALSO a one-line change at
# a seam this commit already opened.
#
# It is not what runs first, because input mixup is the CHEAPER INFORMATION and both of its failure modes
# name manifold mixup as the successor for a DIFFERENT reason, which is what makes running it first
# decisive rather than merely traditional:
#   peak FALLS  -> blending anatomy destroyed the focal findings. Manifold mixup is the fix, because it
#                  never shows the encoder an impossible knee.
#   peak FLAT, train loss ROSE -> the blend fired and the memorisation did not care (aug_long's outcome).
#                  Manifold mixup is the fix for the OPPOSITE reason: the perturbation never reached the
#                  representation where the memorising happens.
# So this arm separates two mechanisms that a manifold-first run would confound, and it does it on code
# that is already written, gated and committed.
# deliberate: --mixup 0.2 (the arm's thesis)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train mixup_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --mixup 0.2 \
  --labels "$LABELS" --out "$PROCESSED/mixup_dense_s2026.pt"
