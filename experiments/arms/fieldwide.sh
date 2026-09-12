#!/usr/bin/env bash
# fieldwide.sh -- the first arm to move the field-of-view jitter off the value every member on disk shares.
#
# bead: knee-r1w4a
# reading: fieldwide_s2026.holdout.npz
#
# WHY THIS IS NOT augmentation AGAIN. `intensity_augment` and `aug_long` are refuted at twelve epochs
# and at twenty-four, and `mixup224`/`mixup224_strong` are flat at two doses -- but every one of those
# is INTENSITY-space (gamma, blur, contrast, noise) on a modality whose findings are geometric. The
# ledger says so itself at experiments/README.md:1378: "WHAT THIS DOES NOT KILL: augmentation of a KIND
# this repository does not have... Nothing has tested a geometric or slice-space perturbation." Nothing
# still has. --scale-jitter was hardcoded at 0.08 until knee-su2sw and --padding-mode at "border" until
# knee-10f10, so no member in the pool of 296 has ever seen either vary. This is a first read.
#
# THE TWO FLAGS ARE ONE LEVER, NOT A CONFOUND. AugmentConfig.padding_mode's own docstring: the border
# rule "was never an argument about EXTENT, and above scale 1 the whole margin is out of frame, so a
# zoom-OUT fabricates tissue by replicating one edge row across it". Widening the jitter under "border"
# would spend half its draws on fabricated margin, which tests the padding rule and not the field. They
# move together because separately neither is the mechanism.
#
# WHY IT COULD MATTER WHEN A NEW MEMBER GENERALLY CANNOT. knee-gcgbk has ELEVEN distinct kernels at
# exactly 0.944 and retired reseating; knee-0v8yz says only member quality at count 1 has ever moved the
# board 0.005+. Diversity BETWEEN members cannot repair a bias ALL of them carry, and this is one:
# --scale-jitter's own help records the 11-member readout of 2026-09-09 finding the generalisation toll
# FLAT across the pool at -0.0336 +/- 0.0038, with the single widened-field member paying least. A blind
# spot every member shares cannot appear as variance between them. The board is not our dataset.
#
# THE READING IS CROSS-SITE, NOT OUR OWN HOLDOUT, AND THAT IS THE POINT. holdout909 is drawn from the
# same field distribution as training, so it is structurally blind to this lever and a flat there
# refutes nothing. The foreign sites are the ruler that can see it: D206 section 5 measured their span
# 20-50% wider than ours, which is the mismatch the widened jitter is supposed to cover.
#
# PRE-REGISTERED BANDS. Control is `labelwise_drop_s2026`, one variable away: 0.8649 midpoint / 0.8984
# ledger75 at epoch 10 of 12. Report both rulers, name the cut (knee-f9aao).
#
#   OWN HOLDOUT (the safety check, not the reading)
#     >= 0.8600   intact. Proceed to the cross-site read whatever the sign.
#     <  0.8550   the widened field is destroying in-distribution signal. Lane dies here; do not
#                 rescue it on a cross-site number, because a member this much worse cannot seat.
#
# ADDENDUM 2026-09-12T14:57Z, written while the arm runs, TIGHTENING the reading and moving no band.
# The -0.0336 +/- 0.0038 is a pool of ELEVEN members and the control is not one of them, so reading
# fieldwide against that mean prices it against eleven confounds at once. `foreign_readout` takes
# --checkpoints as nargs, so the reading is BOTH members in ONE invocation -- one split, one pass, one
# foreign decode -- and the number that decides is `decode_cost(fieldwide) - decode_cost(labelwise_drop)`.
# The pool mean stays as the SCALE that sets the bands below (sd 0.0038 is what makes +0.0130 mean
# something), never as the comparator.
#
#   CROSS-SITE TOLL vs the CONTROL's decode_cost measured in the same pass; bands scaled by the pool sd
#   (the actual reading)
#     >= +0.0130 better (toll shallower than -0.0206)  the shared blind spot is REAL and partly closed.
#                 This is the one result that earns a submission slot on its own, and it re-opens the
#                 whole augmentation axis on the geometric side.
#     +0.0050 to +0.0129  directional only. Seat for diversity, do not re-open the axis on it.
#     -0.0049 to +0.0049  flat on the ONE ruler built to see it. The geometric side closes too, and
#                 augmentation is retired on this corpus entire rather than on the intensity half.
#     <= -0.0050  widening transfers WORSE. Then 0.08 is not an accident of the public baseline, the
#                 cached crop is the binding term, and the next arm is a wider --crop-mm cache rather
#                 than a wider jitter under the one we have.
#
# PRICE: ~0.95 h (280 s/epoch x 12), no submission slot to run it.
# EXPECTED GAIN: P(clears +0.0130 cross-site) ~ 0.30. Priced against the alternatives standing today --
#        a MedicalNet fine-tune is 4-6 h to close 0.067 of frozen gap before it can even reach a
#        +0.0040 seat bar from -0.0028, and every other named axis (foreign mounts, volumetric, SSL,
#        external data, relabelling) is closed. This is the best wall-clock-per-plausible-gain on the
#        board and it attacks the term the offline ruler is blind to, which is where a 0.011 board gap
#        that eleven distinct kernels cannot move is most likely to be hiding.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train fieldwide \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --stochastic-depth 0.05 \
  --scale-jitter 0.25 --padding-mode reflection \
  --labels "$LABELS" --out "$MODELS/fieldwide_s2026.pt"
