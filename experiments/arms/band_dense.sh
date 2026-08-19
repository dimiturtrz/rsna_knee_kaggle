#!/usr/bin/env bash
# The widened band at RESTORED central density: 0.10-0.90, 20 slices.
#
# WHAT band_wide LEFT OPEN. Widening the band to 0.10-0.90 at constant mm-per-slice bought the
# collaterals (MCL +0.0133, ACL +0.0071) and peripheral bone (Fracture +0.0074, Contusion +0.0072) and
# cost the one central-compartment finding class: Medial OA -0.0088, Lateral -0.0013, PF -0.0002. The
# interpretation names dilution as the plausible mechanism and says plainly that nothing measures it —
# four of sixteen slices are new peripheral content, max-pooled, so the central evidence holds a
# smaller share of the pooled representation than it did at twelve.
#
# WHY THIS ARM CAN SEPARATE IT. 20 slices over the 0.8 span puts SIXTEEN slices inside the old
# 0.20-0.80 band, against twelve for the control and twelve for band_wide. The periphery is unchanged
# from band_wide. So the only thing that moves is how much of the pooled representation the centre
# gets.
#
# WHY DENSITY IS NOT THE CONFOUND. knee-ezz already priced it: 20 cached slices against 12 under max
# pooling — which holds physical thickness constant — is -0.0001. Slice density inside the band is
# measured flat, so an OA recovery here belongs to the dilution mechanism and not to seeing more
# slices.
#
# control  band_wide = 0.8553 (336 px, 16 slices, 0.10-0.90, whole-stack, 4e-5), and behind it
#          rate_wholestack = 0.8525 at 12 slices in the old band.
# cost     1.25x band_wide's step time (120 images a study against 96); 12 epochs, one cache build.
#
# WHY THIS ARM NAMES NO CAP, AND WHAT IT USED TO NAME. It ran at --vram-gb 28 after its first launch
# OOMed at 26: band_wide peaked at 21.6 GB reserved for 96 images a study, and 120 needs about 27.
# DeviceBudget now derives that from --slices alone — 20 slices reserve 28.26 GB — so the hand-typed
# number is gone and the measurement above is what the derivation reproduces. The BATCH is deliberately
# not the knob that pays for it: the ranking term forms its pairs inside the batch, so shrinking it
# would change the loss at the same time as the coverage and this arm would answer nothing.
#
# PREDICTION, written before the run. If dilution is the mechanism: Medial OA recovers toward zero
# against rate_wholestack while MCL/ACL/Fracture/Contusion keep their band_wide gains, and the mean
# lands at or above 0.8553. If OA stays negative with the collaterals still positive, dilution is
# REFUTED and the OA loss belongs to something else — the most likely alternative being that the
# outer slices are not merely diluting but actively mislabel-ing OA from tissue that is not joint.
# A mean below the control's 0.8525 would say the extra density costs what knee-ezz says it does not,
# and that arm would have to be re-read before this one is believed.
# RESULT 0.8578 at epoch 8 of 12 — +0.0053 over the control, +0.0025 over band_wide, best single
# member the project has. The prediction is a PARTIAL confirmation: the OA clause hit (Medial OA
# -0.0088 -> +0.0034 against the control and +0.0122 against band_wide with the interval clearing
# zero; PF OA -0.0002 -> +0.0192, resolving against both; Synovitis +0.0218, resolving) and the mean
# cleared 0.8553, but the collateral clause missed — MCL +0.0133 -> +0.0022 and ACL +0.0071 ->
# +0.0034, i.e. -0.0111 and -0.0037 measured directly against band_wide. So dilution is confirmed and
# SYMMETRIC: the pooled budget is what moves, and each arm buys one end at the other's expense.
# The two are complementary, not one superseding the other — paired they read 0.8633, +0.0055 over the
# better of them, and greedy over 23 same-split members now picks band_dense first and band_wide
# second (plateau 0.8661 at seven, up from 0.8646 at six).
# See interpretations/kneemri/2026-08-19_density-inside-the-band-trades-centre-for-edge.md.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$PROCESSED/band_dense_s2026.pt"
