#!/usr/bin/env bash
# calibrated_targets.sh — train on what each report level is WORTH, not on what it was assigned.
#
# THE DEFECT. The reader emits five values and we trained on them as probabilities. Measured against
# the 57 fully-annotated studies in train.csv (696 cells, every label annotated), how often the
# radiologist actually agrees the finding is present:
#
#     assigned   measured   95% CI            n
#       0.08      0.041     [0.013, 0.068]   196
#       0.28      0.136     [0.085, 0.186]   177
#       0.68      0.378     [0.282, 0.474]    98
#       0.82      0.688     [0.611, 0.764]   141
#       0.94      0.958     [0.912, 1.004]    72
#
# FOUR OF FIVE fall outside their own interval, every one of them too high. The model is told 0.68 for
# cells that are positive 38% of the time, and 0.28 for unmentioned findings that are positive 14%.
#
# THE LEVELS THEMSELVES ARE SOUND, which is what makes this a calibration rather than a relabelling:
# agreement is monotone across a span of 0.58, and the reader's confidence column is CONSTANT per
# verdict, so the gradation is real per-study information rather than certainty noise.
#
# THE RULER DOES NOT MOVE. The calibration is applied to the TRAINING table only. Recalibrating the
# holdout too would raise the AUC by construction - the targets would simply agree with themselves
# more - which is the exact failure Targets.pair was written to prevent.
#
# NOR DOES IT LEAK. The rates are fitted on the 58 IMAGE-READ annotations; this arm is scored on the
# fold-0 holdout of the REPORT labels. Different rulers. What it does spend is some independence of
# the annotation-based check, which is stated rather than hidden.
#
# THE FALSIFIABLE PART: the gain must CONCENTRATE where the miscalibration is largest - the 0.68 level
# and the unmentioned-finding level, i.e. the labels whose positives are mostly low-grade. MCL
# (495/219/53) and Effusion (1599/910/157) are the two most skewed toward 0.68 and should move most;
# PF OA (543/658/790) is skewed high, is nearly calibrated already, and should barely move. A uniform
# shift means something else changed and the mechanism is not what moved it.
#
# control  wsfold0_e6 = 0.8457, identical recipe on the assigned values.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train "calibrated_targets" \
  $RECIPE --epochs 12 $PATIENCE \
  --slice-pooling max --whole-stack --calibrated-targets \
  $THROUGHPUT \
  --labels "$LABELS" --out "$PROCESSED/calibrated_targets_s2026.pt"
