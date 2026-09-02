#!/usr/bin/env bash
# calib_span.sh — the label calibration measured on the recipe we actually ship.
#
# WHY THIS CELL IS UNRUN, AND WHY IT IS NOT A KNOB. `calibrated_targets` scored 0.8495 against
# wsfold0_e6 = 0.8457 on 2026-08-16: +0.0038, directional, inside the 0.0062 floor. It was run at the
# operating point of that week — the DEFAULT backbone rate and the 0.20-0.80 band. Every arm since has
# moved: 4e-5 is the settled rate and the span axis closed on 0.10-0.90 with an interior optimum
# (band_span 0.8565 > lr4e-5_ws 0.8525 > band_full 0.8427). So the calibration has never been measured
# on the recipe the project ships, and the member sitting in the champion package was fitted on a
# recipe no later arm reads from.
#
# THE QUESTION IS SUBSTITUTION, which is the one this project keeps getting wrong in the same shape.
# Span and depth-region pooling both answer 'where along the stack does signal live', and stacking them
# cost -0.0049 — the second one applied was distraction, not addition. Calibration answers a DIFFERENT
# question (is what we tell the model about a cell true), so the prior is additive; but that prior is
# exactly what depth_regions_span had, and it lost. Measuring it is the only way to know, and knee-jk3
# names the label axis as the one thing its own logic leaves standing after the decode-geometry family
# came to about +0.005 in total.
#
# control  band_span = 0.8565 (336 px, 12 slices, 0.10-0.90, whole-stack, 4e-5, 12 epochs). SINGLE
#          VARIABLE: this is that invocation with --calibrated-targets added and nothing else touched.
# cost     identical to band_span — the calibration is a load-time remap of the training table, no
#          reader work, no new labels, no extra compute. One 12-epoch fit.
#
# THE RULER DOES NOT MOVE, for the same reason it did not for calibrated_targets: the remap is applied
# to the TRAINING table only. Recalibrating the holdout too would raise the AUC by construction.
#
# PREDICTION, written before the run and inherited from calibrated_targets so it is not re-fitted to
# whatever comes back: the gain must CONCENTRATE where the miscalibration is largest — the 0.68 level
# and the unmentioned-finding level, i.e. labels whose positives are mostly low-grade. MCL (495/219/53)
# and Effusion (1599/910/157) are the two most skewed toward 0.68 and should move most; PF OA
# (543/658/790) is skewed high, is nearly calibrated already, and should barely move.
#
# THE REFUTATION CONDITION is a mean at or below the control WITH MCL and Effusion flat — that reads as
# calibration and span being substitutes, the depth_regions_span shape again, and it retires stacking
# the label axis on the decode axis rather than leaving another cell to run. A uniform lift across all
# twelve refutes the MECHANISM while leaving the mean looking like a win, which is what the per-label
# read exists to catch.
#
# A NEW NAME, DELIBERATELY. calibrated_targets_s2026 is a member of the 0.928 champion package
# (comp/champion.json), so re-running under that name would overwrite a shipped result to answer a
# question about a different recipe.
#
# NO --vram-gb: DeviceBudget derives the cap from --slices (1.335 GB a slice plus 0.22, which every
# 12-slice arm confirms at 16.24 measured).
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train calib_span \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 --calibrated-targets \
  --labels "$LABELS" --out "$MODELS/calib_span_s2026.pt"
