#!/usr/bin/env bash
# calib_dense.sh — the label calibration on the best single member we own, not on the thinner one.
#
# THIS IS STAGE TWO, AND IT IS NOT THE ARM TO LAUNCH FIRST — 2026-08-26. `calib_dense_hold.sh` supersedes
# it as the MEASUREMENT: unheld, `annotated_readings` REFUSES this checkpoint, so this arm cannot report
# the second ruler that knee-9198 requires of a label-side arm, and the one ruler it can report is the one
# knee-bs4l measured as biased against this exact change by 1.5x the noise floor. Run the held pair first
# (`band_dense_hold` then `calib_dense_hold`). This arm becomes the SEATING refit — unheld, on all 4407 —
# and only if the held pair says the corner is real. Launching it first buys a number nobody can read.
#
# WHY THIS ARM EXISTS. Three levers are now measured separately on the same control (lr4e-5_ws,
# 0.8525): span 0.10-0.90 is +0.0039, density 20-over-the-span is +0.0014, and the two are ADDITIVE on
# the mean with an interaction of ~0.0001 (band_dense, 0.8578, is the corner that carries both). The
# third lever, calibration, is being measured on the SPAN corner by calib_span. This arm is the corner
# that carries all three, and it is the only untested cell of the cube.
#
# WHY IT IS NOT A KNOB. Nothing here changes a magnitude; each lever answers a different question —
# span asks what anatomy is covered, density asks how the pooled budget is split inside it, calibration
# asks whether what we tell the model about a cell is true. knee-jk3 leaves the label axis standing
# after the decode-geometry family closed at about +0.005 total, and the whole point of an additive
# structure is that it composes to something the individual arms cannot reach: 0.8525 + 0.0039 + 0.0014
# is already 0.8578 measured, and a calibration term of the size calib_span is testing would put this
# corner at the first member this project has produced above 0.860.
#
# control  band_dense = 0.8578 (336 px, 20 slices, 0.10-0.90, whole-stack, 4e-5, 12 epochs). SINGLE
#          VARIABLE: that invocation with --calibrated-targets added and nothing else touched. The
#          second reading is against calib_span, which holds the calibration and drops the density.
# cost     identical to band_dense — the calibration is a load-time remap of the training table. One
#          12-epoch fit at 1.25x the 12-slice step time.
#
# PREDICTION, written before the run. If calibration is a THIRD independent lever, the delta against
# band_dense is the same delta calib_span measures against band_span, within the floor — that is the
# additivity claim, and it is what makes this corner worth a slot rather than a guess. The per-label
# clause is inherited unchanged from calib_span so it is not re-fitted: the gain concentrates on MCL
# and Effusion, the two labels most skewed toward the 0.68 level, and PF OA barely moves.
#
# THE REFUTATION CONDITION. A mean at or below band_dense WITH MCL and Effusion flat says calibration
# and density are substitutes — the depth_regions_span shape a third time — and retires stacking the
# label axis on the decode axis outright rather than leaving a fourth cell to run. A gain that appears
# uniformly across all twelve labels refutes the MECHANISM while the mean still reads as a win, which
# is what the per-label read exists to catch.
#
# bead: knee-ep74
#
# THE BLOCKING CONDITION HAS BEEN DISCHARGED — 2026-08-26. This header used to read "DO NOT LAUNCH THIS
# BEFORE calib_span LANDS", and calib_span landed at -0.0020, which fired the condition and closed the
# arm unlaunched on 2026-08-19. That number no longer stands. knee-bs4l re-measured the same single
# variable on 2026-08-25 as a matched pair with --hold-annotations on BOTH halves (calib_span_hold vs
# band_span_hold) and read MEAN +0.0072 on the proxy and +0.0129 over the 58 annotation cells. The sign
# flipped by 0.0092, which is 1.5x the 0.0062 paired floor. The premise the block rested on is gone.
#
# WHAT THAT DOES NOT LICENCE. The +0.0072 was measured in the HELD regime. This arm and its control
# hold nothing, which is the UNHELD regime — the one that read -0.0020. So this arm does not inherit
# the +0.0072 prior, and a null here is not a contradiction of knee-bs4l. It is run unheld on purpose:
# a member we would actually seat trains on all 4407, so the unheld number is the one that decides
# whether this becomes a member. knee-p9oc is the named harness suspect — the remap's constants are
# POOLED across findings that do not agree (the 0.68 level is assigned 0.378 while the twelve findings
# run 0.000 to 0.778), so a null is evidence about the pooled estimate as much as about calibration.
#
# NO --vram-gb: DeviceBudget derives the cap from --slices — 20 slices reserve 28.26 GB, which is what
# band_dense's own run reproduced after its hand-typed 26 OOMed.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train calib_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets \
  --labels "$LABELS" --out "$PROCESSED/calib_dense_s2026.pt"
