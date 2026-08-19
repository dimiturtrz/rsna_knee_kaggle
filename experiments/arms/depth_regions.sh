#!/usr/bin/env bash
# One maximum per depth REGION instead of one over the whole band: --slice-regions 4.
#
# WHAT THE BAND ARMS FOUND. band_wide widened the band and bought the collateral ligaments at the
# central compartment's expense (MCL +0.0133, Medial OA -0.0088 against the control). band_dense
# restored the central density and bought the centre back at the collaterals' expense — measured
# directly against band_wide, which holds the periphery FIXED: PF OA +0.0194 [+0.0043,+0.0355] and
# Medial OA +0.0122 [+0.0004,+0.0239] both clearing zero, against MCL -0.0111, Contusion -0.0052,
# Fracture -0.0052. Neither arm changed the anatomy in the half it lost. What moved was that half's
# SHARE of one max-pooled representation.
#
# WHY THAT IS A DESIGN DEFECT AND NOT A TUNING PROBLEM. `slot_features` takes ONE amax over every
# slice in the band, so each feature channel is won by whatever is loudest anywhere along the depth,
# and the loudest thing is the largest structure. A collateral ligament three slices wide at the edge
# of the band competes for the same channel as the whole central compartment. No band choice fixes
# that — it only decides which end pays — and the two members we now ship are the same trade made
# twice in opposite directions. --slice-regions 4 takes the maximum WITHIN each quarter of the depth
# and averages the four, so every part of the band gets its own channel budget.
#
# WHY THIS IS THE CHEAPEST FORM OF THE IDEA. No parameters, no extra encoder passes, no cache: the
# region maximum is taken over features that are already computed, and at one region it is the
# historic pooling bit-for-bit (asserted in tests/unit/kneemri/models/model.py::test_over_depth). So
# it costs one arm to ask, against band_dense's own 0.8578 with every other flag identical.
#
# WHY 4. 20 slices over the 0.10-0.90 band is five slices and ~0.2 of depth per region, which puts
# the outer tenth that band_wide added inside the outermost region rather than spread across a split
# it shares with the centre. It is the coarsest split that separates the two ends the arms disagree
# about; 2 would leave each half still pooling its own periphery against its own centre.
#
# control  band_dense = 0.8578, SAME cache (336px, 20 slices, 0.10-0.90), same recipe, same rate.
# cost     identical to band_dense — this changes an aggregation over features already computed.
#
# PREDICTION, written before the run. If the pooled budget is the mechanism, this arm holds
# band_dense's central gains (PF OA, Medial OA, Synovitis) AND recovers band_wide's peripheral ones
# (MCL, Contusion, Fracture) at the same time, since neither is being spent against the other any
# more — so the per-label deltas against band_dense are positive exactly where band_wide beat it. A
# mean at or below 0.8578 with the collaterals still flat says the trade was never about the pooling,
# and the two band members are simply reading different anatomy well. A mean that RISES while the
# collaterals stay flat would say regions help for a reason this argument does not name, and would
# need the per-label table read before it is believed.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train depth_regions \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT --vram-gb 28 \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --slice-regions 4 \
  --labels "$LABELS" --out "$PROCESSED/depth_regions_s2026.pt"
