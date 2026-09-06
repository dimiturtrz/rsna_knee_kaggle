#!/usr/bin/env bash
# chainfour.sh — knee-orjq's question asked from the OTHER END: put the whole chain ON the
# strongest single term, instead of putting single terms on top of it.
#
# bead: knee-jbum4
# reading: logs/chainfour_ledger.log
#
# WHY THIS IS NOT A REPEAT OF knee-orjq. That work took the calibrated dense base (0.8634) and added
# a SECOND term seven times; all seven landed at or below it. This arm takes the same base and adds
# THREE terms at once. The two designs test different quantities: one link against a saturated base,
# versus a chain worth 2x the floor against a saturated base. blendema224 already cost this project
# one arm for exactly that distinction, and the fix was to make the tested quantity the SUM.
#
# CONTROLS, both on disk, matched on every flag except the terms under test:
#   band_dense_hold_s2026   0.8573  — the bare dense base
#   calib_dense_hold_s2026  0.8634  — the bare base plus --calibrated-targets ALONE (+0.0061)
#   chaindense_s2026        (read it from logs/chaindense_ledger.log before arming this)
#     — the bare base plus the three 224-corner terms, no calibration
#
# THE FOUR TERMS, sizes measured where each was measured:
#   --calibrated-targets                     +0.0061  (at the dense corner, against band_dense_hold)
#   blended labels (--labels blend_three_v1) +0.0073  (at the 224 corner, against native224)
#   --ema                                    +0.0042  (at the 224 corner)
#   --slice-pooling labelwise                +0.0015  (at the 224 corner)
#   arithmetic sum from 0.8573               +0.0191 -> 0.8764
#   at the 0.8089 transfer measured at 224   +0.0155 -> 0.8728
# Either prediction clears the pool best (0.8642) by more than two floors, which is the first time
# any recipe on this project has been able to say that before running.
#
# PRE-REGISTERED, BEFORE THE NUMBER. The control that decides it is calib_dense_hold 0.8634 —
# the strongest single term at this corner — and the floor is 0.0062:
#   >= 0.8696   THE CHAIN SUMS ONTO A SATURATED BASE. Composition is not capped by base quality;
#               knee-orjq's 7/7 was about the SIZE of the second term, not about saturation. This
#               would be the best member the project owns by a readable margin and would put the
#               +0.015 route back on knobs.
#   0.8634-0.8696  THE CHAIN REACHES THE SATURATED BASE AND STOPS. Three terms buy what one term
#               already bought and nothing more — the terms substitute rather than add at the top.
#               Composition still builds a DIVERSE member cheaply; it does not beat a strong one.
#   < 0.8634    knee-orjq REPRODUCED FROM THE OTHER END. A chain worth two floors, added to a
#               saturated base, goes DOWN. That is base quality capping composition, it closes the
#               knob route to +0.015, and it is the answer letter 4 asked Kimi for.
#   <= 0.8450   Harness first, not a result. Verify the blend table loaded, --ema and
#               --calibrated-targets both fired, and the recipe printed 336 px / 20 slices /
#               band 0.1-0.9. Four non-negative terms cannot subtract three floors.
#
# GATING. Run this whatever chaindense reads — the two arms answer different halves and this one's
# controls are all on disk independently. If chaindense landed in its own top branch the prediction
# above holds as written; if it landed lower, the arithmetic prediction drops by the shortfall and
# the BRANCHES DO NOT MOVE, because they are set against calib_dense_hold and not against chaindense.
#
# COST. ~75 min of card. Zero submission slots.
#
# deliberate: --calibrated-targets (term one; the term knee-orjq found strongest at this corner)
# deliberate: --labels blend_three_v1 (term two)
# deliberate: --ema (term three)
# deliberate: --slice-pooling labelwise (term four; substitutes the control's max)
# deliberate: --hold-annotations (matched to both controls; buys the gold ruler in the same run)
# deliberate: --eval-labels report_labels_v2.csv (the proxy must not move with the arm)
# deliberate: --slices 20 --band-low 0.1 --band-high 0.9 --image-size 336 (the controls' corner, held)
# deliberate: every other flag — they are band_dense_hold's and may not be touched in this arm.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
BLEND="$PROCESSED/labels/blend_three_v1.csv"
require "$BLEND"
require "$LABELS"

run_train chainfour \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --image-size 336 --hold-annotations --ema --calibrated-targets \
  --labels "$BLEND" --eval-labels "$LABELS" --out "$MODELS/chainfour_s2026.pt"
