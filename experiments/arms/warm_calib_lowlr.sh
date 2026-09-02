#!/usr/bin/env bash
# warm_calib_lowlr.sh — the follow-up warm_calib_dense pre-registered, against the suspect it named.
#
# THE RESULT THIS ANSWERS, run 2026-08-27 and first read 2026-08-30. `warm_calib_dense` warm-started the
# best solo member this project owns and LOST 0.0072:
#
#   calib_dense_hold (cold, 4e-5)  .7515 .8007 .8280 .8438 .8544 .8554 [.8634] .8620 .8607 .8596 .8603 .8598
#   warm_calib_dense (warm, 4e-5)  .8503 .8469 .8555 .8510 .8549 .8526 [.8562] .8529 .8540 .8518 .8515 .8513
#
# Read the two shapes rather than the two peaks, because they say opposite things.
#
# THE TIME CLAIM IS VALIDATED, and by more than its own docstring asked for. The warm run's FIRST epoch
# reads 0.8503 — a score the cold run does not reach until epoch 4. `fitted_encoder.py` predicted the
# saving as "the first eight epochs, the ones that only re-learn what a member on disk already knows";
# measured, one warm epoch is worth about five cold ones on this recipe. That half of knee-wbwq is settled
# and needs no further arm.
#
# THE QUALITY CLAIM IS REFUTED AT THIS RATE, AND ONLY AT THIS RATE — which is what this arm exists for.
# The warm run's epoch 1 DROPS, 0.8503 to 0.8469, and it then oscillates in a 0.847-0.856 band for eleven
# epochs without ever reaching the parent's 0.8634. A converged trunk does not wander like that under a
# rate it has already annealed through; it wanders when something pushes it off its optimum first.
#
# `warm_calib_dense.sh` named that mechanism in advance, before any of this was measured: "one_cycle ramps
# the backbone rate above 4e-5 before annealing, and a converged trunk can be walked off its optimum by
# the ramp. That outcome sends the next arm to a lower --backbone-lr, not to the graveyard." A warm
# restart does NOT resume the optimiser — the schedule starts over and ramps to its peak near epoch 2 —
# so the parent's carefully annealed weights are hit with a rising rate on epoch 1. The drop is on epoch 1.
# The prediction and the observation are the same epoch. This arm is that pre-registered follow-up, and it
# is being run because the branch fired, not because a number was disappointing.
#
# SINGLE VARIABLE, and the control is a run that already exists. Against `warm_calib_dense` exactly ONE
# flag moves — `--backbone-lr`, 4e-5 to 1e-5. Same encoder file, same twelve epochs, same recipe, same
# held split, same targets. So the difference between the two numbers is the ramp height and nothing else,
# and no fresh control arm has to be run to read it. A quarter is chosen rather than fitted: it is the
# largest reduction that still leaves the trunk able to move, and the same quarter-of-parent ratio
# `cnx_warm_anneal` uses, so the two arms answer the ramp question on a ViT and a CNN at one ratio.
#
# cost   ~40 min at 12 epochs, 0 slots. The control cost nothing — it is on disk.
#
# PRE-REGISTERED, before the run. Two references, and they mean different things:
#   the WARM control  0.8562 — was the ramp the cause?
#   the COLD parent   0.8634 — is warm starting worth doing for QUALITY at all?
#
#   THE RAMP WAS THE CAUSE, AND WARM STARTING WINS — >= 0.8644. Beats the cold parent, which makes this
#     the best single member this project has produced, and it makes warm-starting-at-a-reduced-rate the
#     default for every repeat arm rather than a curiosity. knee-wbwq's quality claim is then live, and
#     the 36-recipe-skeleton waste it documents becomes recoverable.
#   THE RAMP WAS THE CAUSE, QUALITY IS STILL FLAT — 0.8572 to 0.8643. The ramp explains the loss (this
#     clears the warm control by more than the 0.0010 board resolution) but a warm start still does not
#     BEAT a cold one. Then warm starting is a pure TIME mechanism, correctly priced, and it should
#     replace repeat arms on cost alone. That is a real and usable result, not a null.
#   THE RAMP WAS NOT THE CAUSE — 0.8500 to 0.8571. Flat against the warm control. The loss is not the
#     schedule, and the honest reading is that a fully annealed trunk has no usable capacity left for a
#     second cycle at ANY rate. knee-wbwq's quality claim closes; the time claim stands untouched.
#   WORSE THAN THE WARM CONTROL — <= 0.8499. Then lowering the rate hurt, which points at the head
#     rather than the trunk: `--head-lr` is unchanged at 1e-3 and the head is fitted from scratch either
#     way, so a trunk too frozen to follow a fresh head is the next suspect. Do NOT read this as a
#     verdict on warm starting; it is a verdict on this pairing of rates.
#
# WHAT NO BRANCH LICENSES. A warm-started member is correlated with its parent by construction, so no
# outcome here buys SEAT diversity — knee-wbwq is explicit that decode-path diversity is the axis that
# transfers at ~2.4x the board slope and warm starts do not touch it. Any seat claim needs `record_mix`
# against the roster afterwards. And none of this transfers to a NON-DINO trunk on its own:
# `cnx_warm_anneal` asks the same question of a convolutional trunk because the answer is not assumed.
#
# bead: knee-wbwq
#
# deliberate: --backbone-lr
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"
require "$MODELS/encoders/calib_dense_hold.enc.pt"

run_train warm_calib_lowlr \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 1e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --encoder-weights "$MODELS/encoders/calib_dense_hold.enc.pt" \
  --labels "$LABELS" --out "$MODELS/warm_calib_lowlr_s2026.pt"
