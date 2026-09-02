#!/usr/bin/env bash
# How far up the encoder rate goes before it stops paying. Two rungs above the one that just worked.
#
# WHAT IS ALREADY KNOWN. 8e-6 (the inherited 125:1 ratio to the head) scores 0.8457 under whole-stack;
# 4e-5 scores 0.8525, +0.0068 — over the floor, at 336 px, and equal to the 392 px member that was
# credited to resolution until 2026-08-18. Every rate we have ever run is one of those two, so the
# curve has exactly two points and the second is the highest. NOTHING SAYS 4e-5 IS THE PEAK; it says
# 5x beat 1x, which is where a sweep starts rather than stops.
#
# 1e-4 and 2e-4, i.e. 12.5x and 25x the inherited rate. Two rungs because one more point cannot tell a
# plateau from a peak: flat-then-down and up-then-down look identical with a single sample, and the
# whole question is which side of the maximum 4e-5 sits on.
#
# THE PREDICTED FAILURE IS CATASTROPHIC FORGETTING, and it has a visible signature: a self-supervised
# encoder driven too hard loses the pretrained features early, so the arm trails from epoch 1 or 2 and
# never recovers, rather than converging to a slightly worse number. If that is what the logs show, the
# reading is 'past the edge', not 'rate is neutral' — and the next rung to try is between 4e-5 and here.
#
# control  wsfold0_e6 = 0.8457 at 8e-6, and lr4e-5_ws_s2026 = 0.8525 at 4e-5. Everything else — 336 px,
#          12 epochs, whole-stack, max slice-pooling, the shared recipe — is held.
# risk     one training slot per rung, and both checkpoints can only join the pool on merit.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

BASE=($RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT --labels "$LABELS")

run_train rate_1e-4 "${BASE[@]}" --backbone-lr 1e-4 --out "$MODELS/lr1e-4_ws_s2026.pt"
run_train rate_2e-4 "${BASE[@]}" --backbone-lr 2e-4 --out "$MODELS/lr2e-4_ws_s2026.pt"
