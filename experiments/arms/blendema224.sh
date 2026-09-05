#!/usr/bin/env bash
# blendema224.sh — the stack question, asked again at the base knee-orjq explicitly left open.
#
# bead: knee-y8ies
# reading: logs/blendema224_ledger.log
#
# WHY THIS ARM EXISTS AT ALL, GIVEN THE STACK ROUTE WAS CLOSED. knee-orjq tested whether directional
# gains ADD and closed on 7/7 arms landing below their control at the CALIBRATED DENSE corner
# (calib_dense_hold 0.8634, 336 px, 20 slices, 0.1-0.9). Its own close records the limit: "a stack built
# from a DIFFERENT base has never been tried; this is a result about the calibrated corner
# specifically." It also records the counter-example that keeps the question alive —
# interpretations/kneemri/2026-08-27_aggregation-beats-input holds four aggregation flags agreeing on a
# sign for a real 0.8272 -> 0.8523 chain. So composition is not dead as a mechanism; it failed at one
# corner, in the shape of a local optimum, and the honest reading was always that the corner was
# saturated rather than that addition does not happen.
#
# THE BASE IS THE VARIABLE. This arm stacks at the 224 px / 12 slice / 0.2-0.8 corner, which is a
# different base in three ways that matter: it is not calibrated, it is not dense, and it is the corner
# where band224_hold measured the pixel axis flat on two independent rulers earlier today. It is also
# 3.2x cheaper per epoch, which is why the question can be re-asked at all — the same test at the dense
# corner cost 85 minutes and this one costs about twenty.
#
# THE TWO TERMS, both measured POSITIVE at this exact corner rather than transported into it:
#   --labels blend_three_v1  blendhold_s2026 0.8575 against native224_s2026 0.8502, +0.0073. The
#                            strongest 224 member this project owns and the strongest single term here.
#   --ema                    ema224_s2026 0.8544 against the same native224 control, +0.0042. Weight
#                            averaging, and the delta is internally matched — same corner, same control.
# Mechanically independent: one changes what the targets SAY, the other changes which weights are KEPT.
# That independence is the precondition for a sum rather than a substitution, and it is the same pairing
# stackfit made at the dense corner, which is deliberate — re-asking the same question at a new base is
# only informative if the terms are the same ones.
#
# ONE VARIABLE against blendhold: --ema. Every other flag is byte-identical to that arm, --hold-annotations
# and the eval-labels split included, so the second ruler comes free in the same run.
#
# THE ADDITIVE PREDICTION, and it is a subtraction with no correction terms:
#     predicted = blendhold + (ema224 - native224) = 0.8575 + 0.0042 = 0.8617
#
# PRE-REGISTERED, BEFORE THE NUMBER, against the 0.0062 paired floor.
#   >= 0.8679   SUPER-ADDITIVE. Suspect it before believing it — at this corner nothing predicts the two
#               terms reinforcing, and the first suspect is the EMA decay interacting with the blended
#               targets rather than a real synergy. Demand a second seed before it is quoted anywhere.
#   0.8555-0.8679  ADDITIVE. The result the 0.953 plan needs and the one this arm exists for: it says
#               knee-orjq measured a SATURATED CORNER, not a law, and it licenses stacking as a method at
#               the cheap base where a chain of terms is actually affordable. The immediate follow-up is
#               a third term (labelwise pooling, +0.0015 here) on the same base, which is another twenty
#               minutes and turns two points into a slope.
#   0.8513-0.8555  BETWEEN blendhold and the prediction: the terms SUBSTITUTE PARTIALLY. Weaker than
#               additive and still not the dense corner's shape — read it as the blend already carrying
#               part of what EMA smooths, and price any future stack at half its arithmetic sum.
#   <= 0.8513   SUBSTITUTES NOT COMPLEMENTS, below the control by more than the floor. This is the
#               dense corner's 7/7 shape reproduced at a second, structurally different base, and TWO
#               bases agreeing closes composition as a route to 0.953 for this project rather than for
#               one corner of it. That is the most valuable outcome the arm can produce and it is why it
#               is worth running even at a modest prior.
#   <= 0.8400   Suspect the harness first, not the mechanism. Check that --ema fired at all (the run's
#               printed recipe must name it) and that the blend labels loaded, before writing anything.
#
# WHAT IT CANNOT SAY. Two points do not fit a composition rule, and a positive here is evidence about
# THESE two terms at THIS base, not a law about the next three. Each further term still needs its own
# single at this corner before it is stacked — which is exactly the discipline knee-orjq was closed for
# skipping, and it is affordable here in a way it was not at 336.
#
# COST. ~90 s an epoch at 224 px / 12 slices, so about twenty minutes for twelve epochs. Zero submission
# slots. P(clears the 1.5% bar as a MEMBER) is low, around 0.05, and that is not the case for the arm.
# The case is that it is a DECISION arm on the only route the champion's own principles leave open:
# P(it moves the stack question) is high, twenty minutes buys either a licence or a second refutation,
# and a refutation at a second base is worth as much as a licence.
#
# control  blendhold_s2026 = 0.8575, byte-identical recipe without --ema.
#
# deliberate: --ema (the arm's thesis)
# deliberate: --labels blend_three_v1 (the second term, held from blendhold and not under test)
# deliberate: --hold-annotations (matched to blendhold; without it the gold ruler scores training cells)
# deliberate: --eval-labels report_labels_v2.csv (the proxy must not move with the arm)
# deliberate: every other flag — they are blendhold's and may not be touched in this arm.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
BLEND="$PROCESSED/labels/blend_three_v1.csv"
require "$BLEND"
require "$LABELS"

run_train blendema224 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --hold-annotations --ema \
  --labels "$BLEND" --eval-labels "$LABELS" --out "$MODELS/blendema224_s2026.pt"
