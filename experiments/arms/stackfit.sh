#!/usr/bin/env bash
# stackfit — the ADDITIVITY test, and the only queued arm whose ceiling reaches the goal.
# bead: knee-orjq
#
# WHY THIS EXISTS AND WHY NOW. `interpretations/kneemri/2026-08-26_what-0950-actually-costs.md` prices
# first place at **+0.010 offline on one member** and, since RAD-DINO ran and read −0.0371, names no
# single arm that reaches it. Its own conclusion is that the route is a STACK of directional gains —
# `CLAUDE.md`'s "increments below the bar still compose, and a chain of them that sums past it IS a
# result". Nothing in this project has ever tested whether they DO compose. Every arm in the ledger
# moves one flag against a control and is read alone. Two of them are about to have singles measured on
# the same corner in the same hours, which is the first time an additivity test has been cheap.
#
# THE TWO TERMS, AND WHY THESE TWO. Both are already queued as singles ahead of this arm:
#   `--calibrated-targets` — `knee-bs4l` measured the matched pair at **+0.0072 proxy / +0.0129
#     annotated**, which is the largest directional gain currently unrefuted, and `calib_dense_hold`
#     reads it at this exact corner. Its own header predicts the first member above 0.860.
#   `--ema` — `knee-56w5` pre-registered ≥0.8640 against `band_dense`'s 0.8578. A competitor's
#     single-variable ablation of the same flag reads **+0.003**
#     (`research/2026-08-26_the-ablation-table-a-competitor-published.md`), so the honest prior is half
#     the bar and directional. `ema_dense` reads it at this corner.
# They are mechanically INDEPENDENT — one changes what the targets say, the other changes which weights
# are kept — which is the precondition for expecting a sum rather than a substitution.
#
# THIS IS A GATED ARM. Do not run it until BOTH singles have landed and BOTH are non-negative. If either
# reads below its control, this becomes a two-variable arm with a known-bad term in it and diagnoses
# nothing. Read them with `python -m kneemri.experiments.convergence_check` first: under `one_cycle` a
# truncated run's score is a FLOOR, and gating on a floor is gating on nothing.
#
# THE CONTROL IS `calib_dense_hold`, NOT `band_dense_hold`. Against it this arm moves exactly one flag
# (`--ema`), so the ADDITIVE prediction is a subtraction with no correction terms:
#
#     predicted = calib_dense_hold + (ema_dense − band_dense)
#
# `ema_dense` carries no `--hold-annotations` and its control `band_dense` does not either, so its delta
# is internally matched and portable; `band_dense_hold` against `band_dense` prices the hold itself, and
# that price cancels out of the difference. State the arithmetic before the number, not after.
#
# PRE-REGISTRATION, in the order the ladder should be read:
#   >= predicted + 0.0062   SUPER-ADDITIVE. Believe it only after a second seed; this is the one band
#                           where the honest response is suspicion, since two flags do not usually help
#                           each other and the floor is one-seed.
#   within 0.0062 of predicted  ADDITIVE, and this is the result the page needs. It licenses stacking as
#                           a METHOD: the next arms are the remaining unrefuted directional terms
#                           (blend supervision, if `blendhold` returns one) laid on top of this one,
#                           and the +0.010 becomes a construction rather than a hope.
#   <= predicted - 0.0062   SUBSTITUTES, NOT COMPLEMENTS — the `depth_regions_span` shape, where two
#                           levers answering the same question cost more together than either alone.
#                           That KILLS the stack route, and it kills it cheaply and early, which is
#                           worth the card on its own: the page's entire remaining plan rests on an
#                           assumption nothing has tested.
#   <= band_dense_hold      SUSPECT THE HARNESS FIRST. Both terms measured positive alone and negative
#                           together is a plumbing shape before it is a mechanism; diff the recorded
#                           recipe stamps key by key before writing a word about additivity.
#
# WHAT IT CANNOT SAY. Two points do not fit a composition rule. If it reads additive, that is evidence
# about THESE two terms, not a law about the next three, and each further term still needs its own
# single before it is stacked. And the six-member ensemble is 0.939 flat across ten entries, so a member
# gain here converts through `board_transfer` (+1.72) and the tight +0.0125 ensemble bonus, not through
# hope: +0.010 offline is the whole distance to first place, and this arm's ceiling is the only one
# queued that touches it.
#
# COST: ~85 min of card at 336 px / 20 slices, ZERO submission slots, fold-0 holdout. It also reads on
# the annotation ruler (`--hold-annotations`), so `knee-9198`'s both-rulers rule is satisfied without a
# second run.
#
# deliberate: --calibrated-targets (term one; single measured by calib_dense_hold at this corner)
# deliberate: --ema (term two; single measured by ema_dense, portable via band_dense)
# deliberate: --hold-annotations (matches calib_dense_hold, the control, and buys the second ruler)
# deliberate: --slices 20 --band-low 0.1 --band-high 0.9 (the dense corner every control here sits on)
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train stackfit \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --ema --hold-annotations \
  --labels "$LABELS" --out "$PROCESSED/stackfit_s2026.pt"
