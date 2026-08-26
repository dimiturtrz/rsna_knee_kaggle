#!/usr/bin/env bash
# calibsix — the champion's five uncalibrated seats, refitted with `--calibrated-targets`.
# bead: knee-1c00
#
# THIS IS THE UNIFORM-LIFT ARM, AND IT IS A DIFFERENT MECHANISM FROM EVERY MEMBER ARM IN THE QUEUE.
# On 2026-08-26 seat pricing measured the pass-through of a NEW member's solo quality into an ensemble
# seat at **23%** — `calib_dense_hold` is +0.0056 solo and +0.0013 in a seat. That reprices the whole
# add-a-better-member lane down to a 0.943 ceiling. The ambiguity decomposition says why, and says what
# is left:
#
#     ensemble error  =  mean member error  −  ambiguity
#
# A new member moves BOTH terms and they cancel. Lifting EVERY seat with the same flag moves only the
# first. That second quantity has now been measured too, on three matched pairs, at zero card cost:
#
#     calibrated trio  (calib_dense_hold + calib_span_hold + calib_span)   0.8685
#     uncalibrated twins(band_dense_hold + band_span_hold  + band_span )   0.8629
#                                                               delta     +0.0056
#     mean solo delta of the same flag at the same corners                +0.0061
#
# **A uniform lift passes through at 92%, against 23% for a marginal member.** Both numbers are true and
# they do not contradict: the trio and its twins share the same ambiguity, so only the error term moved.
#
# THE PREDICTION, stated as arithmetic before the run:
#     five of six seats × 0.92 pass-through × +0.0061 per seat = +0.0047 offline
#     × board_transfer +1.72 (read live 2026-08-26 15:24Z, 21 entries / 171 pairs) = +0.0081 board
#     champion `blendseat` 0.939 → **0.947**
# Not 0.950 on its own. `margin_calib`'s seat (`knee-myl0`, +0.0018 offline ≈ +0.003 board if it lands)
# is the arm that would carry the rest, and it runs first for that reason.
#
# WHY FIVE AND NOT SIX. `calib_span_ff` already carries the flag; it is the sixth seat and is reused
# unchanged. Every other seat below is its settled arm copied VERBATIM plus exactly one flag, which is
# `final_fit_six.sh`'s own rule: a reassembly, however sensible each flag looks alone, is a configuration
# nobody ran, and with `--fold -1` there is no holdout left to catch it.
#
# THE FLAGS COMPOSE, CHECKED RATHER THAN ASSUMED. `--grade-ladder` is an OBJECTIVE policy (thresholds
# subtracted inside the loss, `kneemri/training/objective.py`); `--calibrated-targets` is a TARGETS policy
# (`core/data/labels/targets.py`). Different stages of the pipeline, so seat two is a genuine single-flag
# change and not two levers fighting over the same quantity.
#
# NONE OF THE FOUR `_cf` OUTPUTS CAN EVER BE SCORED OFFLINE. `--fold -1` means each has seen every holdout
# study, so no output here may be seat-scored, curve-scored or distance-scored. `merged_blend_cf` is the
# one exception — it is fold-0 like its original — and it is therefore the ONLY offline read this arm
# produces. Read it against `merged_blend_s2026` 0.8587 with `convergence_check` first; a truncated score
# under `one_cycle` is a FLOOR. If that one seat reads NEGATIVE, do not ship the entry: it is the only
# evidence available that the flag transfers to corners outside the two it was measured at.
#
# THE THREE RISKS, named before the card is spent:
#   1. The flag was measured at two corners (12-slice 0.1-0.9 span, 20-slice 0.1-0.9 dense). Seats four
#      and five sit at 0.2-0.8 / 12 slices, which is the pre-settlement geometry, and seat five trains on
#      the merged corpus with a blended label table. Transfer there is assumed, not measured.
#   2. `blendthree` swapped three of six seats for better-SUPERVISED twins and read 0.939 flat. That is
#      the nearest precedent and it returned nothing. The defence is size, not kind: blend supervision
#      measured +0.0034 at three seats (≈+0.003 board, inside a tick), this measures +0.0061 at five
#      (≈+0.008). If calibsix also reads flat, the uniform-lift route is refuted at the operating point
#      and the ensemble axis closes for good rather than just for marginal members.
#   3. It costs a submission slot to read at all, and slot 5 of 2026-08-26 is HELD by owner directive
#      (`knee-pubw`). This ships no earlier than 2026-08-27, and it is ASKED FOR before it is spent.
#
# COST: five twelve-epoch fits at 336 px. At `final_fit`'s measured ~425 s/epoch over 4,406 studies that
# is 60 epochs, about **7.1 h of card**, and ZERO submission slots to produce. Overnight card that would
# otherwise idle. The 0.2-0.8 seats are the two to drop first if the card becomes contended — they are
# the ones whose transfer is least evidenced (risk 1).
#
# deliberate: --calibrated-targets (the single flag, added to five seats; measured +0.0061 at this corner)
# deliberate: --fold -1 --ema (inherited verbatim from final_fit_six.sh, not chosen here — with no
# deliberate: best-epoch selection left, ema is the deliverable's only variance control)
# deliberate: --scheme merged (seat five's own corpus, copied from merged_blend.sh)
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

MERGED_BLEND="$PROCESSED/labels/merged_blend_v1.csv"
require "$MERGED_BLEND"

FIT="--fold -1 --ema"
DENSE="--slices 20 --band-low 0.1 --band-high 0.9"
CALIB="--calibrated-targets"

run_train final_ws_cf \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE $CALIB \
  --labels "$LABELS" --out "$PROCESSED/final_ws_cf.pt"

run_train grade_ladder_cf \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE --grade-ladder $CALIB \
  --labels "$LABELS" --out "$PROCESSED/grade_ladder_cf.pt"

run_train through_dense_cf \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE --through-plane $CALIB \
  --labels "$LABELS" --out "$PROCESSED/through_dense_cf.pt"

# The geometry is SPELLED rather than inherited, for the reason final_fit_six.sh states: `rate_wholestack.sh`
# named neither --slices nor the band, so the original was fitted on the parser defaults. Stating them is
# what keeps this arm reproducible after the next default change, in a run with no holdout to notice.
run_train lr4e-5_ws_cf \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 $CALIB \
  --labels "$LABELS" --out "$PROCESSED/lr4e-5_ws_cf.pt"

# Seat five stays FOLD-0, exactly as its original is — it is the champion's one non-`_ff` member, and
# converting it here would change two things at once. It is also the only offline read this arm produces.
run_train merged_blend_cf \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --scheme merged $CALIB \
  --labels "$MERGED_BLEND" --eval-labels "$LABELS" --out "$PROCESSED/merged_blend_cf.pt"
