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
# A new member moves BOTH terms and they cancel. Lifting EVERY seat with the same flag moves the first
# term for certain, and may or may not touch the second. That is measurable from records already on
# disk, at zero card cost, and it was measured — twice, on two nested pools, all six members joined over
# the same 909 shared studies:
#
#     SOLO, the six members one at a time
#       calib_dense_hold 0.8634   band_dense_hold 0.8573     +0.0061
#       calib_span_hold  0.8565   band_span_hold  0.8493     +0.0072
#       calib_span       0.8545   band_span       0.8565     -0.0020
#                                            MEAN OF THREE   +0.0038
#
#     ENSEMBLED, the same members pooled by the shipped rank-average rule
#       calibrated trio 0.8685   uncalibrated twins 0.8629   +0.0056   against mean solo +0.0038
#       calibrated pair 0.8679   uncalibrated pair  0.8595   +0.0084   against mean solo +0.00665
#
# **The pass-through is ABOVE one — 147% on the trio and 126% on the pair.** That is not the 92% a pure
# error-term lift predicts, and the excess is the finding: calibration is buying AMBIGUITY as well as
# quality. The calibrated members disagree with each other more than their uncalibrated twins do, so the
# decomposition's second term moves the helpful way instead of not moving at all.
#
# THE PREDICTION, stated as arithmetic before the run, and deliberately taken at the LOW end:
#     five of six seats × 1.0 pass-through (not 1.3) × +0.0038 per seat = +0.0038 offline
#     × board_transfer +1.72 (read live 2026-08-26 15:24Z, 21 entries / 171 pairs) = +0.0065 board
#     champion `blendseat` 0.939 → **0.946**, with 0.949 the optimistic end if 1.3 holds at five seats
# Not 0.950 on its own under either. `margin_calib`'s seat (`knee-myl0`, +0.0018 offline ≈ +0.003 board
# if it lands) is the arm that would carry the rest, and it runs first for that reason.
#
# THE HONEST SIZE OF ALL OF THIS. Every SOLO delta above is inside the 0.0062 three-seed floor, the
# negative one included, so no single corner's sign is resolved and the per-corner spread is scatter
# rather than structure. Only the +0.0084 pair clears the floor. What licenses the arm is not any one
# pair but that the pooled solo estimate and both ensemble readings agree in sign across two nested
# pools — and it is licensed as a DIRECTIONAL bet on idle card, never as a result.
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
# control  merged_blend_s2026 = 0.8587 (declared 2026-09-03 from this header's own prose, knee-sgzgc; the
# ledger reads 0.8589, inside the 5e-4 resolution tolerance, so the run still resolves by this number)
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
#   4. **THE CONFOUND, AND IT POINTS AT THIS ARM.** Of the three matched pairs, the two that read
#      POSITIVE both carry `--hold-annotations` and the one that reads NEGATIVE does not. The five seats
#      below carry none. `calib_span` and `calib_span_hold` sit on the IDENTICAL corner — 12 slices,
#      0.1-0.9 — and differ in nothing else, yet their calibration deltas are -0.0020 and +0.0072, a
#      0.0092 spread on one flag. Two readings against it: the ledger's own `calibrated_targets` row is
#      +0.0038 against `wsfold0_e6` with no hold at the 0.2-0.8 corner, so no-hold is not uniformly
#      negative; and at n=1 per cell, with each delta a difference of single runs, a 0.0092 spread
#      between two deltas is comfortably inside what scatter produces. The arm stays single-flag rather
#      than acquiring `--hold-annotations` too, because that would be the reassembly this file's own
#      rule forbids. But if `calibsix` reads FLAT, this confound is the first suspect and the mechanism
#      is the second — the honest follow-up is one matched no-hold pair, not a verdict.
#
# COST: five twelve-epoch fits at 336 px. At `final_fit`'s measured ~425 s/epoch over 4,406 studies that
# is 60 epochs, about **7.1 h of card**, and ZERO submission slots to produce. Overnight card that would
# otherwise idle. The 0.2-0.8 seats are the two to drop first if the card becomes contended — they are
# the ones whose transfer is least evidenced (risk 1).
#
# deliberate: --calibrated-targets (the single flag, added to five seats; pooled solo +0.0038 over three
# deliberate: matched pairs, ensembled +0.0056 / +0.0084 on two nested pools)
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
  --labels "$LABELS" --out "$MODELS/final_ws_cf.pt"

run_train grade_ladder_cf \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE --grade-ladder $CALIB \
  --labels "$LABELS" --out "$MODELS/grade_ladder_cf.pt"

run_train through_dense_cf \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE --through-plane $CALIB \
  --labels "$LABELS" --out "$MODELS/through_dense_cf.pt"

# The geometry is SPELLED rather than inherited, for the reason final_fit_six.sh states: `rate_wholestack.sh`
# named neither --slices nor the band, so the original was fitted on the parser defaults. Stating them is
# what keeps this arm reproducible after the next default change, in a run with no holdout to notice.
run_train lr4e-5_ws_cf \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 $CALIB \
  --labels "$LABELS" --out "$MODELS/lr4e-5_ws_cf.pt"

# Seat five stays FOLD-0, exactly as its original is — it is the champion's one non-`_ff` member, and
# converting it here would change two things at once. It is also the only offline read this arm produces.
run_train merged_blend_cf \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --scheme merged $CALIB \
  --labels "$MERGED_BLEND" --eval-labels "$LABELS" --out "$MODELS/merged_blend_cf.pt"
