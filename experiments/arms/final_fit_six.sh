#!/usr/bin/env bash
# final_fit_six.sh — the champion's OTHER six members, refitted on 100% of the corpus.
#
# THIS ARM IS GATED AND MUST NOT BE RUN UNGATED. It is the expensive half of a cost-tiered pair whose
# cheap half is `final_fit.sh` (knee-g8nb): 93 min of card decides these ~7 h. Run this ONLY if
# `kneemri-finalfit` reads **0.936 or above** on the board. At 0.934-0.935 keep the one member and stop;
# at 0.933 or below the data step did not survive the swap and this arm is retired unrun.
# gate: kneemri-finalfit >= 0.936
#
# The line above is the half `arm_check` evaluates; the paragraph is the half it cannot. On 2026-08-23
# the board read 0.934 and this arm ran anyway for five hours, because prose has no consumer (knee-1e3m).
#
# WHAT IT IS. `gradeswap`'s six non-`band_dense` seats, each copied VERBATIM from the arm that settled
# it, plus exactly two flags: `--fold -1` (4,406 studies rather than 3,497, +26%) and `--ema`. Copying
# rather than reassembling is the same rule `final_fit.sh` states in its own header — every reassembly,
# however sensible each flag looks alone, is a configuration nobody ran, and with `--fold -1` there is no
# holdout left to catch it.
#
# WHY `--ema` RIDES ALONG, NAMED RATHER THAN HIDDEN. With `--fold -1` there is no best-epoch selection
# and the trainer keeps the LAST epoch, so ema is the only variance control the deliverable has. It is
# the same confound `final_fit.sh` carries, which is what makes the two comparable: if the single-member
# swap licensed this arm, it licensed "data step + ema", and this arm applies the identical pair.
#
# NONE OF THESE SIX CAN EVER BE SCORED OFFLINE AGAIN. Each has seen every holdout study, so no output of
# this arm may be seat-scored, curve-scored or distance-scored, and none may enter a pool that reports an
# offline mean. A roster built from them ships blind against whatever the board's incumbent is.
#
# NEW NAMES, DELIBERATELY. Every output carries a `_ff` suffix, because the fold-0 originals are seated
# in shipped entries and overwriting one would destroy a result to answer a question about a different
# fit. `comp/champion.json` pins a rebuildable path; these must not stand in it.
#
# COST. Six twelve-epoch fits at 336 px, one at 24 epochs (`band_dense_long`) and one at DINOv2-base
# (`base_dense`, batch 6). On `final_fit`'s measured ~425 s/epoch at 4,406 studies, that is roughly
# 12+24+12+12+12+12 = 84 epochs, about 10 h rather than the 7 h the price was first written at. The
# 24-epoch seat is the one to drop first if the card is contended.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

FIT="--fold -1 --ema"
DENSE="--slices 20 --band-low 0.1 --band-high 0.9"

run_train calib_span_ff \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 --calibrated-targets \
  --labels "$LABELS" --out "$PROCESSED/calib_span_ff.pt"

run_train grade_ladder_ff \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE --grade-ladder \
  --labels "$LABELS" --out "$PROCESSED/grade_ladder_ff.pt"

run_train through_dense_ff \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE --through-plane \
  --labels "$LABELS" --out "$PROCESSED/through_dense_ff.pt"

# The geometry is SPELLED here although its original arm was silent about it. `rate_wholestack.sh` names
# neither --slices nor the band, so `lr4e-5_ws_s2026` was fitted on the parser defaults — 12 slices,
# 0.20-0.80 — which knee-aixw calls the pre-settlement geometry. Those are still today's defaults, so
# inheriting and stating them resolve to the same argv; stating them is what keeps that true after the
# next default change, in the one arm with no holdout to notice.
run_train lr4e-5_ws_ff \
  $RECIPE $FIT --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --labels "$LABELS" --out "$PROCESSED/lr4e-5_ws_ff.pt"

run_train base_dense_ff \
  $RECIPE $FIT --epochs 12 --batch 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant base --backbone-lr 4e-5 $DENSE \
  --labels "$LABELS" --out "$PROCESSED/base_dense_ff.pt"

run_train band_dense_long_ff \
  $RECIPE $FIT --epochs 24 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 $DENSE \
  --labels "$LABELS" --out "$PROCESSED/band_dense_long_ff.pt"
