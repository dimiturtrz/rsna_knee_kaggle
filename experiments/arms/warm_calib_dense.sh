#!/usr/bin/env bash
# warm_calib_dense.sh — the first warm start this project has ever run.
#
# THE GAP THIS OPENS. `kneemri/training/fitted_encoder.py` converts a fitted member into the file
# `--encoder-weights` reads, and its own docstring names the waste it was built to end: "every arm
# repeating an architecture paid for the first eight epochs again — the ones that only re-learn what a
# member on disk already knows." It was then never used. Zero of 131 arm scripts pass --encoder-weights,
# --warm or anything like them. Every member this project owns was fitted cold from the stock
# natural-image trunk.
#
# `--resume` is NOT one of the missing flags and never was a warm start: it exists (knee-c317d), it
# restores the optimiser, schedule, scaler and four random streams, and it is proven to land a killed
# run within 1e-6 of the uninterrupted one. It is opt-in on purpose — a default resume would silently
# continue a stale checkpoint after a recipe edit, which is the arm you can no longer compare.
#
# That is not a small omission. 98 members on disk carry 36 distinct recipe skeletons and the modal one
# accounts for 31 of them — a third of everything we own is the SAME decode path re-run with a different
# loss, label table or calibration, which is exactly the class of change the house rule says to warm-start
# rather than refit from scratch.
#
# THE MECHANISM IS NOT ONLY "FEWER EPOCHS". `FittedEncoder.merge_adapters` folds the member's LoRA
# correction back into the base weights, so the file this reads is an ALREADY-ADAPTED trunk carrying no
# adapters. `--lora-rank` then builds a FRESH rank-r adaptation on top of it. So a warm restart doubles
# the effective adaptation capacity without widening any layer or adding a parameter at serve time — the
# member is the same shape and the same cost as its parent. That is a capacity argument, not a schedule
# argument, and it is the reason this is worth a card hour rather than a footnote.
#
# control  calib_dense_hold_s2026, holdout 0.8634 — the best solo member this project owns, and the
#          source of the encoder this reads. SINGLE VARIABLE against it: --encoder-weights and nothing
#          else. Same recipe, same twelve epochs, same lr, same held split, so the two numbers are
#          directly comparable and a difference is the warm start.
# cost     one 12-epoch fit at ~200 s an epoch, ~40 min. Zero submission slots.
#
# WHY TWELVE EPOCHS AND NOT SIX, given that saving time is half the point. Six would confound the warm
# start with a compressed one_cycle, and `recipe.sh` records that compressing the schedule cost 0.0028 on
# its own. This arm answers "does a warm start beat a cold one at EQUAL budget" — the readable question.
# `warm_calib_dense_short.sh` asks the budget question afterwards, and only makes sense if this one clears.
#
# PREDICTION, written before the run. If the doubled-adaptation argument holds, this lands above 0.8634 —
# and any member above 0.8634 is the best single this project has produced, on the day three board entries
# (calibsix 0.941 at six seats, calibmix 0.941 at five recipes, pixgroup 0.940 at eight) said the seating
# axis is saturated for us and member quality is the only lever left.
#
# THE REFUTATION CONDITION, and the harness suspect named in advance. A score at or below the control says
# the merged trunk carries no capacity a fresh adaptation can use, and warm starting is then a pure
# time-saving with no quality term — still worth having, but a different claim. A score WELL below (say
# under 0.85) is more likely the SCHEDULE than the mechanism: one_cycle ramps the backbone rate above
# 4e-5 before annealing, and a converged trunk can be walked off its optimum by the ramp. That outcome
# sends the next arm to a lower --backbone-lr, not to the graveyard. Verify the flag LANDED before
# believing either: `--encoder-weights` is in `landed.py`'s registry and `PretrainedEncoder.into` raises
# on a variant or key mismatch, so a silent no-op is not among the failure modes — but check the log says
# so rather than assuming it.
#
# bead: knee-wbwq
#
# NO --vram-gb: DeviceBudget derives the cap from --slices — 20 slices reserve 28.26 GB.
# control  calib_dense_hold_s2026 = 0.8634 (declared 2026-09-03 from this header own prose, verified on disk, knee-sgzgc)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
require "$MODELS/encoders/calib_dense_hold.enc.pt"

run_train warm_calib_dense \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --encoder-weights "$MODELS/encoders/calib_dense_hold.enc.pt" \
  --labels "$LABELS" --out "$MODELS/warm_calib_dense_s2026.pt"
