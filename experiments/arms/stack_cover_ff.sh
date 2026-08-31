#!/usr/bin/env bash
# stack_cover_ff.sh — full slice coverage carried into the best recipe the project owns, fit blind.
#
# bead: knee-ujd3
#
# It sits on top of knee-qphd (the coverage cell) and knee-8ft1 (the blind-fit lane).
#
# WHY THIS ARM EXISTS. The board history says the two routes to 0.946 are now countable. Fitting the 36
# scored entries that carry an honest offline number gives board = 1.37 x offline - 0.2598, so 0.946 asks
# for an ensemble offline of 0.8815 against the best ever assembled, honest12's 0.8751 — a gap of +0.0064,
# one paired floor (STRONG_OF_BEST = 0.0062) wide. The ensemble tracks best-single + ~0.015, so that gap
# is +0.0047 ON ONE MEMBER against the pool's best clean fold-0 run (merged_span_cf, 0.8618 over 97
# runs that share its ruler). That is route A, member quality.
#
# Route B, the blind refit, has exactly ONE move left and it is already made: five of the champion's six
# seats are `--fold -1 --ema` and `stackseat` refits the sixth. After that entry scores there is no
# further fit penalty to buy back, and every remaining slot has to be paid for out of route A.
#
# WHAT IT COMBINES, AND WHY THAT IS NOT TWO EXPERIMENTS. `stack224` occupies a cell no member ever has —
# 36 slices of a 30-40 slice stack, the first read that cannot miss a slice — on the BARE recipe, whose
# fold-0 control `native224` scores 0.8502. The pool's best members are made by two other levers, the
# blend-three corpus and distillation from `raptor_teacher.npz`, worth about +0.014 together. Nothing has
# ever carried coverage and those levers at once. Coverage changes WHICH PIXELS the encoder sees; the
# corpus and teacher change WHAT IT IS ASKED FOR. They are independent inputs to the same run, not two
# readings competing for one number.
#
# THE FIT IS BLIND BY DESIGN, NOT BY OVERSIGHT. `--fold -1 --ema` because the destination is the champion,
# where a fold-0 seat pays the penalty route B exists to remove. The cost is stated plainly: no holdout
# number exists for this run, nothing may enter the ledger as its score, and the board is its only
# instrument. `stack224` is its fold-0 sibling and is what says whether the coverage lever is real; this
# arm is what makes the shippable form exist if it is.
#
# THROUGHPUT FLAG DROPPED, deliberately. `$THROUGHPUT` (`--recompute-activations`) costs a measured 19%
# (798 img/s against 946, `kneemri/models/encoder_surgery.py:285`), taken on the 336-px geometry of 864
# images at 577 tokens that genuinely pinned the card. At 224 px this arm runs 432 images at 256 tokens
# and `stack224` measured 21.9 GB peak of 32.6 WITH the flag on. Recomputation trades memory for compute,
# so dropping it raises the peak — predicted near 29 GB, still inside the card. If it OOMs, the finding is
# that the threshold sits between 20 and 36 slices at 224 and the flag is justified there; relaunch with
# `$THROUGHPUT` restored and nothing about the score reading changes.
#
# HARNESS CHECK BEFORE ANY READING. The printed recipe must show `"image_size": "224"`, `"slices": "36"`,
# band [0.05, 0.95], `"slice_pooling": "max"`, `"whole_stack": "True"`, `"fold": "-1"`, `"ema": "True"`,
# `"calibrated_targets": "True"`, a non-zero distill weight, the raptor teacher path, and
# `blend_three_v1.csv` on "labels". `"recompute_activations"` must read "False" — that is the one flag
# this arm removes and the whole memory reading depends on it having landed.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

BLEND="$PROCESSED/labels/blend_three_v1.csv"
require "$LABELS"
require "$BLEND"
require "$PROCESSED/raptor_teacher.npz"

FIT="--fold -1 --ema"

run_train stack_cover_ff \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack \
  --backbone-lr 4e-5 --slices 36 --band-low 0.05 --band-high 0.95 --image-size 224 --calibrated-targets \
  --distill-weight 0.3 --teacher-targets "$PROCESSED/raptor_teacher.npz" $FIT \
  --labels "$BLEND" --eval-labels "$LABELS" --out "$PROCESSED/stack_cover_ff.pt"
