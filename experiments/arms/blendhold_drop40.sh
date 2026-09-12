#!/usr/bin/env bash
# bead: knee-wy4vt
# reading: blendhold_drop40_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `blendhold`'s byte-identical line with one flag added: `--slot-dropout 0.40`. blendhold is a SEATED
#   member of `cleanseat_v1` (0.942 on the board, 6 members). Nothing else changes.
#
# THE GAP IT CLOSES, WHICH NOBODY HAD NAMED
#   Every member of the champion roster carries `--calibrated-targets` / `--hold-annotations` and most
#   train on `$BLEND`. `slot_dropout` has ONLY ever run on the plain-label 224/20-slice band recipe
#   (`slot_dropout`, `slot_dropout_hi`, `slot_dropout_max`, and now `mergedk_drop40`). So the operator
#   with the LARGEST cross-site gain this project has measured has never met a seat, and the champion
#   roster carries ZERO cross-site hardening. The owner's own statement is that the real Kaggle test set
#   is not our dataset — the board IS the cross-site axis, and that is the axis this operator moves.
#
# WHY blendhold IS THE SEAT TO TEST ON
#   Two independent reasons, both read off `seat_value --against cleanseat_v1` today. It is the cheapest
#   member to drop (-0.0006, the smallest of the six), so it is the seat under the least pressure; and it
#   is the seat BOTH mergedk candidates were assigned when the tool held the roster at six and asked
#   which member each would replace. If any seat is contestable, it is this one.
#
# DOSE IS PINNED, NOT SWEPT
#   0.40 is the operating point `knee-yotj8` closed on: decode_cost -0.0271 / -0.0229 / -0.0214 at
#   0.25 / 0.40 / 0.55, with 0.55 landing inside the pre-registered SATURATING band. No dose is re-asked.
#
# THE HOLDOUT IS THE WRONG RULER AND THAT IS SAID BEFORE THE RUN
#   `slot_dropout`'s gain is CROSS-SITE (+0.0132 vs control on `foreign_readout`) and holdout909 is not
#   cross-site. On the plain recipe the operator read ~flat offline; on the mergedk corpus it read
#   +0.0040, inside a null band. So a flat here refutes nothing. The DECIDING ruler is
#   `seat_value --against cleanseat_v1` with blendhold swapped out, and `foreign_readout` for the
#   cross-site claim.
#
# PRE-REGISTRATION (written before the run)
#   seat >= +0.0040   the operator earns a seat. Package on shelf 2 and ship the swap. This is the bar
#                     under which this ruler has never been shown to pick a board tick.
#   seat +0.000..+0.004
#                     directional. Keep for the pool, re-open nothing, and do NOT spend a slot on it.
#   seat < 0.000      the operator does not transfer to a blended-label seat. The axis stays open on the
#                     OTHER five recipes only if `foreign_readout` still shows the cross-site gain here —
#                     if it does not, the operator's gain is a property of the plain-label recipe and
#                     that is a real finding about what it was actually buying.
#   holdout           expected FLAT (within +-0.006 of blendhold 0.8575) by the argument above. A read
#                     outside that band on EITHER side is a fact about the recipe interaction, not about
#                     the operator, and is not to be quoted as a cross-site number.
#
# deliberate: --slot-dropout 0.40 — imported operating point, not a swept knob (knee-yotj8).
# deliberate: every other flag copied from blendhold.sh unchanged, so the seat comparison is one variable.
#
# control  blendhold_s2026 at 0.8575, already on disk and already seated. Costs no run.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

BLEND="$PROCESSED/labels/blend_three_v1.csv"
require "$LABELS"
require "$BLEND"

run_train blendhold_drop40 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --hold-annotations \
  --slot-dropout 0.40 \
  --labels "$BLEND" --eval-labels "$LABELS" --out "$MODELS/blendhold_drop40_s2026.pt"
