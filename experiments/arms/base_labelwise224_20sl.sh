#!/usr/bin/env bash
# base_labelwise224_20sl.sh — the validated trunk given DEPTH instead of pixels.
#
# bead: knee-mm6v0
# reading: base_labelwise224_20sl_s2026.holdout.npz
#
# WHAT LICENSES THIS ARM. `base_labelwise224` landed 0.8899 ledger75 against `labelwise224`'s 0.8832 —
# +0.0067, above the 0.0062 paired floor, converged over 12 epochs on 909 studies. Capacity is not inert
# once the readout can spend the width. The shipping family (labelwise_dense 0.8964,
# labelwise_dense_drop40_long36 0.9030) beats that from a SMALL trunk, and it differs on two axes at
# once: 336 pixels AND 20 slices. This arm reads the second one.
#
# WHY NOT THE PIXEL AXIS, WHICH WAS TRIED FIRST AND FAILED TO RUN. `base_labelwise336` OOMed twice, at a
# 20 GB cap (18.67 allocated) and again at 27 (23.21 allocated, 2.38 more requested, 31.84 on the card).
# base at 336/12sl does not fit at batch 12, and batch is the one dimension that may not be cut: the
# pairwise `--rank-weight 1.0` term sees `--batch` studies within a step regardless of `--accumulate`, so
# a cut takes it from 66 pairs to a handful and confounds every comparison against the 0.88-0.90 controls.
# Recording the measurement rather than the intent: the variant-blind derivation said 36 GB for this
# corner and was closer to right than the 1.89x base/small ratio measured at 224/12sl, which predicted
# 17.6. That ratio understates as tokens grow and should not be reused at a higher resolution.
#
# TWO VARIABLES, AND THE PAIR THAT CALIBRATES THEM. The 224px/20sl/0.2-0.8 array is not decoded and
# building one costs hours and ~37 GB on a disk at 95%. The 224px/20sl/0.1-0.9 array IS on disk, so this
# arm moves slices 12 -> 20 AND the band 0.2-0.8 -> 0.1-0.9 together against `base_labelwise224` 0.8899.
# That is two variables, and it is stated rather than hidden. What makes it readable anyway: the same
# two-variable step is already measured on a MAX head, where `native224` (12sl, 0.2-0.8) reads 0.8800 and
# `band224` (20sl, 0.1-0.9) reads 0.8801 — +0.0001, flat. So the input change carries nothing by itself
# on a head that discards per-slice detail, and anything this arm reads above noise is the INTERACTION
# with the labelwise readout, which is the disowned-ruler claim AXES.md section B makes in general.
#
# MECHANISM. Twelve slices over a 0.2-0.8 band sample the joint every ~4 mm; twenty over 0.1-0.9 sample
# it every ~3 mm and reach further into the periphery. A meniscal tear or a cruciate disruption is a THIN structure, and the failure mode being
# tested is that it falls between sampled planes entirely — a SAMPLING change, like resolution, but
# along depth rather than within a plane. A labelwise head attends each finding over its own slices, so
# more planes is more for it to attend over rather than more to average away, which is why this is worth
# reading on this head and was not on a `max` one.
#
# PRICE, out loud. `base_labelwise224` ran 171 s/epoch at 12 slices; 20 slices puts this near 285 s, so
# ~1 h of card for 12 epochs. Zero submission slots, control free. Whether the 224px/20sl/0.2-0.8 array
# is already decoded is what `arm_check` answers; if it is not, the decode is the arm's real cost and the
# price above is wrong. P(clears 0.8961) ~ 0.35. P(clears the 1.5% bar) ~ 0.10.
#
# BANDS, pre-registered, ledger75, against base_labelwise224 0.8899.
#   * >= 0.8961  depth sampling carries the gap under labelwise where it read +0.0001 under max, and the shipping family's lead over the 224 corner is
#                about slices rather than pixels. The next arm is base at 20 slices with the dense recipe.
#   * 0.8899-0.8961  directional. Keep nothing, compose nothing.
#   * <  0.8899  depth is saturated at 12 under this head, which leaves pixels as the remaining
#                explanation for the shipping family's lead — and pixels are known not to fit on this
#                card for a base trunk, so that verdict re-points the axis at the small trunk.
#
# READ THE FOCAL LABELS, NOT ONLY THE MEAN. Lateral Meniscus -0.1925, ACL -0.1225, MCL -0.0964 lead the
# model-side error census and are the thin structures a depth-sampling change has to move. If the mean
# moves and those do not, the delta is not this mechanism. Say so.
#
# HARNESS CHECK. The printed recipe must name "variant": "base" AND "slices": "20" AND
# "slice_pooling": "labelwise" AND "image_size": "224".
#
# control  base_labelwise224_s2026 = 0.8899 ledger75 @ epoch 8, 909 studies. No control run is owed.
#
# deliberate: --slices 20 --band-low 0.1 --band-high 0.9 (the arm's thesis, and the array that exists)
# deliberate: --vram-gb 24 — named rather than derived. base measured 12.99 GB allocated at 224/12sl;
#             activation memory is linear in slices, so 20 slices predicts ~21.6. 24 is that plus slack,
#             and unlike the 336 attempt it is a scaling WITHIN the resolution the ratio was measured at.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train base_labelwise224_20sl \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --image-size 224 --variant base --vram-gb 24 \
  --labels "$LABELS" --out "$MODELS/base_labelwise224_20sl_s2026.pt"
