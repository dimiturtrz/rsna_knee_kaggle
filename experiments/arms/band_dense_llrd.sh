#!/usr/bin/env bash
# band_dense_llrd.sh — open the WHOLE encoder, but let only its top move fast.
#
# WHAT IS NEW HERE, and it is a mechanism this repo did not have until today. Every arm ever run gave
# each opened block ONE learning rate. `StudyModel.parameter_groups` built exactly two groups — head at
# `head_lr`, every opened block at a flat `backbone_lr` — and a grep for layer-wise decay across
# `kneemri/`, `experiments/README.md` and `research/` returned zero hits. `Adaptation`'s own docstring
# records depth and rate as ENTANGLED and calls it unresolved; a flat rate is why. Raising the rate until
# the top blocks move applies that same rate to the earliest blocks, which hold the most general features
# and are the most damaging to disturb, so the depth bracket could only ever be read along a diagonal.
# That is not a subtle confound: our bracket found unfreezing every block worth +0.0008 and could not
# distinguish "more blocks do not help" from "the extra blocks were being wrecked at the top's rate".
#
# THE MECHANISM. Under `--layer-decay 0.7` the TOP block keeps this recipe's own 4e-5 and each block
# below it is divided down: block 6 sees 6.7e-6, block 0 sees 7.9e-7. So the encoder's general layers get
# a whisper rather than a rewrite, while supervision gains a gradient path through all twelve. The
# hypothesis is that the useful part of depth is a SMALL correction to the early blocks, and that a flat
# rate has been buying the correction and the damage together and reading their sum.
#
# ONE MECHANISM against `band_dense` (0.8578), which is the shipped recipe and the champion's first
# member: identical cache, band, slices, schedule and top-block rate. What moves is that blocks 0-5 open
# at a decayed rate and blocks 6-11 are re-graded by depth instead of sharing 4e-5. It is deliberately
# NOT the SSL-import axis — `encoder_adapt_thawed` is testing 12 flat blocks at 4e-5 over an adapted
# trunk and is running 0.017 BELOW its own six-block control at matched schedule position, which is the
# same wrecking this arm exists to remove; but that axis sits at 0.73 against a shipped 0.858, and
# rescuing it does not move the score. This arm is on the recipe that ships.
#
# THE BATCH IS A FORCED SECOND VARIABLE AND IT IS PRICED. Twelve trainable blocks retain twice the
# activation stack and OOM at this recipe's default batch 12 with `--recompute-activations` already on;
# the VRAM cost is a function of trainable DEPTH, not of the rate, so decay does not buy it back.
# `--batch 2 --accumulate 6` holds the effective batch at 12. `small_accum` measured that substitution at
# 0.8579 against `small_batch6`'s 0.8598 and the control's 0.8578 — three points inside a 0.0020 spread,
# a third of the 0.0062 noise floor. The ranking half of the loss is a within-micro-batch RATIO and so is
# genuinely not pooled across the six groups; that was measured too rather than argued: pair mass per step
# falls 11x while the estimator's coefficient of variation rises 0.326 -> 0.386 and no label goes dark
# (worst: Synovitis contributes nothing on 12.4% of steps against 11.3%). Real, sub-noise, named here
# rather than discovered in the delta.
#
# READ IT ON, in order:
#   * ABOVE 0.8578 by more than 0.0062 — depth was never the problem, the flat rate was, and the whole
#     depth axis has been measured through a broken instrument and is worth re-running.
#   * ROUGHLY 0.8578 — decay is neutral, and the +0.0008 depth bracket stands as a real null rather than
#     a confounded one. Cheap, and it retires the suspicion.
#   * BELOW — the early blocks do not want to move at any rate, which is a positive finding about WHERE
#     this encoder's transferable structure lives and closes the depth axis for good rather than leaving
#     it "unresolved" as `Adaptation` has it now.
# The TRAINING LOSS is the second instrument: if the early blocks are being wrecked at 4e-5, the flat
# 12-block arm's loss floors high and this one's should track `band_dense`'s 1.2162 -> 0.5780 curve.
#
# control  band_dense = 0.8578 (best epoch 8), byte-identical but six blocks at one flat 4e-5.
# reference small_accum = 0.8579, which is what prices the batch-2 substitution above.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band_dense_llrd \
  $RECIPE --epochs 12 --batch 2 --accumulate 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --unfreeze-last 12 --layer-decay 0.7 \
  --labels "$LABELS" --out "$PROCESSED/band_dense_llrd_s2026.pt"
