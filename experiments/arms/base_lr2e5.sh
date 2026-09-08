#!/usr/bin/env bash
# base_lr2e5.sh — DINOv2-base's fourth draw, and the only thing that moves is the backbone rate.
#
# WHY THIS FAMILY. D128 §2's re-bucket prices every family against the expected max of the same number
# of draws from the default family's own 154-arm distribution. `base` is the ONLY row above it:
#
#   base        n=3  best 0.8886  E[max|null] 0.8867   +0.0019
#   resnet50d   n=1  best 0.8659  E[max|null] 0.8665   -0.0006
#   mricore     n=1  best 0.8647  E[max|null] 0.8658   -0.0011
#   vjepa2l     n=2  best 0.8693  E[max|null] 0.8834   -0.0141  p=0.055
#   ortholarge  n=6  best 0.8758  E[max|null] 0.8899   -0.0141  p=0.0008
#
# +0.0019 at n=3 is p~0.60 and is NOT a result — it is the only non-`default` family that has not been
# refuted, which is a different and much weaker sentence. This arm buys the fourth draw.
#
# WHY THE RATE AND NOT SOMETHING ELSE. `base_clean` runs `--backbone-lr 4e-5`, which is the ViT corner
# rate FITTED ON DINOv2-SMALL and transferred to a trunk with roughly four times the arithmetic per
# image. That transfer is exactly the confound D118 §2 named: a family swapped in under a recipe fitted
# on `default` is not an exchangeable draw, and every knob in this line (`--slice-pooling max`,
# `--whole-stack`, `--slices 20`, the 0.1/0.9 band) was fitted on small. The rate is the one of them
# that scales with model size on a known mechanism rather than on a preference, so it is the first to
# move. 2e-5 is one halving — a single step, not a sweep, because at 4.5 h a step is what the card buys.
#
# WHAT IT COSTS, SAID BEFORE IT RUNS. `base_clean.sh:59` prices this line at **~4.5 h of card, 0 slots**
# — roughly four times small's arithmetic at the same 336 px / 20 slices, batch 6 with accumulate 2.
# D128 §2 asked for TWO base arms at "~1.4 h each"; that figure is not this family's. Two draws are 9 h,
# a third of the UTC day, so this is queued as ONE and the second waits on this one's read.
#
# PRE-REGISTERED, before it runs, and it is D128 §2's own bar:
#   > 0.8899       base's best exceeds E[max|null] at n=6 — the first honest signal any non-default
#                  trunk has produced; the family gets priority and the second draw queues immediately
#   0.8867-0.8899  base stays above its n=3 expectation and below its n=4-6 one; the family is neither
#                  refuted nor supported, and it is three-to-five lucky small arms until a draw says
#                  otherwise. That sentence gets written down rather than deferred again
#   < 0.8867       the rate was not the constraint AND the family drops below its own null at n=4; the
#                  last unrefuted non-default trunk joins the other five, and the trunk axis closes on
#                  a measurement instead of on a name-parse
#
# WHY NOT re-run `base_clean` verbatim for a pure seed draw. Seeds are banned, and a verbatim re-run is
# a seed draw wearing a new checkpoint name. The rate move is what makes this a new CELL rather than a
# second sample of one.
#
# bead: knee-rag9t
# reading: D:/models/rsna_knee_abnormality/base_lr2e5_s2026.holdout.npz
#
# deliberate: --backbone-lr 2e-5 (the swept knob, the only difference from base_clean)
# deliberate: --variant base (explicit; the default is SMALL, encoder_flags.py:52-56, which is the
#             defect that made D117's name-parsed buckets readable as families in the first place)
# deliberate: --epochs 12 (matched to base_clean, so the two are comparable at the same budget)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train base_lr2e5 \
  $RECIPE --epochs 12 --batch 6 --accumulate 2 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant base --backbone-lr 2e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --ema --hold-annotations \
  --labels "$LABELS" --out "$MODELS/base_lr2e5_s2026.pt"
