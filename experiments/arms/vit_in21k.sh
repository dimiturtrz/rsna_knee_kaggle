#!/usr/bin/env bash
# vit_in21k.sh — the PRETRAINING OBJECTIVE, with the architecture held fixed.
#
# bead: knee-fffo
# deliberate: --variant vitin21k (the arm's thesis — the objective swap, at zero architectural distance)
#
# WHY THIS ARM. `swin_supervised` is this project's first supervised-pretrained trunk, and it varies TWO
# things at once against every incumbent: the objective (IN-22k human labels, not self-distillation) AND
# the architecture (shifted-window attention, not global attention over a flat grid). Whatever number it
# returns, it cannot say which half carried it. This arm can. `google/vit-base-patch16-224-in21k` is an
# EXACT architectural twin of `v3base` — ViT-B/16, twelve blocks, 768 wide, the same global attention over
# the same patch grid at the same stride — and differs from it in the pretraining objective ALONE: AugReg
# supervised IN-21k against DINOv3 self-distillation on LVD-1689M.
#
# SINGLE VARIABLE, and it is the objective. This is `v3base_dense.sh`'s invocation with `--variant
# vitin21k` substituted for `--variant v3base`. Rate, slices, band, pooling, whole-stack, epochs, schedule
# and batch are all held at the values `v3base_dense`'s 0.8455 was measured under, which makes that run
# the matched control and this comparison a clean objective swap. At 336 px and patch 16 the token count
# is identical too — 441 either way — so even the compute is matched.
#
# THE EFFECTIVE BATCH IS 6, SPLIT. `--batch 6` OOMs on a verified-free card: AugReg keeps no memory-saving
# path DINOv3 has, and the smoke asked 23.3 GB of a 26.1 GB allowance before dying. `--batch 2
# --accumulate 3` restores the gradient of six studies at the activation memory of two, so the optimiser
# sees the control's batch. It is not byte-identical: anything pooled WITHIN a step now sees two studies
# rather than six. With `--slice-pooling max` over slices of one study that pooling is intra-study, so no
# cross-study statistic changes; the residual difference is the gradient noise of three smaller forwards.
#
# WHERE IT SITS ON THE knee-y59c CURVE, and this is the argument for expecting a real number. That
# measurement anti-correlates solo strength with distance-from-DINOv2 across seven trunks, and the two
# far-end deaths are `mae_matched` (0.6735) and `focus_mcl` (0.5332). An architectural twin sits at ~0 on
# that axis. If a supervised ViT still dies here, the curve is not about architecture at all.
#
# NOT IMAGENET NORMALIZATION, and it is declared in the registry as `HALF`. AugReg states 0.5/0.5 —
# a property of that pretraining CODEBASE, not of its corpus, which is why it cannot be guessed from the
# data the way RAD-DINO's greyscale constants can. Feeding this trunk the ImageNet constants is the
# silent handicap `Normalization` exists to prevent, and it would land squarely on the one hypothesis
# this arm was built to test.
#
# UNGATED AND APACHE-2.0, so unlike `v3base_dense` there is no HF token step. No Kaggle mount yet: this
# trains here and cannot be submitted until the weights are mirrored, the same dataset upload
# `dimiturnt/swin-base-384` already is, deferred until a member earns the seat.
#
# control  v3base_dense = 0.8455 (best epoch 6), byte-identical recipe on the architectural twin.
#          Its seat value against the five-member champion was -0.0005.
#          base_dense (DINOv2-base, same recipe) = 0.8512 is the other twin and the stronger control.
# cost     v3base_dense ran ~2.5-2.9 h at these settings; identical token count, so expect the same.
#
# PRE-REGISTERED, four outcomes. Floor 0.0062, so the twin's band is 0.8393-0.8517.
#   BEATS THE TWIN — solo >= 0.8517. Supervised IN-21k features transfer BETTER here than DINOv3's at
#     fixed architecture, which would make the objective a lever on COMPETENCE and reopen every trunk
#     decision this project made by reading a self-supervised checkpoint's number.
#   MATCHES THE TWIN — 0.8393-0.8516. The objective does not move competence, and the seat value is then
#     the entire question: a matched solo with a POSITIVE seat says the objective moves the trunk's
#     OPINION and not its competence, which is exactly the diversity `knee-m69u` is asking for and the
#     first mechanism that could supply it without costing strength by construction.
#   COSTS — 0.8000-0.8392. Supervised pretraining transfers worse to this task at fixed architecture.
#     Still price the seat before concluding: `base_dense` lost its control outside the floor and won
#     the board anyway, so a solo below the band is not a verdict about the member.
#   REFUTES — < 0.8000, at zero architectural distance. That closes the objective axis outright rather
#     than leaving it confounded in `swin_supervised`, and it would make the y59c curve a statement
#     about pretraining DATA SCALE rather than about architecture.
#
# DO NOT ARM THIS WHILE `swin_supervised` HOLDS THE CARD. Two of our jobs on one device stretched a
# three-minute member to eleven once already; `card_check` blocks on exactly that and is the gate here.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train vit_in21k \
  $RECIPE --epochs 12 --batch 2 --accumulate 3 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant vitin21k --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$PROCESSED/vit_in21k_s2026.pt"
