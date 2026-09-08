#!/usr/bin/env bash
# latent_adapt.sh — does the TARGET cause the collapse, or does the MASKING?
#
# reading: latent_dense_s2026.holdout.npz
# deliberate: objective latent, at the rate and epoch count that destroyed the pixel one
# bead: knee-lm37v
#
# WHAT THIS ANSWERS. Every masked adaptation this project has run destroyed the encoder's routing.
# Researcher D251/D252 measured five checkpoints at n=240: four of four adapted encoders degenerate,
# `adapt_null` (`--lr 0`) reproducing stock to three decimals. `pretrain_unitfix` drove its loss
# 0.9122 -> 0.5570 over eight epochs and was read as an objective fitting for a fortnight; it was an
# encoder switching attention off. Two shapes, both fatal: blocks 8-11 at 5.372-5.532 against
# `ln(257) = 5.549` on one checkpoint, block 11 at 0.000-0.035 nats on the other three.
#
# Rate is already excluded as the lever. `adapt_small_e6` collapsed at 1e-4 and `adapt_small_e6_lr1e5`
# collapsed at 1e-5, so a tenfold cut bought nothing. That leaves TWO candidates, and they are separable:
# the masking (predicting anything from a corrupted input at this scale) or the TARGET (regressing raw
# pixels and their spectrum, which rewards a texture average and needs no routing to produce one).
#
# THE MATCHED SMOKE ALREADY SPLIT THE EARLY TRANSIENT AND FOUND NOTHING. Three optimiser steps at 5e-5,
# identical masking, identical seed, identical slices: the pixel objective ends at 0.63 of stock's own
# room, the latent objective at 0.65. The target does not change the first three steps. But D252's
# collapse arrived at epoch 6-8, not step 3 — so the transient is not the question and this arm is.
#
# ONE VARIABLE against `pretrain_unitfix`: `--objective latent` in place of `spectral`. The rate stays
# 1e-4, the epochs stay 8, the slices-per-step stay 96, the cache and the seed are unchanged. The rate
# is deliberately the one that destroyed the pixel run — a latent arm that survives AT THE DESTRUCTIVE
# RATE says the target carried it, and a latent arm run somewhere safer would say nothing at all.
#
# THE GATE IS LIVE AND MAY END THIS ARM. `kneemri.training.attention_health` measures stock at epoch
# zero on held slices and refuses any epoch where a block keeps under 0.25 of stock's own distance to
# either end. Sized on D252: every degenerate cell there keeps under 0.115, stock keeps 1.00. A refusal
# is the arm's SECOND pre-registered outcome, not a crash, and it costs minutes instead of the forty
# the pretrain would otherwise spend proving what the gate already saw.
#
# PRE-REGISTERED, three outcomes:
#   1. GATE REFUSES. The masking is the cause, the target is not, and the whole masked-adaptation axis
#      closes on a mechanism rather than on a tally of flat scores.
#   2. GATE PASSES, fine-tune >= 0.8578. The target was the cause. Latent MIM is the adaptation path
#      this project has been unable to find, and the rate ceiling reopens.
#   3. GATE PASSES, fine-tune indistinguishable from the control. Routing survives and buys nothing —
#      which brackets the axis (the damage was never what capped the gain) and points at the data.
#
# control  band_dense = 0.8578 (best epoch 8), byte-identical fine-tune with the flag off.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
ADAPTED="$MODELS/adapt_latent_e8.pt"
require "$LABELS"

"$REPO/.venv-arm/Scripts/python.exe" -m kneemri.training.pretrain \
  --objective latent --epochs 8 --slices-per-step 96 --lr 1e-4 --seed 2026 \
  --out "$ADAPTED" 2>&1 | tee "$MODELS/experiments/adapt_latent_e8.full.log"

require "$ADAPTED"

run_train latent_adapt \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --encoder-weights "$ADAPTED" \
  --labels "$LABELS" --out "$MODELS/latent_dense_s2026.pt"
