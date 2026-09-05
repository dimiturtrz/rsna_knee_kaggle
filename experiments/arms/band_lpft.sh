#!/usr/bin/env bash
set -euo pipefail
#
# band_lpft - LP-FT on the INCUMBENT recipe, at the incumbent's own rate, so the only variable is the head.
#
# WHAT THIS CHANGES. Stage two is `band_dense` with one flag added: `--init` pointing at stage one's
# checkpoint. Same trunk, same 20 slices, same 0.1-0.9 band, same max slice pooling, same whole stack,
# same 4e-5 backbone rate, same 12 epochs. Stage one is that same recipe with `--unfreeze-last 0`, which
# fits the head alone over a FROZEN trunk. So the pair reads one thing: does the head the trunk meets on
# the first gradient step matter.
#
# WHY THIS ARM AND NOT `ortho_lpft` AGAIN. `ortho_lpft` asked the same question and cannot answer it.
# It moved TWO variables off its rate point - the warm start AND a backbone rate an order lower
# (1e-5 against `ortho_dense`'s 4e-5) - so its control was never `ortho_dense` but the unrun
# `ortho_fresh_lowlr`, and the triple is still missing that leg. It also died at epoch 4 of 12
# (`convergence_check`: TRUNCATED, best e4 of 5 run), so its 0.7937 is a FLOOR and not a reading.
# Completing that triple costs two arms on `ortholarge`, whose direct fine-tune reads 0.8452 - below
# the incumbent this arm tests. This arm costs ONE, its control is already on disk, and the recipe it
# would change is the one every strong member here is built from.
#
# THE MECHANISM, and it is measured rather than argued. Kumar et al. (arXiv:2202.10054) prove and then
# measure that fine-tuning from a RANDOMLY INITIALISED head distorts pretrained features: the first
# gradients are large and arbitrary, and they move the trunk about twenty times further than the same
# fine-tune started from a fitted head (Living-17: ID features move 0.0188 under FT against 0.0011 under
# LP-FT). Their headline is +1% in-distribution and +10% out-of-distribution across ten shift datasets,
# and they show early stopping does NOT recover it - the FT trajectory never passes through a point that
# matches LP-FT's OOD accuracy. Davila et al. (arXiv:2406.10050) sweep three rates over three CNNs across
# five medical modalities and find LP-FT beats direct fine-tuning in more than half the pairs.
#
# THE OOD HALF IS THE PART WORTH THE CARD HERE. Our holdout is 909 studies drawn from the same corpus we
# train on, so it reads Kumar's ID axis, where the published gain is ~+1% - under this project's 1.5%
# bar and only just over the 0.0062 floor. The BOARD is the OOD axis: a different institution's studies,
# which is exactly the shift their +10% is measured on. So a holdout gain at or a little over the floor
# is the SIGNAL, not the prize, and the prize is only readable on a slot.
#
# THE PRICE, before it runs. Stage one is 12 epochs with no backward pass through the trunk (patience 3
# under that ceiling, so it usually stops earlier - `ortho_lpft_head` ran all twelve and was still
# climbing at 0.7537). Stage two is `band_dense`'s 361 s x 12 = 1.20 h. Call it ~1.9 h all in. The
# plausible gain is Kumar's ID +0.010 at maybe two-in-five odds of clearing 0.0062 - so this arm is NOT
# sold as a 1.5% result on its own. It is priced as a RECIPE change: it applies to every member this
# project fits from here on, and to a reseat of the champion's own seats, which is the only way a
# sub-bar per-member delta reaches the bar.
#
# control  band_dense = 0.8578 (best epoch 8), byte-identical flags except `--init`. That is the whole
#          reason this arm is on `band_dense` and not on a trunk whose control does not exist yet.
#
# PRE-REGISTERED against band_dense = 0.8578, written before the run, with the 0.0062 holdout floor
# setting every boundary:
#
#   s >= 0.8700            LP-FT CLEARS THE BAR ON THE HOLDOUT ALONE, which is above what Kumar's ID
#              number predicts. Re-fit the champion's seats under it and price the reseat offline before
#              any slot; a gain this size on the ID axis says the OOD one is worth a board read.
#   0.8700 > s >= 0.8640   THE PUBLISHED ID GAIN REPRODUCES. Directional, not a finding, and that is the
#              honest reading of +0.006 to +0.012 against a 0.0062 floor. It licenses ONE thing: an
#              instrument slot that reads the OOD half, because that is where the mechanism's larger
#              number lives and no offline ruler here can see it.
#   0.8640 > s >= 0.8516   FLAT. The head initialisation does not matter at this rate on this trunk, and
#              the ~20x feature-distortion result does not reach a DINOv2 trunk fine-tuned at 4e-5 with
#              only its trailing blocks open. LP-FT CLOSES as a recipe change here; `ortho_lpft`'s triple
#              stays open only as a question about FOREIGN trunks, where the distortion gap was 0.148.
#   s < 0.8516             Worse than the control by more than the floor. SUSPECT THE WARM START BEFORE
#              THE MECHANISM, in this order: that `--init` loaded (the recipe line must record it and
#              `WarmStart` loads strictly, so a partial load raises rather than lands), that stage one's
#              head was not stopped so short it is worse than random, and that stage one's frozen-probe
#              holdout is a sane number rather than a collapse. Only after those does this branch say
#              anything about LP-FT.
#
# VERIFY THE FLAG LANDED BEFORE READING ANY SCORE. Stage two's recipe line must carry `init` naming
# band_lpft_head.pt, with `backbone_lr` 4e-05 and `unfreeze_last` at the control's value. Stage one's
# must carry `unfreeze_last` 0 - a stage one that quietly trained the trunk is a slow fine-tune wearing
# LP-FT's name, and it would read as a null on a mechanism that was never tested.
#
# deliberate: --init is the ONE variable this arm moves off band_dense. The rate is deliberately NOT
# lowered, which is where `ortho_lpft` went wrong: Kumar's mechanism is about the head the trunk meets,
# not about how fast it moves afterwards, and lowering the rate alongside makes a gain unattributable.
#
# reading: logs/band_lpft_ledger.log
# bead: knee-cyk91 (this arm; knee-6hfi is the OrthoDiffusion smoke and was a stale citation)
# also: knee-a2q7, knee-cj40f
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band_lpft_head \
  $RECIPE --epochs 12 --patience 3 --slice-pooling max --whole-stack $THROUGHPUT \
  --unfreeze-last 0 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$MODELS/band_lpft_head.pt"

run_train band_lpft \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --init "$MODELS/band_lpft_head.pt" \
  --labels "$LABELS" --out "$MODELS/band_lpft_s2026.pt"
