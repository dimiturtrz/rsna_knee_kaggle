#!/usr/bin/env bash
# llrd224.sh — native224's recipe with a per-depth learning rate over the six opened blocks, at a top
# rate chosen so the LADDER'S MEAN RATE EQUALS THE CONTROL'S. `llrd224_flat.sh` is the second point,
# identical but WITHOUT the rate compensation. Read the two together; the pair exists to resolve a
# confound the ledger already refused to read past.
#
# bead: knee-s9cr
#
# WHAT THIS ARM IS FOR, in one sentence: `band_dense_llrd` is the only layer-decay measurement this
# project owns, the ledger REFUSED its verdict in writing, and nothing has re-run it since.
#
# THE PRIOR MEASUREMENT AND WHY IT IS NOT AN ANSWER. `band_dense_llrd` (README:2074-2115, arm
# `experiments/arms/band_dense_llrd.sh:58-60`) read 0.8483 against `band_dense`'s 0.8578 — −0.0095, past
# the floor and the wrong way — and its own pre-registered ladder said a BELOW result "closes the depth
# axis for good". The ledger then refused that reading and said why: the arm moved TWO variables.
# `EncoderSurgery.backbone_groups` (`kneemri/models/encoder_surgery.py:236-244`) anchors the ladder at
# the TOP opened block and divides everything below it down, so `--unfreeze-last 12 --layer-decay 0.7`
# did not add six slow blocks to a control that trained six at 4e-5 — it added six slow blocks AND cut
# the control's own six to a geometric mean of 0.7^2.5 = 0.41 of their rate. The ledger's own diagnosis:
# "TRAINING is the red link", the loss floored 0.13 high while still descending, "the signature of too
# little rate, not of wrecked early blocks". So what was measured was an UNDERFIT, and the depth axis
# was never tested. The key is nonetheless counted as varied by `switch_coverage.py:79-81`, which scrapes
# `--flag` tokens out of the shell and cannot see a confound.
#
# HOW THIS ARM REMOVES THE CONFOUND. Two changes to the prior design, both forced by that diagnosis.
# (1) The ladder length is left at the DEFAULT `unfreeze_last = 6` (`trainer.py:68`), which is exactly
# the set of blocks the control trains — so no block that was frozen in the control becomes trainable
# here, and the only thing that moves is how the SAME six blocks are graded. (2) The top rate is raised
# so the ladder's geometric-mean rate reproduces the control's flat 4e-5: with six blocks and decay 0.75
# the factors are 1, 0.75, 0.5625, 0.4219, 0.3164, 0.2373, whose geometric mean is 0.75^2.5 = 0.487, so
# the top block runs at 4e-5 / 0.487 = 8.2e-5 and the bottom at 1.95e-5. The arithmetic mean gives
# 7.3e-5 instead; both compensations land inside the same measured plateau, so the choice of statistic
# does not straddle a boundary. That plateau is measured, not assumed: `rate_1e-4` read 0.8490 (+0.0033
# on `lr4e-5_ws`, inside the floor) and `rate_2e-4` collapsed to 0.7391, so 8.2e-5 is comfortably below
# the edge and 1.95e-5 is comfortably above the 8e-6 that `rate_isolate` showed was too slow to move a
# block. The whole ladder sits inside rates this project has already characterised.
#
# THE MECHANISM. `backbone_groups`' own docstring states the entanglement: at one flat rate "depth and
# rate can only be measured along a diagonal", and raising the rate until the top blocks move applies
# that same rate to the earliest blocks, "which hold the most general features and are the most damaging
# to disturb". A 4e-5 flat over six blocks is that diagonal. LLRD is the transfer literature's standard
# separation of the two, and it is the single most likely candidate for an unnamed fine-tuning "tweak"
# on a ViT trunk — which is what this session is hunting.
#
# WHY THIS IS NOT COVERED BY THE 224 CLOSURE. `mixup224_long` (README:4178-4215) closed every axis that
# "perturbs how the SAME supervision is presented to the SAME encoder". This changes the optimiser's
# path over the encoder's depth. Nothing about the supervision or its presentation moves.
#
# WHY 224. `native224` read the in-plane axis flat (0.8502 vs 0.8525) at 94 s an epoch against 290, and
# its standing instruction is that matched comparisons run at 224 from here. The prior measurement cost
# ~580 s an epoch at batch 2 / accumulate 6; the corrected PAIR costs less than that one arm did.
#
# control  native224 = 0.8502, best at epoch 7, converged not truncated. 12 epochs, 12 slices, 0.20-0.80
#          band, whole-stack, 224 px, six opened blocks at a flat 4e-5, head 1e-3, same holdout. NOT
#          band_dense and NOT llrd224_flat: the control is the flat-rate arm at the same mean rate.
# cost     ~19 min for twelve epochs at 94 s, ~8124 MiB reserved. The ladder emits six optimiser groups
#          instead of one and costs nothing measurable for it.
# price    19 min for P(clears the 0.0062 floor) = 0.30. The axis is genuinely untested rather than
#          refuted, its one reading is explained by a named second variable that this design removes,
#          and rate is the only axis in this ledger that has ever moved the number by more than the
#          floor in either direction.
#
# HARNESS CHECK BEFORE READING ANY NUMBER. The printed recipe must say "layer_decay": 0.75,
# "backbone_lr": 8.2e-05 and "image_size": "224". `Trainer.encoder_rate_decay` (`trainer.py:166`) is a
# `Landed`-compared field read off the OPTIMISER's top two backbone groups rather than off the flag that
# asked for it, so the run cannot pass with the decay unapplied — the measured ratio must read 0.75.
# That check is the reason the prior arm's plumbing was green and its TRAINING was not.
#
# THE FALSIFIABLE PART, written before the run, READ AS A PAIR with llrd224_flat.
#   CONFIRMED   the peak clears 0.8564 AND llrd224_flat does not. That is the only outcome that isolates
#               the ladder SHAPE from the mean rate, and it re-opens the depth axis `band_dense_llrd`'s
#               own ladder tried to close — at which point `--unfreeze-last 12` with compensation is the
#               obvious next arm and the 336 transport is worth a slot.
#   DIRECTIONAL positive under the floor with the flat point below it. Kept as free and mechanism-sound;
#               not a finding, no confirmation seed.
#   REFUTED     the peak falls by more than the floor WITH the training loss tracking the control's
#               curve. That combination — a matched fit that generalises worse — is a real refutation of
#               grading these six blocks by depth, because the underfit explanation is excluded by
#               construction here.
#   NOT A VERDICT, and pre-registered as such: the peak falls AND the training loss floors high while
#               still descending, the exact signature the prior arm produced. That would mean the
#               geometric-mean compensation is the wrong statistic and the arm is underfit again, and it
#               is read as a harness failure of THIS arm, not as evidence about layer decay.
#   BOTH FLAT   the mean rate is what the encoder responds to and its distribution over depth is not.
#               Closes an axis the ledger currently records as closed-with-a-refused-verdict, which is
#               worth the pair on its own.
#
# deliberate: --layer-decay 0.75 (the arm's thesis — the six opened blocks graded by depth)
# deliberate: --backbone-lr 8.2e-5 (NOT a second variable: the compensation that holds the ladder's
#             geometric-mean rate at the control's 4e-5, which is precisely what band_dense_llrd omitted)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train llrd224 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 8.2e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --layer-decay 0.75 \
  --labels "$LABELS" --out "$PROCESSED/llrd224_s2026.pt"
