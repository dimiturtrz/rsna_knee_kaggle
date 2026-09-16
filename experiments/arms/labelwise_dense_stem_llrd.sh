#!/usr/bin/env bash
# bead: knee-p6fr8
# reading: labelwise_dense_stem_llrd_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `labelwise_dense_stem`'s line plus `--layer-decay 0.75`. It is the follow-up that arm's own
#   pre-registration named before it ran: "if the loss curve diverges from the control's inside the
#   first two epochs the rate is the suspect, not the axis -- the honest follow-up is layer_decay".
#   The curve diverged at e1 (0.7605 against the control's 0.7772) and kept diverging, so this is the
#   named re-test, not a second guess at the same question.
#
# WHY THE RATE IS THE SUSPECT AND WHY THIS FLAG IS THE ANSWER
#   `encoder_surgery.py:443` gives the stem `backbone_lr * layer_decay ** len(opened)`. The stem arm
#   ran at `layer_decay 1.0`, so that product was 4e-5 -- the stem, freshly unfrozen and never once
#   trained across 296 arms, stepped at the full trunk rate. The stem is also the widest-fan-out tensor
#   in the trunk, so each of those steps moved every token every block saw. At 0.75 with six blocks
#   open the stem lands at 7.1e-6, a 5.6x cut.
#
# WHY 0.75 AND NOT A SWEPT VALUE
#   Layer-wise LR decay for ViT fine-tuning is published at 0.65-0.75 (BEiT, and DINOv2's own recipe).
#   0.75 is the MILDEST of that band, chosen because the flag is not stem-specific: there is no stem LR
#   on `TrainConfig`, so every value that slows the stem also slows the six open blocks. The mildest
#   published value buys the largest stem cut per unit of trunk slowdown.
#
# THE CONFOUND, STATED
#   This arm is two flags from the control and one from `labelwise_dense_stem`. A gain here is
#   therefore attributable to "stem at a lower rate" ONLY against the stem arm's own curve, which is
#   the comparison this is read on. Against the control it also carries the block decay, and that
#   reading needs a decay-without-stem arm that nothing owes yet.
#
# PRE-REGISTRATION, read at e3 against the two curves already on disk
#   stem arm e0..e3   0.7311 / 0.7605 / 0.7657 / 0.7729   (layer_decay 1.0)
#   control  e0..e3   0.7305 / 0.7772 / 0.7995 / 0.8193   (no stem, layer_decay 1.0, peak 0.8672)
#   tracks the CONTROL   the rate was the whole story. Let it run; price the peak against 0.9048.
#   between the two      the rate is part of the story. Let it run only if trajectory_check can reach.
#   tracks the STEM ARM  the rate is NOT the story and the stem axis closes for real at this depth.
#                        That closure is the arm's product and is worth the card on its own.
#
# deliberate: every other flag inherited VERBATIM from labelwise_dense_stem, which inherited them from
#             labelwise_dense_drop40_long. Three runs on one recipe line, one flag apart each.
#
# PRICE: ~2.7 h to a peak, ~0.5 h to the e3 verdict that trajectory_check gates.
# P(clears +0.0040) ~ 0.25.
#
# control  labelwise_dense_drop40_long_s2026 = 0.9048 ledger75 (on disk). No control run is owed.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train labelwise_dense_stem_llrd \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --slot-dropout 0.40 --unfreeze-stem --layer-decay 0.75 \
  --labels "$LABELS" --out "$MODELS/labelwise_dense_stem_llrd_s2026.pt"
