#!/usr/bin/env bash
# mricore_rate.sh — `mricore` at 4e-4, the learning rate two other families independently peaked at.
#
# WHY THIS ARM EXISTS. The encoder axis was believed to be 4-6x the knob axis, on D190 §3's two rows
# (convnext -0.0262, dinov3-base +0.0184). Priced over 217 canonical-frame members at a MATCHED recipe
# (D194 §3), it is not: DINOv2's best against the best alternative family is 0.0123 @base224 (n=27) and
# 0.0143 @base_dense (n=7), against a within-DINOv2 knob spread of 0.0144 / 0.0086 at those same
# recipes. The 4-6x is the distance to `radiodinos8`, a patch-8 SMALL trunk — a sample of capacity, not
# of the trunk axis. Controlled for recipe, the encoder axis is KNOB-SIZED and below the 0.015 bar.
#
# WHAT IS UNPRICED IS THE LEARNING RATE. Six of thirteen families were trained at exactly ONE rate, and
# five of those six at `4e-5` — DINOv2's default, which nobody chose for them; it was the value in the
# arm that was copied. The two families that DID get a sweep both peak an order of magnitude above it:
#
#   convnext @336/20sl/12ep    4.0e-5 -> 0.7927   1.2e-4 -> 0.7993   4.0e-4 -> 0.8186   1.2e-3 -> 0.7762
#   resnet50d                 1.26e-4 -> 0.8196                      4.0e-4 -> 0.8393
#
# +0.0259 and +0.0197 from the borrowed default to the same 4e-4, measured independently. The mechanism
# is unremarkable, which is what makes it credible: `4e-5` was tuned for a self-supervised ViT at
# `unfreeze_last=6`, and a differently-pretrained trunk wants more.
#
# WHY `mricore` AND NOT ANOTHER OF THE FIVE. It is the best of them — 0.8390, 0.0295 behind DINOv2's
# best — on ONE arm, at the borrowed rate, and `convergence_check` reads it UNDERTRAINED: it peaked at
# epoch 11 of 12, so the budget bound it rather than the recipe. Of the alternatives, `swin` is flat and
# closed (3 recipes, 2 LRs, +0.0003 total), `rad-dino` moved +0.0014 across its pair, `radimagenet` is
# closed EMPTY at `foreign_encoders.py:56`, and `convnext` is the sweep this argument is BUILT on, with
# a ceiling of 0.8186. `vjepa2` is the runner-up and stays unarmed for now: two recipes but one LR, so
# its +0.0007 climb is not evidence about this axis at all.
#
# THE SINGLE VARIABLE IS THE RATE. Every other flag is byte-identical to `mricore_corpus.sh` — same
# 336px, same 20 slices, same 0.1-0.9 band, same 12 epochs, same batch 2 / accumulate 3, same labels.
# Epochs are NOT raised despite the UNDERTRAINED reading: a higher rate converges sooner, and moving the
# budget too would make a gain unattributable between the two. If 4e-4 also peaks on its last epoch,
# THAT is the arm that raises the budget.
#
# control  mricore_corpus_s2026 = 0.8390 @midpoint/n909/e11, on disk, same fold, same recipe minus this
# one flag. No matched control run is needed and none should be queued.
#
# PRE-REGISTERED, and the transfer is an ANALOGY not a measurement — convnext and resnet50d are both
# convolutional and `mricore` need not be, which is the weakest link in the whole argument and the
# reason this is worth RUNNING rather than worth assuming:
#   LANDS   — 0.8590 or above (+0.0200, the low end of the two measured transfers). The borrowed default
#     was the binding constraint on this trunk, and the same question is then owed to `vjepa2`, `medsam`,
#     `radiodino` and `raddino`, which is four more card-hours and a real axis.
#   PARTIAL — +0.0060 to +0.0200. Directional, above the paired floor and below the bar. Rate matters
#     here and less than it did for the convolutional pair; the follow-up is `vjepa2`, not a second
#     mricore rate.
#   FLAT    — within +/-0.0060. The transfer does not hold off the convolutional families, and the
#     one-LR hole in D194 §4 stops being a lead. Do not sweep the other four on this argument.
#   REFUTES — 0.8330 or below. 4e-4 is past this trunk's optimum, which would say the convnext curve's
#     shape (peak at 4e-4, collapse by 1.2e-3) is narrower than assumed and the right probe for the
#     remaining families is 1.2e-4, not 4e-4.
#
# THIS IS AN INSTRUMENT, NOT A SEAT. The honest ceiling on the LANDS branch is LEVEL with DINOv2's
# best, not past it, and the member axis is closed on the board with seven flats. What it buys is a
# strongly diverse member and an answer about a five-family hole — not an entry.
#
# reading: logs/mricore_rate_ledger.log
# bead: knee-m69u
#
# deliberate: --backbone-lr 4e-5 -> 4e-4 is the ONE variable. Every other flag is mricore_corpus's and
#   may not be touched in this arm.
# deliberate: --epochs 12 unchanged despite the UNDERTRAINED reading on the control, so a gain is
#   attributable to the rate and not to the budget.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train mricore_rate \
  $RECIPE --epochs 12 --batch 2 --accumulate 3 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant mricore --backbone-lr 4e-4 --slices 20 --band-low 0.1 --band-high 0.9 \
  --labels "$LABELS" --out "$MODELS/mricore_rate_s2026.pt"
