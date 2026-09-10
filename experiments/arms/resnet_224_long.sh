#!/usr/bin/env bash
# resnet_224_long.sh — the same CNN at the epoch budget the foreign recipe actually asks for.
#
# bead: knee-7niv7
# reading: resnet_224_long_s2026.holdout.npz
#
# ONE VARIABLE AGAINST `resnet_224_short`: `--epochs 48` instead of 12. Every other flag is byte-identical,
# so the pair is a matched two-point read of the epoch axis and no third arm is needed to interpret it.
# `--one-cycle` is defined over total steps, so 48 epochs is a genuinely longer schedule and not the same
# schedule run four times.
#
# WHY LENGTH IS ITS OWN QUESTION HERE AND NOT ON OUR SEATS. Twelve epochs is the ledger's budget because
# that is where a lightly-adapted ViT peaks — `--backbone-lr` 4e-6-4e-5 moves the trunk barely at all, so
# most of the fitting is a head on frozen-ish features and it converges early. This arm moves the WHOLE
# trunk at 2e-4 from ImageNet weights, which is a different optimisation problem with a much longer
# runway. The foreign recipe's own budget is the evidence: it asks for 48, and nobody here has ever
# checked whether the 12 our ledger standardised on is a fact about the task or a fact about our recipe.
#
# GATE: DOES NOT RUN unless `resnet_224_short` reads >= 0.80. A four-times-longer run on a trunk whose
# short arm missed the band would be four hours spent confirming a recipe defect.
#
# PRE-REGISTERED, on the 909 at the ledger cut, against band224 and against its own short twin:
#   >= 0.84            the CNN lane REOPENS. The follow-up is a seat test on the champion's vote, and
#                      the 5-fold twin D11 asks for becomes affordable as the project's per-column ruler.
#                      [HIS ruler: 0.875]
#   0.80 - 0.84        directional. The lane is alive and under-powered; the member is worth a seat test
#                      only if it decorrelates, which is a separate reading of the same npz.
#   long - short >= 0.015   "longer" is REAL on this trunk, independent of where the absolute lands, and
#                      that is a fact about our 12-epoch standard that transfers to every whole-stack arm.
#   < 0.80             recipe suspect, NOT the trunk. A trunk that gets worse with four times the budget
#                      at a fixed rate diverged; read the loss curve before the score.
#
# Noise floor 0.0062 paired, so only the >= 0.84 band and the 0.015 gap clear it. The other two are
# readings about WHERE TO LOOK NEXT, and they are written down as such.
#
# PRICE: ~2 h 30 at 48 epochs, measured — the short arm ran 185 s/epoch at batch 4 and this arm is the
# same geometry. No submission slot. Report the s/epoch WITH the score.
#
# THE BATCH WAS NOT BYTE-IDENTICAL AND THAT IS WHAT FROZE THE MACHINE (corrected 2026-09-10). The line
# above read `--batch 8 --accumulate 1` while `resnet_224_short.sh` reads `--batch 4 --accumulate 2`, so
# the "ONE VARIABLE ... every other flag is byte-identical" claim at the top of this file was FALSE and
# the pair was never the matched two-point read it advertises. The short arm's own header records why it
# moved: the card DERIVED 50887 MiB against 30991 free at batch 8, so batch dropped to 4 with
# `--accumulate 2` holding the effective batch at his 8. That fix was never propagated here. The 2026-09-09
# attempt therefore ran at batch 8 and logged `peak memory 46.94 GB reserved` against a 32 GB card — it was
# spilling into shared system memory, the machine became unusable, and the owner killed it at epoch 9 of 48.
# The freeze was this flag, not the length of the run.
# BatchNorm is unaffected, by the short arm's own arithmetic: a study is 20 slices and BN sees IMAGES, so
# batch 4 gives it (480, ...) — an order of magnitude past the 32-per-worker where Wu & He's curve is flat.
#
# deliberate: --batch 4 --accumulate 2 (matches resnet_224_short exactly, which is what makes the epoch
#             axis the ONE variable this arm claims to move)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train resnet_224_long \
  $RECIPE --epochs 48 --variant resnet50d --image-size 224 $THROUGHPUT \
  --through-plane --slice-pooling max --slices 20 --band-low 0.1 --band-high 0.9 \
  --whole-stack --unfreeze-last 16 --backbone-lr 2e-4 --head-lr 2e-4 --weight-decay 0.05 \
  --batch 4 --accumulate 2 \
  --labels "$LABELS" --out "$MODELS/resnet_224_long_s2026.pt"
