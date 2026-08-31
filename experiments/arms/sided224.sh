#!/usr/bin/env bash
# sided224.sh — native224's recipe with `--pool sided` and nothing else moved. The first of four arms
# that transport the AGGREGATION family onto the 224 control; the family has never once run at 224.
#
# bead: knee-7d53
#
# WHY THE FAMILY AND WHY NOW. `experiments/README.md:5024-5075` splits fourteen arms by WHAT they
# changed and the split is total: every arm that gave the model MORE INPUT (more pixels, more slices, a
# tighter crop) lost on the six model-bottlenecked findings, and every arm that changed HOW THE SLICES
# ARE AGGREGATED gained — `sided_pool` +0.0076, `attend_dense` +0.0059, `depth_regions` +0.0016,
# `through_dense` +0.0005, four unrelated mechanisms, no exceptions in either direction. All four ran at
# the dense read (20 slices, 0.1-0.9, 336 px). Every matched comparison since `native224` runs at 224 at
# a third of the card, and the whole positive half of that table is unmeasured there.
#
# THE MECHANISM, and it is about WHERE in the field a response sat, not how much of the field there was.
# Every `TokenPooling.reduce` is typed `"n t d" -> "n f"` (`kneemri/models/token_pooling.py`): the head
# receives one vector per slot and cannot express laterality, so a response in the medial compartment and
# the same response in the lateral one reach the classifier identically. `ClassAndSidedMean` splits the
# square patch grid down the midline column and returns three parts — class token, left mean, right mean.
# `Laterality` mirrors 100% of the corpus, so a column split IS the medial/lateral split and half the
# label vocabulary (Lateral vs Medial Meniscus, Lateral vs Medial OA) is a laterality question the pooled
# vector currently answers by averaging both answers together.
#
# THE RESOLUTION ARGUMENT, WHICH IS WHAT THIS ARM IS FOR. I predict sided pooling matters MORE at 224,
# not less, and the reason is that its gain is a function of the SPLIT and not of patch size. 224/14 = 16
# gives a 16x16 = 256-token grid (still square, so `ClassAndSidedMean` accepts it; it raises on a
# non-square token count) against 24x24 = 576 at 336. A focal structure occupies a fixed extent in mm, so
# it covers ~2.25x FEWER patches at 224 — but it also occupies the same FRACTION of the field, and a mean
# over the half-grid dilutes by fraction, not by count. What the downsample actually costs is per-token
# contrast (each patch embedding averages the lesion with more normal tissue), and that cost falls on
# every pooling equally, control included. What it takes away asymmetrically is the head's ability to
# recover laterality IMPLICITLY: with 256 tokens instead of 576 collapsed into one vector, there is less
# residual spatial structure for a global mean to leak, so making the split EXPLICIT should carry more.
# The refutation of that argument is a 224 reading at or below the control against +0.0076 at 336, and
# that reading is worth its nineteen minutes: it would say the sided gain is a fine-grid effect that the
# native read cannot buy, which prices every future 224-vs-336 member decision.
#
# control  native224 = 0.8502, best at epoch 7, converged not truncated. 12 epochs, 12 slices, 0.20-0.80
#          band, whole-stack, backbone 4e-5, 224 px, same labels, same 909-study holdout, token pooling at
#          the default. The single variable is --pool.
# cost     ~19 min for twelve epochs at 94 s. Same cache as native224 (12 slices, 0.2-0.8, 224), so no
#          decode; the pooling is a head-side reduction and adds no encoder pass.
# price    19 min for P(clears the 0.0062 floor on the MEAN) = 0.15. The dense reading was +0.0076 on the
#          focal SIX and +0.0036 on the mean — under the floor on the column the board scores — so a
#          clear on the mean here needs the resolution argument above to be not just right but larger
#          than the effect it transports. P(reads positive on the focal six) is much higher, ~0.5, and
#          that is the reading this arm is actually bought for.
#
# HARNESS CHECK BEFORE READING ANY NUMBER. The printed recipe must say "pool": "sided" and
# "image_size": "224", and the log must load 4,406 studies and hold out 909. A pooling that never reached
# the model produces a run byte-identical to the control; a delta of exactly 0.0000 is a PLUMBING
# verdict, not a null. `ClassAndSidedMean` returns three parts where the default returns two, so the head
# width changes — a run whose parameter count matches native224's did not switch.
#
# THE FALSIFIABLE PART, written before the run.
#   CONFIRMED   the mean peak clears 0.8564 (control + floor). Sided pooling survives the native read and
#               the aggregation family is the axis to spend the rest of the window on.
#   DIRECTIONAL positive under the floor, focal six at or above +0.0076. Kept as free and mechanism-sound
#               — it costs no encoder pass — and it feeds `fuse_attend`; not a finding on its own.
#   TRANSPORT LOSS  focal six positive but visibly under +0.0076. The mechanism holds and the coarser
#               grid costs part of it; this is the reading that prices 224 against 336 for the roster,
#               and it is a real answer rather than a null.
#   REFUTED     the mean falls by more than the floor. The three-part head is wider on a 256-token grid
#               that has less to say per token, and paying head parameters for a split the field can no
#               longer resolve is a real way to lose. That would close sided pooling AT 224 — not at 336,
#               where it is measured positive, and the ledger must say which.
#
# deliberate: --pool sided (the arm's thesis — an explicit medial/lateral split at the native read)
# deliberate: --image-size 224 (native224's control, per that row's standing instruction)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train sided224 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 \
  --image-size 224 --pool sided \
  --labels "$LABELS" --out "$PROCESSED/sided224_s2026.pt"
