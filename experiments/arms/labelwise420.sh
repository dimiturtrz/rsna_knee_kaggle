#!/usr/bin/env bash
# labelwise420.sh — the resolution axis re-read ABOVE where we already ship.
#
# bead: knee-ss5rd
# reading: labelwise420_s2026.holdout.npz
#
# THE CELL THIS PROJECT ACTUALLY OWES. AXES.md section B disowns `resolution_420`, `extended_slots`,
# `multiscale` and knee-ezz: each was measured on a MAX-pooling head, which demonstrably suppresses a
# real input change, so all four are recorded UNREAD rather than refuted. The sanctioned re-open is to
# re-run ONE of them under `--slice-pooling labelwise`, one axis at a time, against its own labelwise
# control. Swept all 490 arms 2026-09-22: 420 and 448 exist ONLY under `max` (fovjitter160,
# fovjitter160_reflect, crop160_span, intensity_augment). labelwise x 420 does not exist.
#
# WHY NOT 336, WHICH IS WHERE THE FIRST DRAFT OF THIS ARM POINTED. Because 336 is the DEFAULT
# (`IMAGE_SIZE = 336`) and the whole shipping family already sits there: labelwise_dense 0.8964,
# labelwise_dense_drop40_long36 0.9030, chaindense 0.8904, all 336. A 224-vs-336 arm would have bought a
# clean per-unit slope for a step this project took months ago, at two hours of card, and changed
# nothing about what runs next. 336 -> 420 is 1.56x the pixels ABOVE the shipping corner, and it is the
# cell no head that can use it has ever seen.
#
# MECHANISM. At 336 over a ~140 mm FOV a 1-3 mm meniscal tear spans ~3-8 px; at 420, ~4-9. The claim is
# not that this optimises a representation of the finding — it is that the finding is closer to
# REPRESENTABLE. Our model-side error census is led by exactly the focal labels that starve: Lateral
# Meniscus -0.1925, ACL -0.1225, MCL -0.0964. A labelwise head is the first readout that can route those
# pixels PER FINDING instead of through one query pulled twelve ways, which is why the same input change
# read flat through `max` and why that flat read is not evidence about the input.
#
# EVIDENCE AGAINST, STATED BEFORE THE RUN. `focal_labelwise224` finished 2026-09-22 at 0.8796 against
# labelwise224's 0.8832 — the focal top-k patch tail is inert under a labelwise head too. That is a
# SELECTION mechanism (re-weight tokens the model already holds) rather than a SAMPLING one (tokens that
# do not exist at the lower resolution), so it does not refute this arm, but it is correlated evidence
# that this head is not token-starved. It is why P below is 0.20 and not the 0.30 first written down.
#
# ONE VARIABLE against `labelwise_dense`: `--image-size 420`. Byte-identical otherwise, and that control
# is free and on disk at 0.8964 ledger75, epoch 7 of 12, 909 studies.
#
# BANDS, pre-registered, ledger75.
#   * >= 0.9026  420 clears the 0.0062 paired floor above the shipping corner. Section B's four rows
#                become readable rather than disowned, 448 is the next cell, and the champion family has
#                a resolution lever it has never pulled.
#   * 0.8964-0.9026  directional. Keep nothing, compose nothing.
#   * <  0.8964  resolution is saturated at 336 under labelwise TOO. Section B's caveat is then ANSWERED
#                for this axis rather than quoted, which closes four rows at once — most of what this
#                arm is worth, and the reason it is armed as an instrument.
#
# READ THE FOCAL LABELS, NOT ONLY THE MEAN. If the mean moves and Lateral Meniscus / ACL / MCL do not,
# the delta is not this mechanism and the arm is a lucky reroll of a head. Say so.
#
# HARNESS CHECK. The printed recipe must name "image_size": 420 AND "slice_pooling": "labelwise".
#
# control  labelwise_dense_s2026 = 0.8964 ledger75 @ epoch 7, 909 studies. No control run is owed.
#
# deliberate: --image-size 420 (the arm's thesis)
# deliberate: --batch 6 --accumulate 2 — at batch 12 arm_check prices 420/20sl at ~30 GB, over the 22 GB
#   cap. Accumulation restores the GRADIENT of batch 12, but --rank-weight's pairwise term is pooled
#   per micro-batch and sees 6 studies, not 12. That is a second variable. It is accepted, not matched:
#   a matched 336/batch-6 control costs ~1 h, and is owed only if this arm PASSES (a smaller rank pool
#   is not known to help, but that is an assumption, not a measurement).
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train labelwise420 \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --image-size 420 --batch 6 --accumulate 2 \
  --labels "$LABELS" --out "$MODELS/labelwise420_s2026.pt"
