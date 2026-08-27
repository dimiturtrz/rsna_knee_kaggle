#!/usr/bin/env bash
# labelwise_dense.sh — per-diagnosis slice pooling at the corner that has twenty slices to pool.
#
# bead: knee-377t
#
# WHY THIS RE-OPENS AN AXIS THE LEDGER CLOSED. `labelwise224` read +0.0015 over its control and the
# ledger wrote the verdict "58 image-read labels cannot supervise twelve separate attention channels".
# That sentence is measuring the wrong denominator. The 58 radiologist-annotated studies are the EVAL
# set; the training supervision for every arm in this family is the report label table over the whole
# corpus, twelve findings across ~4,349 studies. Twelve attention channels are supervised by ~52,000
# label cells, not 58. The stated reason the axis closed is not a fact about this arm.
#
# AND THE ARM RAN WHERE ITS MECHANISM CANNOT FIRE. `labelwise224` pooled TWELVE slices. A per-diagnosis
# query buys the right to send different findings to different depths, and at twelve slices inside a
# 0.2-0.8 band there is almost nothing to send them to. The axis was tested at its own worst corner.
#
# THE EVIDENCE THAT SAYS SO IS NEW, AND IT IS FOREIGN. `kneemri.evaluation.foreign_gate` scored Raptor's
# `coatnet_v4_full` — whose head is a per-finding attention pool, the same mechanism one axis up —
# through our caches on the 58 gold studies:
#
#     336px_12sl  0.8700     336px_16sl  0.8727  (+0.0027)     336px_20sl  0.8773  (+0.0046)
#
# Their per-finding head gains MORE from 16->20 than from 12->16: the curve is still steepening at the
# depth we stopped at. Against that, our own slice-count sweep read -0.0001 for 20 slices over 12
# (`knee-ezz`) — but that sweep was run under MAX pooling, which commits all twelve findings to one
# window and throws the extra slices away. The two axes were each closed in the absence of the other:
#
#     slice count, tested under slot-max        -> -0.0001   (the head cannot spend the slices)
#     per-diagnosis pooling, tested at 12 slices -> +0.0015   (the pooling has no slices to spend)
#
# Neither has ever been run WITH the other, and the interaction is exactly what the foreign curve is a
# picture of. This arm is that cell.
#
# THE OTHER REASON THE CELL IS EMPTY. `attend_dense` looks like it already tested pooling at this corner
# and read -0.0009 — but knee-7gyw found the aggregator sat in no parameter group, so its gate was still
# exactly zero after a full epoch and the arm compared the arithmetic MEAN against the max. The ledger
# records the consequence: "the pooling-operator cell REOPENS". Working attention of any kind, shared or
# per-diagnosis, has never been read at 336 px x 20 slices.
#
# ONE VARIABLE. Byte-identical to `calib_dense_hold` (ledger 0.8634, the best solo member this project
# owns) except `--slice-pooling max` -> `labelwise`. Same recipe, epochs, backbone rate, slices, band,
# calibrated targets, held annotations, label table and seed. The control is already measured, so no
# control run is queued and nothing is spent re-establishing it.
#
# WHY THE HELD FAMILY. `--hold-annotations` leaves a holdout record beside the checkpoint, so a win is
# priceable as a SEAT by `seat_value` without a second fit, and a loss is still a member with a record.
# The calibrated corner is also the one `calibsix` actually ships, so a win lands in the roster we serve
# rather than in a family we would then have to port it to.
#
# PRICE: ~2.5 h on a card that is otherwise idle after contrast_dense frees. No submission slot — the
# reading is offline against a control already on the ledger.
#
# PRE-REGISTERED BANDS, against calib_dense_hold 0.8634 and the 0.0062 paired floor:
#   >= 0.8696  the interaction is real; per-diagnosis pooling pays only when there is depth to spend it
#              on, and the follow-up is a uniform re-fit of the calibrated seats, not a seventh seat.
#   0.8666-0.8695  directional and the right size to be the interaction; NOT a finding on one arm. The
#              honest next step is the 16-slice cell, which separates depth-scaling from a one-off.
#   0.8573-0.8665  FLAT. The mechanism does not scale with depth either, and the axis closes for a
#              second and better reason than the one in the ledger. Still a diversity seat at +0.0015's
#              profile — different internal structure, same quality — which is what `labelwise224` earned.
#   <= 0.8572  worse than the max at this corner. Suspect the pooled tensor's diagnosis axis colliding
#              with `SlotHead`'s own per-diagnosis query BEFORE suspecting the mechanism: both now
#              replace the same axis and `slot_head.py:120` averages over it. Check that first.
#
# HARNESS CHECK, BEFORE THE VERDICT AND NOT AFTER. The printed recipe must name "slice_pooling":
# "labelwise" and "slices": 20. The gate is zero-initialised on purpose, so at step zero this arm IS the
# arithmetic mean and a dead gate is INVISIBLE in the score — which is the precise way `attend_dense`
# lost a year of reading. Dump the checkpoint and assert `slices.pool.gate.3.weight` is non-zero before
# any number here is written down.
#
# VRAM: 20 slices reserve 28.26 GB by DeviceBudget's own derivation, and labelwise replaces the slice
# axis with a 12-wide diagnosis axis AFTER the backbone, so the growth lands on the pooled tensor and not
# on the activations. It has never been run at this size; smoke it before it is queued.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train labelwise_dense \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --labels "$LABELS" --out "$PROCESSED/labelwise_dense_s2026.pt"
