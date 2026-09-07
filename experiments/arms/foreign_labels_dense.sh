#!/usr/bin/env bash
# foreign_labels_dense.sh — our best solo recipe, supervised by HIS report table instead of ours.
#
# bead: knee-npmz7
# reading: foreign_labels_dense_s2026.holdout.npz
#
# THE QUESTION, AND WHY IT IS NOT THE ONE THE 2x2 ASKS. Cells A'/C price HIS recipe on two corpora. This
# prices HIS LABELS under OUR recipe, which is the only cell that can say whether the report reader is
# what carries his lead — the 2x2 cannot, because every one of its cells reads his labels and his trunk
# together.
#
# WHAT THE RESEARCHER MEASURED, AND WHY IT POINTS HERE (D8, 2026-09-07). The two tables ORDER reports
# almost identically — Spearman >= 0.89 on 8 of 12 findings — and they differ at ONE CUT. Our 0.68 level
# is 38% gold-positive pooled (MCL 0 of 6, Fracture 1 of 4); his table calls that same level negative.
# So ours over-calls exactly where the annotations say nothing is there, and his 909 lead GROWS from
# 0.030 to 0.043 when both are read under the ruler the gold prefers. A trunk taught our cut is taught
# to over-call the columns where the gap lives, and that is a supervision defect, not a capacity one.
#
# ONE VARIABLE. Byte-identical to `labelwise_dense` — same epochs, pooling, backbone rate, slices, band,
# calibrated targets, held annotations and seed — except the table `--labels` names. Its control is
# therefore already on the ledger and no control run is queued.
#
# AND THE RULER DOES NOT MOVE WITH IT. `--eval-labels` pins the holdout to OUR table while training reads
# his. Without that pin the arm would change what it is taught AND what it is graded by in one step, and
# a table that merely agrees with itself would read as a win — the shape of every self-scoring result
# this project has had to retract. The two rulers are then applied to the SAME predictions afterwards,
# which is a second reading of one run and costs no card.
#
# `--hold-annotations` IS IMPLICIT HERE and kept anyway. His table carries 0 of the 58 gold studies, so
# there is nothing for the flag to hold out of training; it stays because it is what leaves the holdout
# record beside the checkpoint, and dropping it would make this the one arm in the family that cannot be
# priced as a seat.
#
# PRE-REGISTERED (researcher's bands, adopted), read on the >= 0.75 ruler against labelwise_dense:
#   >= +0.015  the labels carry the gap. The follow-up is a table swap across the calibrated seats, not
#              a seventh member — and his reader becomes an asset to replicate rather than a rival.
#   +0.005 to +0.015  directional and the right size; NOT a finding on one arm. The second point is the
#              same cell at the >= 0.5 cut, which separates "his table is better" from "the cut is".
#   < +0.005   labels RETIRE as an axis and the lead points at the recipe — dense windows, CoAtNet — which
#              is where cells A'/C are already looking. That is a real answer, not a null.
#   <= -0.015  his table is worse under our recipe, which given the gold agreement above would say the
#              two tables are not interchangeable at all and the cut is doing something our trunk needs.
#              Suspect the join first: 4,349 rows against our corpus, and a study his table lacks lands
#              at 0.5, which is our own PRESENT default and binarises to absent.
#
# HARNESS CHECK, BEFORE THE VERDICT. The printed recipe must name his path under "labels" and ours under
# "eval_labels". They are two flags one letter apart in the arm and one word apart in the log, and an arm
# that silently trained and scored on the same table would print a plausible number for the wrong cell.
#
# PRICE: ~2.5 h, no submission slot. Queued after cell C, before resnet_224 and jitter-off.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

HIS_LABELS="D:/data/external/raptor/labels/labels_llm_soft.csv"
require "$LABELS"
require "$HIS_LABELS"

run_train foreign_labels_dense \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --labels "$HIS_LABELS" --eval-labels "$LABELS" --out "$MODELS/foreign_labels_dense_s2026.pt"
