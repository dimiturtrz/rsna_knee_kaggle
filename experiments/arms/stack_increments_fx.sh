#!/usr/bin/env bash
# stack_increments_fx.sh — the same arm as stack_increments_base, with ONE flag moved: the label table.
#
# WHY THIS ARM EXISTS. The label-correction lane has never been tested on the card. Three tables were
# scored against the 57 radiologist-annotated studies, which is the only ruler whose truth is not
# report-derived:
#
#   blend_three_v1.csv   mean 0.8955   Fracture 0.8147   <- what every arm to date trains on
#   blend_hybrid.csv     mean 0.8978   Fracture 0.8713
#   blend_hybrid_fx.csv  mean 0.9030   Fracture 0.9346   <- this arm
#
# blend_three_v1 is DOMINATED: it degrades Fracture by 0.057 against the plain report label (0.8713),
# self-inflicted on the second-weakest column. blend_hybrid_fx adds one correction on top of
# blend_hybrid and nothing else -- Fracture := monotone_rescale(Fracture + 0.25 * Contusion) -- which
# was pre-registered before measurement, cleared two independent permutation nulls (p = 0.0022 on
# report_labels_v2, p = 0.0010 on blend_hybrid), carries a bootstrap CI excluding zero, and shows a
# coherent placebo pattern: all four helping partners are the acute-trauma set, all seven hurting ones
# degenerative. Verified byte-wise to change the Fracture column and NOTHING else (eleven columns
# +0.0000 on gold).
#
# THE MECHANISM. Report-derived Fracture misses bone-contusion impaction injury. Of the four gold
# studies where the reader said UNK and the radiologist said fracture, all four reports describe a bone
# bruise and none uses a fracture word. Contusion is the report's name for what the radiologist scored
# as a fracture, so it carries the missing signal.
#
# WHAT THIS ARM IS NOT. It is NOT the negation pass. That correction flags 876 cells corpus-wide but
# only 15 of them land in the 57 gold studies, so the gold ruler cannot arbitrate it -- it read -0.0024
# on a denominator of 15 and that is noise, not a refutation. It stays out of this table and out of this
# arm until it has a ruler that can see it.
#
# control  stack_increments_base_s2026, RUNNING on the same card at the same hour. Byte-identical
#          except --labels. This arm therefore needs NO control of its own, which is the whole reason
#          it is built as a twin rather than against band_dense 0.8578.
# cost     12 epochs at 336 px, ~395 s an epoch measured on the twin = ~79 min.
#
# PRE-REGISTERED against whatever stack_increments_base finishes at. Floor 0.0062, board resolution 0.0015.
#   CARRIES   — at or above twin + 0.0062. The gold-ruler gain survives training and the label lane is
#               open: next move is the negation pass with a ruler that can see it, not a slot.
#   PARTIAL   — twin + 0.0015 to twin + 0.0061. Directional, keep the table, do not plan around it.
#   NULL      — within +/- 0.0015. The table's gold advantage does not survive training. That is a real
#               finding: it says the 57-study ruler does not transfer, which prices the ENTIRE label
#               lane, including the blocked rubric re-read, and it retires the lane rather than one file.
#   REGRESSION— below twin - 0.0062. The correction hurts under training despite helping on gold. Then
#               the suspect is the soft-value scale rather than the ordering, since the rescale is
#               order-preserving and cannot change an AUC.
#
# bead: knee-hbml
#
# twin bead: knee-0wta
# deliberate: --labels        (blend_hybrid_fx.csv -- THE ONE MOVED FLAG)
# deliberate: --eval-labels   (report_labels_v2.csv -- the ruler must NOT move with the arm)
# deliberate: --distill-weight
# deliberate: --teacher-targets
#
# control  stack_increments_base_s2026 = 0.8635 (declared 2026-09-03 from this header own prose, verified on disk, knee-sgzgc)
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

BLEND="$PROCESSED/labels/blend_hybrid_fx.csv"
require "$LABELS"
require "$BLEND"
require "$PROCESSED/raptor_teacher.npz"

run_train stack_increments_fx \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --distill-weight 0.3 --teacher-targets "$PROCESSED/raptor_teacher.npz" \
  --labels "$BLEND" --eval-labels "$LABELS" --out "$MODELS/stack_increments_fx_s2026.pt"
