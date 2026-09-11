#!/usr/bin/env bash
# chainfour_v5blend.sh -- chainfour with ONE variable moved: the TRAINING label table.
#
# bead: knee-t9fzu
# reading: chainfour_v5blend_s2026.holdout.npz
#
# WHY THIS CELL AND NOT ANOTHER STACK. D381 measured what this project had never separated: SOLO value
# and SEAT value are different quantities. Per-attribute half-split seat deltas against the shipped
# honest six, 40 readings, LogitAverage, macro AUC midpoint @holdout909:
#
#     merged-rijeka  +0.0005 +/- 0.0005   36/40      (seated 0x, licence-blocked)
#     labelwise      +0.0005 +/- 0.0005   31/40      (seated 1x)
#     calib-tgt      +0.0002 +/- 0.0006   29/40
#     blend-three    +0.0001 +/- 0.0005   18/40
#     hold-annot     -0.0000 +/- 0.0007   17/40
#     ema            -0.0003 +/- 0.0005    8/40
#     distill        -0.0007 +/- 0.0004    2/40      (seated 4x -- the WORST seat)
#
# Seat delta tracks REPRESENTATION in the six, not solo marginal: distill has the second-largest solo
# marginal (+0.0125) and is the worst possible addition, because the six already carry it four times
# over. Every attribute the table INDEXES is now priced at or below half a noise tick. So no arm that
# stacks indexed attributes can move these six seats, and the only licensed axes are the ones the table
# does not index: teacher identity, label TABLE, fold geometry.
#
# THE ONE VARIABLE. Against `chainfour` (0.8648, second on the honest pool, clean, k=5, NO distill):
# --labels moves from blend_three_v1.csv to report_labels_v5blend.csv. Everything else is held
# byte-for-byte -- 12 epochs, labelwise pooling, whole-stack, lr 4e-5, 20 slices, band 0.1/0.9, image
# 336, hold annotations, ema, calibrated targets, and report_labels_v2.csv still the EVAL table so the
# number is comparable to every other member on disk.
#
# WHY v5blend, AND WHY THIS IS NOT THE FOURTH GOLD-RULER REPEAT. On the blind hand-adjudicated gold
# ruler -- 469 cells, key sealed until the last verdict was written, the only ruler in this project not
# fitted on the same labels the members were trained under -- the tables order
#
#     discord_mask 0.6982 > v5blend 0.6813 > v6tears 0.6734 > v2_corrected 0.6272 > band_span 0.6008
#
# and v5blend is the best table that is not itself a masking scheme. That ruler has now named a
# preference the board declined THREE times (v2 seating, corrected-T1 seating -> cleanseat_v1 0.942,
# blind gold seating -> goldseat_v1 0.944 dead flat). All three spent the ruler on SEATS inside a
# saturated seating, where D370 already puts fixed-count reseating at 0 for 57. Not one of them ever
# spent it on SUPERVISION. A label table is what the member LEARNS FROM, not which pre-trained function
# gets a vote, so this is a different mechanism and a genuinely untried form of the hypothesis -- and
# it is the form that can still be wrong in an informative direction.
#
# EINSTEIN GATE. labelwise + blend_three + no-distill is TAKEN: chainfour 0.8648 and chaindense 0.8628.
# labelwise + report_labels_v2 is taken by the labelwise_* family. No arm in experiments/arms/ passes
# --labels report_labels_v5blend.csv, and no member on disk trains on it. The cell is empty.
#
# HARNESS CHECK BEFORE READING THE NUMBER. The printed recipe must name report_labels_v5blend.csv on
# "labels" and report_labels_v2.csv on "eval_labels" -- if both read v2 the variable never landed and
# this is a re-run of labelwise224 at 336. It must also show "slice_pooling": "labelwise",
# calibrated_targets true, hold_annotations true, ema true, and NO teacher file. Three runs have entered
# this ledger as facts about an architecture when they were facts about a flag that never landed.
#
# PRE-REGISTERED @holdout909 midpoint, against chainfour's 0.8648 and the honest ceiling 0.8649. Note
# the eval table is v2 in both arms, so a table that disagrees with v2 is scored by a ruler that
# disagrees with it -- a NEGATIVE solo here does not refute the table, it prices the disagreement.
#   >= 0.8665   the supervision table is a real solo lever and the gold ruler transfers when it picks
#               what a member learns rather than which member votes. That re-opens the whole label-table
#               axis as a MEMBER axis and is the largest thing available on this card.
#   0.8630-0.8664  flat against chainfour. The table is a lateral move at solo level and the member is
#               judged on its ensemble marginal into the six ALONE -- which is the point: it is seated
#               zero times on the one axis D381 says still pays.
#   0.8560-0.8629  below the clean twin. Either the table is worse supervision, or v2-as-eval is
#               penalising the disagreement. Separate the two by re-scoring this member under v5blend
#               as the eval table before drawing any conclusion -- that costs no card.
#   <= 0.8559   more than a whole solo band down. The table is bad supervision independent of the eval
#               ruler, and the gold ordering does not survive contact with training at all. Names the
#               fourth non-transfer and closes the gold ruler as a design instrument for good.
#
# PRICE: ~1.7 h of card (336px, 12 epochs, chainfour's own footprint), ZERO submission slots. Label
# table already on disk. Expected solo 0.860-0.868 -- this is not bought as a solo improvement, it is
# bought as the only unseated axis left after D381.
#
# deliberate: --labels (report_labels_v5blend.csv, the single moving variable vs chainfour)
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

V5BLEND="$PROCESSED/labels/report_labels_v5blend.csv"
require "$V5BLEND"
require "$LABELS"

run_train chainfour_v5blend \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --image-size 336 --hold-annotations --ema --calibrated-targets \
  --labels "$V5BLEND" --eval-labels "$LABELS" --out "$MODELS/chainfour_v5blend_s2026.pt"
