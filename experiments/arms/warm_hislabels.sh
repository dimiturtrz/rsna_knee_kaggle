#!/usr/bin/env bash
# warm_hislabels — the three axes that buy CORR without losing solo, stacked on one arm (D45 §3).
#
# WHY CORR AND NOT SOLO. D43 swapped five champion parents for distilled children of their own family
# across all 31 subsets: children carrying +0.0052 of solo read −0.0014 on the block, and undistilled
# siblings at +0.0036 solo read −0.0002. Same-family solo lift reaches the block at ~0.08 of itself, not
# at the seat's w. So an arm is now written to move a member's ERROR PATTERN, and the gate is a swap, not
# a solo delta. D45's corr map says exactly three axes on this trunk lower corr to the five-parent block
# to ~0.90 while holding solo at 0.889-0.896, where every sibling sits at 0.93:
#   warm restart      `warm_calib_dense` corr 0.900 / solo 0.8917 (fresh LoRA on an already-adapted trunk)
#   his labels        `hisfluid_hold`    corr 0.899 / solo 0.8890 (ONE of his columns moved corr by 0.03)
#   labelwise pooling `labelwise_dense`  corr 0.911 / solo 0.8959 (swap +0.0013 — the only member on disk
#                                                                  that clears the D43 gate today)
# Nothing on disk carries two of them together except `foreign_labels_dense` (his labels + labelwise),
# which is on the card as this is written. This arm carries all three.
#
# FLAG FOR FLAG, and where each flag comes from — no flag here is new to the project:
#   `--encoder-weights .../calib_dense_hold.enc.pt` + `--backbone-lr 1e-5`  = warm_calib_lowlr.sh:77-81
#   `--slice-pooling labelwise`                                            = labelwise_dense.sh:77-79
#   `--labels $HIS_LABELS --eval-labels $LABELS`                           = foreign_labels_dense.sh:58-61
#   12 epochs, 20 slices, band 0.1-0.9, --calibrated-targets, --hold-annotations, --whole-stack, $THROUGHPUT
#                                                                          = the stackfit line every parent shares
# `--encoder-weights` loads the MERGED BACKBONE ONLY (`kneemri/training/fitted_encoder.py:15-19`; head and
# pooling are rebuilt from the flags), so a warm trunk under a DIFFERENT pooling than the one it was fitted
# with is legal — `warm_calib_*` already did it under max, and this is the same load under labelwise.
#
# --eval-labels IS PINNED TO v2 AND THAT IS WHAT MAKES THE ARM READABLE. The targets are his 14-level
# soft table and the ruler stays ours; `recal_holdannot` is the incident where the ruler moved with the
# arm and a +0.0477 turned out to be two rulers. Assert from metrics row 0 that `eval_labels` names v2
# and `labels` names his csv — two flags one letter apart in the arm and one word apart in the log.
#
# --calibrated-targets IS KEPT, AND IS PROBABLY A NO-OP HERE. `LevelCalibration` fits one empirical gold
# rate per unique reader value over the GOLD rows (`evaluation/level_calibration.py:75-110`), and his table
# carries none of the 58 gold studies, so the fit sees no cells and every level is left untouched
# (:139-152). It is kept anyway because `foreign_labels_dense` carries it (`.sh:60`): dropping it here
# would make this arm differ from that one by THREE things instead of the two axes it is meant to add.
#
# PRICE. labelwise 406 s/epoch, warm_calib_lowlr 365-403 → ~405 s x 12 ≈ 81 min + the holdout pass ≈ 1.4 h,
# 12 GB class. Expected (D45): solo 0.887-0.892 on the v2 ruler, corr to block 0.86-0.88 (three axes at
# 0.90 each, sub-additive). Priced at solo 0.889 / corr 0.87: sixth seat (b) = −0.0009 + 0.076·0.0025
# + 0.0139·0.13 = +0.0011 offline; honest 1/3 = +0.0034 offline ≈ +0.002 board; (c) replaces raptor
# = −0.0210, never. This is a SIXTH-SEAT arm and it is not a slot on its own — under D45 §3's fits no
# DINOv2-small member can be a 1/3 seat at all (at corr 0.90 that needs solo ≥ 0.9057, 0.010 above the
# best member the project owns).
#
# PRE-REGISTERED, on the D43 gate rather than on a solo:
#   `d42_block.py --only warm_hislabels_s2026:calib_span_s2026`
#   swap ≥ +0.0010 AND corr(child, block−parent) ≤ corr(parent, block−parent) + 0.01
#                       → SEAT IT sixth, and it becomes the parent a second stacked axis is measured from.
#   +0.0005..+0.0010    → directional, no seat, ledger line only.
#   < +0.0005           → the three axes DO NOT STACK: each buys ~0.03 of corr alone and the sum buys
#                         nothing, which closes this trunk for corr as D44 closed it for capacity.
#
# ARM ONLY IF `foreign_labels_dense` READS corr-to-block ≤ 0.89 AT solo ≥ 0.885 (mastermind 17:05Z). If
# twelve of his columns do not decorrelate our trunk, the warm restart alone reads +0.0009 as a sixth seat
# (measured) and there is nothing left to stack.
#
# deliberate: --encoder-weights (warm restart — fresh LoRA on an adapted trunk, the capacity argument)
# deliberate: --backbone-lr 1e-5 (its own basin; 4e-5 from a warm start is warm_calib_dense, already read)
# deliberate: --slice-pooling labelwise (the only axis on disk whose swap clears the gate)
# deliberate: --labels $HIS_LABELS (his 14-level soft table as targets; the ruler stays v2)
# deliberate: --eval-labels (the ruler must not move with the arm; recal_holdannot is the incident)
# reading: warm_hislabels_s2026.holdout.npz
# bead: knee-rn3yp
# control  warm_calib_lowlr_s2026 = 0.8908 corr 0.898 and labelwise_dense_s2026 = 0.8959 corr 0.911 (896, L75mid)
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

HIS_LABELS="D:/data/volumetric/mri/processed/raptor/labels/labels_llm_soft.csv"
ENCODER="$MODELS/encoders/calib_dense_hold.enc.pt"
require "$LABELS"
require "$HIS_LABELS"
require "$ENCODER"

run_train warm_hislabels \
  $RECIPE --epochs 12 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 1e-5 --slices 20 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --encoder-weights "$ENCODER" \
  --labels "$HIS_LABELS" --eval-labels "$LABELS" --out "$MODELS/warm_hislabels_s2026.pt"
