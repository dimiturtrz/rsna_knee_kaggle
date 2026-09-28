#!/usr/bin/env bash
# bead: knee-bpuei
# reading: elr_hold_*_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   The gate for Liu et al.'s early-learning regulariser (ELR, NeurIPS 2020; code 428aa973) on the
#   shipping labelwise line: a held control and ONE ELR arm, identical but for --early-weight.
#
# WHY TWO FITS AND NOT ONE AGAINST labelwise_ext_drop40_long
#   ELR aims at noisy report-extracted targets, so its deciding ruler is gold-58, the radiologist cells.
#   Every existing member trains on those 58 studies (--hold-annotations still fits the rows), so gold-58
#   on them is in-sample (knee-x4muo). --hold-annotation-studies withholds the studies, and no arm has
#   ever used it — so the parent has no held twin and the control has to be fitted here.
#
# PRE-REGISTERED, read on BOTH rulers (`annotated_readings` gold-58, and ledger75):
#   ledger75 >= +0.004 over the held control AND gold-58 not below it  -> ELR transfers; rebuild the
#                                        soup parent on the full-data line (knee-frtry path), then SCE.
#   inside +-0.004 on ledger75           -> null for this recipe; SCE inherits it (same class, weaker).
#   gold-58 up, ledger75 flat            -> ELR cleans labels the board may score; one slot decides.
#
# deliberate: every flag but --hold-annotation-studies and --early-weight is labelwise_ext_drop40_long's
#             line byte for byte, so the pair is single-variable and the parent prices the hold cost.
# deliberate: --early-weight 3.0 and the default --early-beta 0.7 are Liu's published values, not swept.
# deliberate: no --vram-gb, as on the parent line: DeviceBudget derives the cap from --slices, and 22
#             overstates it by 12.44 GB (arm_check, knee-ylp0). Still UNSHAREABLE (knee-l2d7o): owner call.
#
# CARD. 2 x 24 epochs at ~280 s = ~3.7 h. Owner-GPU-gated; nothing else of ours may share the card.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

for WEIGHT in 0 3.0; do
  NAME="elr_hold_w${WEIGHT%.*}"
  run_train "$NAME" \
    $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
    --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --scheme extended \
    --calibrated-targets --hold-annotations --hold-annotation-studies \
    --batch 4 --accumulate 3 --slot-dropout 0.40 --early-weight "$WEIGHT" \
    --labels "$LABELS" --out "$MODELS/${NAME}_s2026.pt"
done
