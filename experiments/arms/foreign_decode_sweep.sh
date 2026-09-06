#!/usr/bin/env bash
# foreign_decode_sweep.sh — is the foreign family's SERVE-TIME slice band tuned for our geometry?
#
# WHAT IS ALREADY KNOWN, AND WHY THAT OPENS THIS. Their released kernel serves 0.12-0.88 while their
# training code uses 0.2-0.8, on ONE set of weights — so their own serve-time band is already a choice
# made after training, not a property of the checkpoints. On our data that choice is worth +0.032 over
# their training band (`foreign_dinov3_matched` 0.8812 against `foreign_dinov3_band208`). A dial that
# moves the family by 3 points and was tuned on THEIR corpus has no reason to sit at its optimum on
# ours, and the seat that entry 56060097 just shipped reads through it.
#
# WHAT IS SWEPT, AND WHAT IS NOT. Band only, two NEW points either side of theirs: 0.08-0.92 (wider) and
# 0.16-0.84 (narrower). The middle point of the grid is `foreign_dinov3_matched`, which is on disk — the
# card is not spent re-serving a band already served. The slice-count axis {16, 24} is NOT swept here:
# `N_SLICE` is a constructor argument of their `Net` read from the checkpoint cfg, so a served count that
# is not 16 builds a model the state dict does not fit. That is an assertion about their code, and it is
# the reason this arm is two runs and not six.
#
# HOW IT IS READ, written before the number. Fold-matched pooled AUC per band (each study by the ONE fold
# that never trained on it), then the emitted record priced as a seat at 6/13 on the fixed base the
# matched band was priced on. Pre-registered by mastermind at 16:33Z:
#   * >= +0.005 seat delta over the matched band — the family's decode is OURS to tune, ForeignEntry's
#     SLICE_BAND changes for the next entry, and the 23:30Z slot has a mechanism-backed candidate.
#   * < +0.003 — their serving corner transfers to our corpus and the decode axis closes for this family.
#     That closure is worth the ten minutes: it also says the +0.032 was about their TRAINING band being
#     wrong, not about band tuning being available.
#
# COST. ~10-15 min of card, 909 studies x five folds x two bands. ZERO submission slots.
#
# deliberate: --band (the only moving variable; everything else is their lifted program untouched)
# deliberate: two points not six (the slice axis is refused by their own constructor, see above)
# reading: foreign_dinov3_band0892.holdout.npz
# reading: foreign_dinov3_band1684.holdout.npz
# control  foreign_dinov3_matched = 0.8812 fold-matched pooled, +0.0178 at 6x on the served base of 8.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

FOREIGN_KERNEL="$REPO/external/dinov3folds/bend_the_knee_to_dinov3_ensembled.py"
FOREIGN_WEIGHTS="D:/data/external/kaggle_foreign/mattiaangeli_dinov3"
RAW="$(dirname "$PROCESSED")/../raw/rsna_knee_abnormality"
HOLDOUT="$MODELS/band_dense_s2026.holdout.npz"

require "$LABELS"
require "$HOLDOUT"

sweep_band() {
  low="$1"
  high="$2"
  tag="$3"
  served="$MODELS/foreign_dinov3_$tag.npz"
  if [ -e "$served" ] && [ -z "$ALLOW_OVERWRITE" ]; then
    echo "REFUSED foreign_decode_sweep - '$served' already holds a scan. ALLOW_OVERWRITE=1 to replace it." >&2
    return 1
  fi
  "$REPO/.venv/Scripts/python.exe" -m kneemri.inference.foreign_folds \
    --kernel "$FOREIGN_KERNEL" \
    --checkpoints "$FOREIGN_WEIGHTS" \
    --series-root "$RAW/train_series" \
    --series "$RAW/train_series.csv" \
    --studies "$HOLDOUT" \
    --band "$low" "$high" \
    --out "$served"
  "$REPO/.venv/Scripts/python.exe" -m kneemri.evaluation.foreign_price \
    --predictions "$served" \
    --holdout "$HOLDOUT" \
    --labels "$RAW/train.csv" \
    --emit "$MODELS/foreign_dinov3_$tag.holdout.npz"
}

sweep_band 0.08 0.92 band0892
sweep_band 0.16 0.84 band1684
echo "ARM DONE foreign_decode_sweep $(date -u +%FT%TZ)"
