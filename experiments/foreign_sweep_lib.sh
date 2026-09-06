#!/usr/bin/env bash
# foreign_sweep_lib.sh — serving mattiaangeli's five DINOv3 folds at ONE band, and pricing that band.
#
# Shared by every band arm because the two steps never change: `foreign_folds` scans the 909 holdout
# studies through their lifted kernel at the band it is given, and `foreign_price` reads that scan
# fold-matched (each study by the ONE fold that never trained on it) and emits a holdout record a seat
# ruler can price. Only the band moves, so a second copy of these eleven lines would be a second
# definition of what "serving the foreign family" means.
#
# The environment it needs comes from the arm that sources it: FOREIGN_KERNEL, FOREIGN_WEIGHTS, RAW,
# HOLDOUT, MODELS, REPO.

sweep_band() {
  low="$1"
  high="$2"
  tag="$3"
  served="$MODELS/foreign_dinov3_$tag.npz"
  if [ -e "$served" ] && [ -z "$ALLOW_OVERWRITE" ]; then
    echo "REFUSED sweep_band - '$served' already holds a scan. ALLOW_OVERWRITE=1 to replace it." >&2
    return 1
  fi
  "$REPO/.venv/Scripts/python.exe" -m kneemri.inference.foreign_folds     --kernel "$FOREIGN_KERNEL"     --checkpoints "$FOREIGN_WEIGHTS"     --series-root "$RAW/train_series"     --series "$RAW/train_series.csv"     --studies "$HOLDOUT"     --band "$low" "$high"     --out "$served"
  "$REPO/.venv/Scripts/python.exe" -m kneemri.evaluation.foreign_price     --predictions "$served"     --holdout "$HOLDOUT"     --labels "$RAW/train.csv"     --emit "$MODELS/foreign_dinov3_$tag.holdout.npz"
}
