#!/usr/bin/env bash
# Occlusion diagnostics on a trained member: what is the model actually reading?
#
# Not a training arm - it spends no gradient and produces three measured deltas rather than a picture.
# See kneemri/attribution.py for why saliency is deliberately absent.
#
# 2026-08-11, on base + ranking (0.8283):
#   slots   AX_FLUID_FS +0.0193, the other five +0.0042..+0.0094 (each inside the 0.006 noise floor)
#   region  whole 0.8448 | centre-only 0.8021 | border-only 0.8006
#   depth   best window at slice 7 = 0.8332, range 0.7976..0.8332 over ten windows
source "$(dirname "$0")/../lib.sh"
CHECKPOINT="${1:-$PROCESSED/rankbase24_s2026.pt}"
require "$LABELS" "$CHECKPOINT"

uv run python -u -m kneemri.attribution \
  --checkpoint "$CHECKPOINT" --labels "$LABELS" --studies 400 2>&1 | tee "$LOGS/attribution.full.log"
echo "ARM DONE attribution"
