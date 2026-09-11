#!/usr/bin/env bash
# after_depth.sh — relaunch of the depth ladder after `after_v2corrected.sh` died between its two
# smokes at 15:01Z on 2026-09-11, leaving the card idle for nineteen minutes with `ok
# labelwise_depth20_b10` as the last line it ever wrote. The v2_corrected wait and the depth20_b10
# smoke are BOTH already done and are not repeated here: this script picks up at the depth24 smoke.
#
# WHY THE CONTROL STILL RUNS FIRST. `labelwise_depth24` trims batch 12->10 to fit 24 slices in
# 31.8 GB, and `RECIPE` carries `--rank-weight 1.0`, whose pairwise population IS the batch
# (head_trial.py:231). So batch is a second knob moving with depth. `labelwise_depth20_b10` is the
# matched control that absorbs it, and reading depth24 against `labelwise_dense` alone would price
# depth and batch as one thing.
#
# The reading is owed on the checkpoint, not on the launch: dump it and assert
# `slices.pool.gate.3.weight` is non-zero before any depth number is written down.
source "$(dirname "$0")/lib.sh"

SMOKE="$MODELS/smoke_depth_20260911"

cd "$(dirname "$0")/.." || exit 1
UV_NO_SYNC=1 uv run python -m kneemri.experiments.arm_check \
  --only labelwise_depth24 --smoke "$SMOKE" --smoke-studies 60 || exit 1

echo "=== SMOKE CLEAN at $(date -u +%H:%MZ) - launching labelwise_depth20_b10 ==="
bash "$(dirname "$0")/arms/labelwise_depth20_b10.sh"

echo "=== CONTROL DONE at $(date -u +%H:%MZ) - launching labelwise_depth24 ==="
bash "$(dirname "$0")/arms/labelwise_depth24.sh"
