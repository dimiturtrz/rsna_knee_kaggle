#!/usr/bin/env bash
# after_depth2.sh — third launch of the depth ladder. TWO watchers have now died at the SAME point:
# the instant `arm_check --smoke` returned. `after_v2corrected.sh` wrote `ok labelwise_depth20_b10`
# and nothing after it; `after_depth.sh` wrote `ok labelwise_depth24` and nothing after it. Neither
# printed an error, neither hit its `|| exit 1` (the smoke passed in both cases), and in both the
# next line owed was a plain `echo`. That is not a script bug — it is the launching process being
# reaped with its children, and `uv run` inside arm_check is what holds the group open long enough
# for it to matter.
#
# So this script contains NO arm_check. Both smokes are already `ok` on disk and re-proving them is
# what killed the last two launches. It runs the two arms and nothing else.
#
# Order unchanged and it is not arbitrary: `labelwise_depth24` trims batch 12->10 to fit 24 slices
# in 31.8 GB, and `RECIPE` carries `--rank-weight 1.0`, whose pairwise population IS the batch
# (head_trial.py:231). `labelwise_depth20_b10` is the matched control that absorbs that second knob.
#
# The reading is owed on the checkpoint, not the launch: dump it and assert
# `slices.pool.gate.3.weight` is non-zero before any depth number is written down.
cd "$(dirname "$0")/.." || exit 1

echo "=== launching labelwise_depth20_b10 (control) at $(date -u +%H:%MZ) ==="
bash experiments/arms/labelwise_depth20_b10.sh

echo "=== CONTROL DONE at $(date -u +%H:%MZ) - launching labelwise_depth24 ==="
bash experiments/arms/labelwise_depth24.sh

echo "=== LADDER DONE at $(date -u +%H:%MZ) ==="
