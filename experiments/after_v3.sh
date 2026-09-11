#!/usr/bin/env bash
# after_v3.sh — the depth ladder yields the card to the label axis.
#
# WHY THE QUEUE WAS REORDERED MID-FLIGHT. `after_depth2.sh` had `labelwise_depth24` queued behind the
# running control. That arm is an INSTRUMENT and its own pre-registration predicts flat-to-slightly
# negative — roughly 80 minutes of 5090 against a ~0-expected-score read. D344 landed while the control
# was at epoch 2 and put a 40%-of-all-positive-labels correction on disk with its fix already written.
# Ratio decides the order, not momentum: the label arm goes first and depth24 keeps its slot behind it.
#
# `after_depth2.sh`'s parent shell (win pid 75276) was killed WITHOUT its process group, so the control
# arm it had already launched keeps running and finishes normally; only the queue behind it was taken
# over. This script picks that queue up.
#
# NO arm_check IN HERE. Two watchers died at the instant `arm_check --smoke` returned — the launching
# process reaped with its children, `uv run` holding the group open long enough for it to matter. Both
# smokes that mattered are already `ok` on disk, and `v3_labels_span` reuses `band_span`'s cached decode
# (images_336px_12sl_0.1-0.9band.npy), so there is nothing new to smoke.
#
# The gate is the CONTROL's log, and the two spellings are different variables: the FILE carries the
# checkpoint's name and the MARKER carries the `run_train` name.
cd "$(dirname "$0")/.." || exit 1

LOG="D:/models/rsna_knee_abnormality/experiments/labelwise_depth20_b10_s2026.full.log"
CONTROL_PID="${1:-}"

echo "=== waiting for labelwise_depth20_b10 at $(date -u +%H:%MZ) ==="
while true; do
  if grep -q "ARM DONE labelwise_depth20_b10" "$LOG" 2>/dev/null; then
    echo "=== control finished at $(date -u +%H:%MZ) ==="
    break
  fi
  if [ -n "$CONTROL_PID" ] && ! ps -W | grep -q "[^0-9]$CONTROL_PID[^0-9]"; then
    echo "=== control process gone at $(date -u +%H:%MZ) - releasing the card anyway ==="
    break
  fi
  sleep 30
done

echo "=== launching v3_labels_span at $(date -u +%H:%MZ) ==="
bash experiments/arms/v3_labels_span.sh

echo "=== V3 DONE at $(date -u +%H:%MZ) - launching labelwise_depth24 ==="
bash experiments/arms/labelwise_depth24.sh

echo "=== QUEUE DONE at $(date -u +%H:%MZ) ==="
