#!/usr/bin/env bash
# after_v2corrected.sh — wait for `v2_corrected`, smoke both depth arms, then run the depth ladder's
# last two rungs in order: the matched control first, the 24-slice corner second. Armed 2026-09-11
# 14:56Z with v2_corrected at epoch 9.
#
# WHY THIS AND NOT ANOTHER LABEL ARM. The label axis has returned three measurements this week
# (confident_span -0.0116, discord_mask -0.0417, v2_corrected pending) and D340 has just moved its
# remaining value into the EVALUATION channel, which is card-free. The foreign-data axis closed on
# D337 (five flat readings plus a licensing block). Depth is the only DATA axis still open, its
# cache is already on disk, and the only depth evidence taken under a head that can SEE depth is
# still steepening where we stopped: 12sl 0.8700 -> 16sl 0.8727 -> 20sl 0.8773.
#
# ORDER IS NOT ARBITRARY. `labelwise_depth24` trims batch 12->10 to fit 24 slices in 31.8 GB, which
# moves --rank-weight's pairwise population as a second knob. `labelwise_depth20_b10` is the arm
# that absorbs that confound, so it runs FIRST: if depth24 is read against labelwise_dense alone,
# any delta prices depth and batch as one thing.
#
# SMOKE BEFORE QUEUE. Neither arm has ever run. 60 studies, ~2 min, and it writes the truncated
# `_first60` caches on a reading axis nothing has decoded. Per the arm header, the checkpoint must
# be dumped and `slices.pool.gate.3.weight` asserted non-zero before any number is written down —
# that assertion is NOT in this script; it is owed on the reading, not on the launch.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN: file `v2_corrected_s2026.full.log`,
# line `ARM DONE v2_corrected`. The gate is an OR with pid liveness — the marker alone hangs forever
# on a silent exit-0 death, the pid alone fires the next arm the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/v2_corrected_s2026.full.log"
SMOKE="$MODELS/smoke_depth_20260911"

prev_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*v2_corrected*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE v2_corrected" "$LOG" 2>/dev/null && break
  prev_alive || break
  sleep 30
done
echo "=== CARD FREE at $(date -u +%H:%MZ) - smoking the depth arms ==="

cd "$(dirname "$0")/.." || exit 1
UV_NO_SYNC=1 uv run python -m kneemri.experiments.arm_check \
  --only labelwise_depth20_b10 --smoke "$SMOKE" --smoke-studies 60 || exit 1
UV_NO_SYNC=1 uv run python -m kneemri.experiments.arm_check \
  --only labelwise_depth24 --smoke "$SMOKE" --smoke-studies 60 || exit 1

echo "=== SMOKE CLEAN at $(date -u +%H:%MZ) - launching labelwise_depth20_b10 ==="
bash "$(dirname "$0")/arms/labelwise_depth20_b10.sh"

echo "=== CONTROL DONE at $(date -u +%H:%MZ) - launching labelwise_depth24 ==="
bash "$(dirname "$0")/arms/labelwise_depth24.sh"
