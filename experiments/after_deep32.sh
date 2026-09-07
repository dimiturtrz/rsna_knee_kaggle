#!/usr/bin/env bash
# Wait for the `weakdraw` chain to release the card, then build the 32-slice cache and run `deep32_batch6`.
#
# WHY `weakdraw_frx` AND NOT `weakdraw_flat`. `arms/weakdraw.sh` makes THREE `run_train` calls -- the
# spread, then `weakdraw_flat`, then `weakdraw_frx`. Gating on the flat one would preempt its own
# chain's last arm and put two of ours on the card, which is the contention `card_check` exists to
# refuse. The gate is the LAST arm of the chain.
#
# WHY THE SMOKE IS IN HERE. `deep32_batch6` is UNCACHED: no 336px/32-slice truncated cache exists, and
# `arm_check --smoke` is what writes one (60 studies, under a minute). Running it here rather than by
# hand means the card is never idle waiting for a human to notice the gate cleared.
#
# WHY THIS PATH AND NOT `logs/`. A `run_train` arm re-runs from a snapshot whose `tee` writes
# `$MODELS/experiments/<checkpoint>.full.log`; `logs/` holds the `foreign_run` arms, a different
# launcher. The FILE carries the checkpoint spelling and the MARKER TEXT carries the `run_train` name --
# two different variables, and this greps the marker as an existing `ARM DONE` line spells it.
set -u

cd /d/personal_projects/rsna_knee_kaggle || exit 1

MODELS="D:/models/rsna_knee_abnormality"
GATE="$MODELS/experiments/weakdraw_frx_s2026.full.log"
LOG="logs/after_deep32.log"
SMOKE="$MODELS/smoke_deep32"

echo "waiting for 'ARM DONE weakdraw_frx' in $GATE" >> "$LOG"
until grep -q "ARM DONE weakdraw_frx" "$GATE" 2>/dev/null; do sleep 30; done
echo "gate cleared at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

UV_NO_SYNC=1 .venv/Scripts/python.exe -m kneemri.experiments.arm_check \
  --only deep32_batch6 --smoke "$SMOKE" --smoke-studies 60 >> "$LOG" 2>&1
echo "smoke returned $? at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

bash experiments/arms/deep32_batch6.sh >> "$LOG" 2>&1
echo "deep32 watcher finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
