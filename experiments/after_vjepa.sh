#!/usr/bin/env bash
# Wait for vjepa_cold to release the card, smoke unfreeze_full, and run it only if the smoke passes.
#
# THE GATE IS MARKER **OR** pid, never one alone. The marker is what a completed `run_train` writes; the
# pid is what a crashed one leaves behind. A marker-only gate holds the card idle forever after a death,
# and a pid-only gate fires the next arm into a card the dying process has not released.
#
# THE LOG FILE IS NAMED FOR THE CHECKPOINT AND THE MARKER TEXT IS NOT. `_arm_log` takes the basename of
# `--out`, so the file is `vjepa_cold_s2026.full.log`; `_mark` writes the `run_train` name, so the line
# is `ARM DONE vjepa_cold`. The seed is in one and never in the other.
#
# THE pid IS THE WINDOWS pid, read from `card_check`, so liveness is tested with `ps -W`, never `kill -0`.
#
# THE SMOKE IS A GATE, NOT A FORMALITY. Twelve unfrozen blocks hold roughly twice the backward state of
# six, and the arm inherits a batch sized for six. Sixty studies answer the VRAM question in under a
# minute; an OOM twenty minutes into the real run answers it for the price of the card.
set -u
cd "$(dirname "$0")/.."

PID="${1:?usage: after_vjepa.sh <windows pid of vjepa_cold>}"
LOG="D:/models/rsna_knee_abnormality/experiments/vjepa_cold_s2026.full.log"
SMOKE="C:/Users/User/AppData/Local/Temp/claude/D--personal-projects-rsna-knee-kaggle/ce6d6a84-6c92-422b-be9b-be73c2627e72/scratchpad/smoke_unfreeze_full"

echo "waiting on vjepa_cold pid $PID / $LOG at $(date -u +%FT%TZ)"
until grep -q "ARM DONE vjepa_cold" "$LOG" 2>/dev/null || ! ps -W | grep -q "[^0-9]$PID[^0-9]"; do sleep 60; done
echo "card released at $(date -u +%FT%TZ)"

rm -rf "$SMOKE"
UV_NO_SYNC=1 .venv/Scripts/python.exe -m kneemri.experiments.arm_check \
  --only unfreeze_full --smoke "$SMOKE" --smoke-studies 60
SMOKED=$?
echo "SMOKE unfreeze_full exited $SMOKED $(date -u +%FT%TZ)"
if [ "$SMOKED" -ne 0 ]; then
  echo "SMOKE FAILED - arm NOT launched, card left free $(date -u +%FT%TZ)"
  exit "$SMOKED"
fi

bash experiments/arms/unfreeze_full.sh
echo "QUEUE unfreeze_full exited $? $(date -u +%FT%TZ)"
