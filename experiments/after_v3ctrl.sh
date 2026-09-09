#!/usr/bin/env bash
# Wait for latent_register_deep to release the card either way, then run the plain v3small control.
#
# THE GATE IS THE PARENT WATCHER, NOT A MARKER. `latent_register_deep` has two exits and only one of them
# writes an `ARM DONE` line: if the attention gate refuses the mask-0.60 adaptation at epoch 0, the arm dies
# in `run_pretrain` and `run_train` never runs, so `_mark` never fires. Its parent (`after_regdeep.sh`)
# echoes `QUEUE latent_register_deep exited` after the arm returns, under `set -u` with no `-e`, so that line
# lands on BOTH paths. The pid arm covers the third: the parent itself being killed, which writes nothing.
#
# THE pid IS THE WINDOWS pid of after_regdeep.sh, so liveness is `ps -W`, never `kill -0`.
set -u
cd "$(dirname "$0")/.."

PID="${1:?usage: after_v3ctrl.sh <windows pid of after_regdeep.sh>}"
LOG="logs/after_regdeep.log"

echo "waiting for 'QUEUE latent_register_deep exited' in $LOG / pid $PID at $(date -u +%FT%TZ)"
until grep -q "QUEUE latent_register_deep exited" "$LOG" 2>/dev/null || ! ps -W | grep -q "[^0-9]$PID[^0-9]"; do
  sleep 60
done
echo "card released at $(date -u +%FT%TZ); line present: $(grep -qs 'QUEUE latent_register_deep exited' "$LOG" && echo yes || echo no)"

bash experiments/arms/v3small_control.sh
echo "QUEUE v3small_control exited $? $(date -u +%FT%TZ)"
