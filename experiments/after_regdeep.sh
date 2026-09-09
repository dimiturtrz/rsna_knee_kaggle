#!/usr/bin/env bash
# Wait for latent_register_dense to release the card, then run the mask-0.60 rung on the same trunk.
#
# THE GATE IS THE MARKER *OR* THE pid, never one alone. `latent_register_dense` is a `run_train` arm, so
# `_mark` does fire and an `ARM DONE latent_register_dense` line will exist — but the FILE it lands in is
# named for the CHECKPOINT (`latent_register_dense_s2026.full.log`) while the marker TEXT carries the
# `run_train` name without the seed, and it lives under `$MODELS/experiments/`, not under `logs/`. Three
# separate ways to arm this wrong, each of which holds the card idle while `queue_check` reads OK. The pid
# arm covers the fourth: a death writes no marker at all.
#
# THE pid IS THE WINDOWS pid from `card_check`, so liveness is `ps -W`, never `kill -0` — an MSYS pid is a
# different number and testing it reads a live process as dead.
set -u
cd "$(dirname "$0")/.."

PID="${1:?usage: after_regdeep.sh <windows pid of latent_register_dense>}"
LOG="D:/models/rsna_knee_abnormality/experiments/latent_register_dense_s2026.full.log"

echo "waiting for 'ARM DONE latent_register_dense' in $LOG / pid $PID at $(date -u +%FT%TZ)"
until grep -q "ARM DONE latent_register_dense" "$LOG" 2>/dev/null || ! ps -W | grep -q "[^0-9]$PID[^0-9]"; do
  sleep 60
done
echo "card released at $(date -u +%FT%TZ); marker present: $(grep -qs 'ARM DONE latent_register_dense' "$LOG" && echo yes || echo no)"

bash experiments/arms/latent_register_deep.sh
echo "QUEUE latent_register_deep exited $? $(date -u +%FT%TZ)"
