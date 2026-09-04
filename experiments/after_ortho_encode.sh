#!/usr/bin/env bash
# Waits for the running card arm to exit, then encodes the pixel cache through OrthoDiffusion's
# three plane denoisers. The wait is on the WINDOWS pid card_check reported, tested with tasklist:
# an MSYS pid tool reads a Windows python as dead and would start a second job on the card.
set -u
cd "$(dirname "$0")/.."

# `set -u` DOES NOT GUARD THIS, measured 2026-09-04. The unbound `${1}` below expands inside a pipeline,
# so the SUBSHELL dies and the script does not; `tasklist` then fails, the leading `!` turns that failure
# into true, and the `until` loop exits on its first iteration straight into the six-hour encode. That is
# how two of these builds landed on a card a live arm was holding (`medsam_dense` halved, 508 -> 280
# img/s), and it is why this file no longer lives in `experiments/arms/`: `arm_check.invocations()` runs
# every `arms/*.sh` bare to read its flags, so the lint that checks arms was the launcher.
[ $# -ge 1 ] || { echo "usage: $(basename "$0") <windows-pid to wait on>" >&2; exit 2; }

until ! tasklist //FI "PID eq ${1}" 2>/dev/null | grep -q "${1}"; do sleep 60; done
echo "ARM DONE through_dense_ramp — starting ortho encode $(date -u +%FT%TZ)"
timeout 21600 .venv/Scripts/python.exe -m kneemri.embeddings.build_ortho_embeddings --batch 4
echo "ARM DONE ortho_encode $(date -u +%FT%TZ)"
