#!/usr/bin/env bash
# Waits for the running card arm to exit, then launches foreign_distill_calib so the card is not idle
# between them. The wait is on the WINDOWS pid card_check reported, tested with tasklist: an MSYS pid
# tool reads a Windows python as dead and would start a second job on a card a live arm still holds.
set -u
cd "$(dirname "$0")/.."

# `set -u` does not guard an unbound ${1} inside a pipeline — the subshell dies, the script does not,
# tasklist fails, the leading `!` turns that failure into true and the until loop exits on its first
# iteration straight onto a busy card. Guard it by hand (see after_ortho_encode.sh).
[ $# -ge 1 ] || { echo "usage: $(basename "$0") <windows-pid to wait on>" >&2; exit 2; }

until ! tasklist //FI "PID eq ${1}" 2>/dev/null | grep -q "${1}"; do sleep 60; done
echo "ARM DONE foreign_distill_s2026 — starting foreign_distill_calib $(date -u +%FT%TZ)"
bash experiments/arms/foreign_distill_calib.sh
