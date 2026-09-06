#!/usr/bin/env bash
# Waits for foreign_distill_calib to exit, then runs the card queue in the order the 17:25Z letter set:
# the decode sweep first (ten minutes, and the only queued work that FEEDS the 23:30Z entry), then the
# 0.7 teacher mix (the DIRECT test of the mechanism the distill null was explained by), then dissent
# (the indirect one). It replaces after_foreign_distill_calib.sh, which went straight to dissent.
#
# The wait is on the WINDOWS pid card_check reported, tested with tasklist: an MSYS pid tool reads a
# Windows python as dead and would start a second job on a card a live arm still holds.
set -u
cd "$(dirname "$0")/.."

# `set -u` does not guard an unbound ${1} inside a pipeline — the subshell dies, the script does not,
# tasklist fails, the leading `!` turns that failure into true and the until loop exits on its first
# iteration straight onto a busy card. Guard it by hand (see after_ortho_encode.sh).
[ $# -ge 1 ] || { echo "usage: $(basename "$0") <windows-pid to wait on>" >&2; exit 2; }

until ! tasklist //FI "PID eq ${1}" 2>/dev/null | grep -q "${1}"; do sleep 60; done
echo "ARM DONE foreign_distill_calib_s2026 — starting the queue $(date -u +%FT%TZ)"

bash experiments/arms/foreign_decode_sweep.sh
echo "QUEUE decode sweep exited $? $(date -u +%FT%TZ)"

bash experiments/arms/foreign_distill_mix07.sh
echo "QUEUE mix07 exited $? $(date -u +%FT%TZ)"

bash experiments/arms/foreign_dissent.sh
echo "QUEUE dissent exited $? $(date -u +%FT%TZ)"
