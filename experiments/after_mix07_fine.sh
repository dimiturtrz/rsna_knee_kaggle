#!/usr/bin/env bash
# Waits for foreign_distill_mix07 to exit, then runs the FINE band sweep before dissent.
#
# It replaces the tail of after_calib_chain.sh, which had dissent (~70 min) queued directly behind
# mix07. The coarse sweep placed the band peak inside a 0.04 step, and the fine sweep is the only
# queued work that can still feed the 23:30Z slot — its kernel has to be pushed by 21:00Z to make
# census by 22:30Z, which a 70-minute arm in front of it makes impossible. Dissent loses nothing by
# waiting: nothing downstream of it is on a clock today.
#
# after_calib_chain.sh was KILLED rather than edited, because bash reads a script by byte offset while
# it runs and an insert ahead of the interpreter's position feeds it garbage. Killing the chain's own
# shell leaves the arm it had already started running under its own.
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
echo "ARM DONE foreign_distill_mix07_s2026 — starting the queue $(date -u +%FT%TZ)"

bash experiments/arms/foreign_decode_fine.sh
echo "QUEUE fine sweep exited $? $(date -u +%FT%TZ)"

bash experiments/arms/foreign_dissent.sh
echo "QUEUE dissent exited $? $(date -u +%FT%TZ)"
