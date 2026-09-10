#!/usr/bin/env bash
# after_fastmri.sh — wait for `ortho_fastmri`, then run `ortho_fastmri_masked` on the same card. Armed
# 2026-09-10 22:10Z, with the first arm at epoch 1 of 12 and due to finish around 23:30Z.
#
# IT IS NOT GATED ON A SCORE, deliberately, and that is the difference from `after_resnetlong.sh`. The
# masked arm answers a question in TWO of the three branches its control pre-registers — a flat with the
# ACL-up/OA-down shape, and an outright regression — and in the third (a clear win) the follow-up is a
# 336/20 DECODE, which is a RAM job that `build_cache` must run alone and so could never be this
# watcher's payload anyway. So there is no branch in which the card should sit idle waiting for a human
# to read a number, and the only gate that matters is that the first arm actually produced one.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN: the file is
# `ortho_fastmri_s2026.full.log` (basename of --out minus .pt) while the line inside reads
# `ARM DONE ortho_fastmri`. And the wait is an OR with pid liveness, because the marker alone hangs
# forever on a silent exit-0 death while the pid alone fires the next arm the instant a crash frees the
# card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/ortho_fastmri_s2026.full.log"
CKPT="$MODELS/ortho_fastmri_s2026.pt"

first_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*ortho_fastmri*' -and \$_.CommandLine -notlike '*masked*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE ortho_fastmri" "$LOG" 2>/dev/null && break
  first_alive || break
  sleep 30
done
echo "=== ortho_fastmri GATE OPEN at $(date -u +%H:%MZ) ==="

if [ ! -f "$CKPT" ]; then
  echo "=== NO CHECKPOINT - ortho_fastmri died; masked arm NOT launched, since it would price a corpus against nothing ==="
  exit 1
fi

bash "$(dirname "$0")/arms/ortho_fastmri_masked.sh"
