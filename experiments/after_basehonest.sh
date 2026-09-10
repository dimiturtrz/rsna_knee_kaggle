#!/usr/bin/env bash
# after_basehonest.sh — wait for `base_honest_batch`, then rebuild the cache `band_wide` reads from.
# Armed 2026-09-10 18:5xZ while the trunk arm was on epoch 0 at ~655 img/s.
#
# WHY A DECODE AND NOT A CARD ARM. `base_honest_batch` is the deciding cell on the capacity axis and its
# three branches are pre-registered, but NONE of them names its own successor: the ladder's next rung
# (base at batch 6, or large at any batch) does not fit this card at 336/20, so even the top branch buys
# a DESIGN question rather than a queued arm. Queueing a trunk arm now would be guessing which branch
# fires. This decode is owed either way and needs no branch.
#
# WHAT IS OWED. The 2026-09-10 cache sweep deleted nine members' arrays. `band_wide_s2026` is the one
# that matters: `band_dense` + `band_wide` = 0.8633 paired, +0.0055 over the better of them, and a greedy
# search over 23 same-split members picks those two first and second. It is the best PAIR the project
# owns. Its checkpoint survived, so it can still be SERVED — what it cannot do without
# `336px_16sl_0.1-0.9band` is be re-scored, re-derived or witnessed offline. This is a REBUILD, not a
# re-train: card time buys nothing here, disk and CPU do.
#
# WHY IT WAITS RATHER THAN RUNS NOW. `build_cache.sh`'s own header: RUN IT ALONE. On 2026-08-11 a build
# beside a trainer had the OS kill BOTH silently - the arm at epoch 7 of 24, the build at 2200 of 4407,
# neither leaving an error, 93.6 GB of RAM against a 63.7 GB memmap plus a ~27 GB trainer. And there is
# no resume: `open_memmap(mode="w+")` recreates the file zeroed, so a kill costs the whole decode. That
# is also why this decode must not be pre-empted once it starts.
#
# THE LOG IS NAMED FOR THE CHECKPOINT AND THE MARKER FOR THE RUN. File
# `base_honest_batch_s2026.full.log` (basename of --out minus .pt); line inside `ARM DONE
# base_honest_batch` (run_train's name). Greping the checkpoint spelling in the marker waits forever
# with an idle machine behind it, and `queue_check` reports OK throughout.
#
# AND THE WAIT IS AN OR WITH PID LIVENESS, because the marker alone hangs forever on a silent exit-0
# death while the pid alone starts the decode the instant a crash frees the card.
source "$(dirname "$0")/lib.sh"

LOG="$MODELS/experiments/base_honest_batch_s2026.full.log"

arm_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*base_honest_batch*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  grep -aq "ARM DONE base_honest_batch" "$LOG" 2>/dev/null && break
  arm_alive || break
  sleep 30
done

echo "base_honest_batch released the machine at $(date -u +%Y-%m-%dT%H:%M:%SZ); decoding 336px_16sl_0.1-0.9band"
bash "$(dirname "$0")/arms/build_cache.sh" --image-size 336 --slices 16 --band-low 0.1 --band-high 0.9
