#!/usr/bin/env bash
# after_ortholora.sh — TRUNCATE the CARVE budget sweep at its third point, then smoke and run `ortho_lora`.
#
# WHY IT TRUNCATES, decided 2026-09-09 at 14:23Z with the third point 36 minutes in. The sweep's question
# was existential -- does ANY adapt budget move the decode cost -- and its first three points answer it:
# |dtheta| 2e-5 -> -0.0307, 2e-3 -> -0.0309, 2e-2 -> -0.0360, with the KEPT auc collapsing 0.8504 -> 0.7570
# at the third. That is monotone damage and it is the BUDGET DAMAGES branch. The fourth point sits at
# |dtheta| ~ 1e-1, which the arm's own header calls "the trunk is being broken, not adapted" -- a
# catastrophe that is guaranteed before it runs and changes no branch after it. It costs ~50 minutes of a
# card that has a MODEL waiting, so it is not run. A truncation is not a promotion: the sweep's reading is
# whatever its first three points say, and the ledger records it as truncated-by-decision.
#
# THE GATE IS AN OR OF THREE, and the previous version's third leg WAS DEAD. It tested
# `nvidia-smi --query-compute-apps` for an empty list, which on this WDDM desktop returns 32 pids whatever
# is running, so the silent-exit-0 death it was written for could never have fired it. The replacement asks
# for OUR process by its command line, which `Get-CimInstance Win32_Process` does expose for bash and
# python even though MSYS `ps -W` does not.
#   * the third point's artefact on disk — the intended finish.
#   * the sweep's own DONE marker — in case it gets past point four before this fires.
#   * no live `foreign_readout` process — the silent death, now actually detectable.
source "$(dirname "$0")/lib.sh"

LOG="logs/after_budget.log"
THIRD="$MODELS/carve_budget_lr3s10.npz"

sweep_alive() {
  powershell -NoProfile -Command \
    "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*foreign_readout*' }).Count" \
    2>/dev/null | tr -d '\r' | grep -qv '^0$'
}

while true; do
  [ -f "$THIRD" ] && break
  grep -aq "BUDGET SWEEP DONE" "$LOG" && break
  sweep_alive || break
  sleep 30
done
echo "=== SWEEP GATE OPEN at $(date -u +%H:%MZ) ==="

# The fourth point is refused rather than waited out. Killing the reader releases the card; the sweep's
# shell then runs its own echo and exits, which is why the marker may appear after this line.
powershell -NoProfile -Command \
  "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*foreign_readout*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" \
  2>/dev/null
echo "=== FOURTH POINT REFUSED - see this script's header ==="
sleep 20

# A recipe is smoked before it is queued for an hour and a half. Sixty studies, 48 train / 12 holdout.
# ortholarge has never been run under adapters, so the failure this catches is a variant LoRALinear cannot
# wrap, and it costs two minutes rather than the arm's first epoch.
# THE VERDICT IS THE ARM'S OWN LINE, NOT THE EXIT CODE, and that distinction cost 31 minutes of idle
# card on 2026-09-09. `arm_check` exits non-zero for any finding in the WHOLE arms directory -- here, six
# other arms carrying CRLF (knee-rlg5) -- so `|| exit` read a repo-wide hygiene finding as this recipe
# failing on the card and refused to launch a smoke that had printed `ok ortho_lora`, twice. The gate
# that matters is what the smoke said about THIS arm: `BROKEN <name>` or `NO ROOM <name>`.
SMOKE_LOG="$MODELS/smoke_ortho_lora.check.log"
UV_NO_SYNC=1 uv run python -m kneemri.experiments.arm_check \
  --only ortho_lora --smoke "$MODELS/smoke_ortho_lora" --smoke-studies 60 2>&1 | tee "$SMOKE_LOG"
if grep -qaE "^(BROKEN|NO ROOM) ortho_lora" "$SMOKE_LOG" || ! grep -qa "^ok ortho_lora" "$SMOKE_LOG"; then
  echo "=== SMOKE FAILED - ortho_lora NOT launched ==="
  exit 1
fi
echo "=== SMOKE PASSED ==="

bash "$(dirname "$0")/arms/ortho_lora.sh"
