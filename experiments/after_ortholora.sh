#!/usr/bin/env bash
# after_ortholora.sh — smoke and then run `ortho_lora` the moment the CARVE budget sweep releases the card.
#
# THE GATE IS AN OR OF THREE, and each leg covers a way the other two hang.
#   * the sweep's own marker in its log — the clean finish.
#   * the LAST point's artefact on disk — the marker is echoed by the arm script, so a death between the
#     last read and the echo would strand a completed sweep.
#   * an EMPTY compute-app list on the card — the silent exit-0 death, which is exactly how the perstudy
#     read failed earlier today. Without this leg a crashed sweep holds the card idle indefinitely while
#     `queue_check` reads OK, because a live watcher gating an owed arm is all it can see.
# `nvidia-smi` is asked rather than the process table because the question is about the DEVICE (knee-j99t),
# and the other project shares this card by arrangement — so the leg fires only on OUR absence being
# consistent with the other two legs, which is why it is an OR and not a sole trigger.
source "$(dirname "$0")/lib.sh"

LOG="logs/after_budget.log"
LAST="$MODELS/carve_budget_lr2s10.npz"

while true; do
  grep -aq "BUDGET SWEEP DONE" "$LOG" && break
  [ -f "$LAST" ] && break
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && break
  sleep 30
done
echo "=== SWEEP GATE OPEN ==="

# A recipe is smoked before it is queued for an hour and a half. Sixty studies, 48 train / 12 holdout.
# ortholarge has never been run under adapters, so the failure this catches is a variant that LoRALinear
# cannot wrap, and it costs two minutes rather than the arm's first epoch.
UV_NO_SYNC=1 uv run python -m kneemri.experiments.arm_check \
  --only ortho_lora --smoke "$MODELS/smoke_ortho_lora" --smoke-studies 60 || {
  echo "=== SMOKE FAILED - ortho_lora NOT launched ==="
  exit 1
}
echo "=== SMOKE PASSED ==="

bash "$(dirname "$0")/arms/ortho_lora.sh"
