#!/usr/bin/env bash
# window_grand.sh — the grand-recipe GPU window, unattended (knee-feb62.10).
# probe224 pair -> grand_recipe -> grand_labelwise_depth24 only if grand clears its gate.
# Gate: grand_recipe@ledger75 >= labelwise_dense_drop40_long_s2026@ledger75 0.9048 + 0.008 = 0.9128.
set -u
cd "$(dirname "$0")/.."
export UV_NO_SYNC=1
source experiments/roots.sh
HOLDOUTS="$MODELS"
GATE=0.9128

stamp() { echo "[$(date -u +%H:%MZ)] $*"; }

uv run python -m kneemri.experiments.card_check || { stamp "card_check refused"; exit 1; }

for arm in probe224_control probe224_grand grand_recipe; do
  stamp "start $arm"
  bash experiments/arms/$arm.sh
  stamp "end $arm"
done

score=$(uv run python -m kneemri.evaluation.ledger75 --holdouts "$HOLDOUTS" --only grand_recipe_s2026 2>/dev/null \
  | awk '$1=="grand_recipe_s2026" {print $3}')
stamp "grand_recipe@ledger75 ${score:-MISSING} vs gate $GATE"
if [ -n "$score" ] && awk -v s="$score" -v g="$GATE" 'BEGIN {exit !(s >= g)}'; then
  stamp "gate passed, start grand_labelwise_depth24"
  bash experiments/arms/grand_labelwise_depth24.sh
  stamp "end grand_labelwise_depth24"
else
  stamp "gate not passed, card released"
fi
stamp "WINDOW DONE"
