#!/usr/bin/env bash
# Wait for `lean_weak_columns`, then re-run `hisrecipe_distill` (now that its missing import is fixed)
# and put `resnet50d_short` behind it. The card does not idle overnight.
#
# WHY hisrecipe_distill IS BEING RUN A THIRD TIME. It died twice today, both times in under a second:
#     external/raptor/knee-mri-training-the-twelve-finding-model.py:68
#     ModuleNotFoundError: No module named 'sklearn'
# The first reading — a `uv run` re-sync emptying the extras under a launching arm — was WRONG, and it
# was wrong in the way that matters: `scikit-learn` imports fine from `.venv`, so the check that
# "proved" the env was healthy asked the wrong interpreter. Arms do not run in `.venv`. `lib.sh:34`
# exports `UV_PROJECT_ENVIRONMENT="$REPO/.venv-arm"`, a second environment nothing else ever syncs, and
# `scikit-learn` was absent from THAT one. Installed at 18:26Z (`uv pip install --python
# .venv-arm/Scripts/python.exe scikit-learn`, 1.9.0), verified by importing from `.venv-arm` itself.
# The general rule this cost an hour of card to learn: **verify an arm's dependency with
# `.venv-arm/Scripts/python.exe`, never with `.venv`** — the two environments are provisioned
# separately and only one of them ever runs an arm.
#
# AND AN EXIT CODE IS NOT EVIDENCE. Both deaths were recorded as `hisrecipe_distill returned 0`, because
# `foreign_run.py:71` runs the foreign script through `runpy.run_path` and does not translate a failure
# into a non-zero status. `queue_check` read OK with an idle card behind it. So this script asserts the
# CHECKPOINT, and says the card is idle when there is none (knee-, filed 18:25Z).
#
# ORDER (mastermind 18:17Z / 18:20Z): lean_weak_columns -> hisrecipe_distill -> resnet50d_short. Lean is
# already on the card. `resnet50d_short` is last because its pre-registration buys a RULER and no board
# path, while hisrecipe's read can still decide a slot; D53 closed the width axis at nine seats, so
# neither is a seating operation and neither is expected to move the board tonight.
set -u

cd /d/personal_projects/rsna_knee_kaggle || exit 1

LOG="logs/after_lean_chain.log"
MODELS="D:/models/rsna_knee_abnormality"
GATE="$MODELS/experiments/lean_weak_columns_s2026.full.log"
GATE_ALT="logs/lean_weak_columns_s2026.full.log"
export UV_NO_SYNC=1

echo "waiting for 'ARM DONE lean_weak_columns' in $GATE or $GATE_ALT" >> "$LOG"
until grep -q "ARM DONE lean_weak_columns" "$GATE" 2>/dev/null || grep -q "ARM DONE lean_weak_columns" "$GATE_ALT" 2>/dev/null; do sleep 30; done
echo "gate cleared at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

run_arm() {
  local name="$1" script="$2" checkpoint="$3"
  echo "launching $name at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  bash "$script" >> "$LOG" 2>&1
  local status=$?
  if [ -f "$checkpoint" ]; then
    echo "$name LANDED $checkpoint at $(date -u +%Y-%m-%dT%H:%M:%SZ) (exit $status)" >> "$LOG"
  else
    echo "$name LEFT NO CHECKPOINT at $(date -u +%Y-%m-%dT%H:%M:%SZ) (exit $status) — the card is idle" >> "$LOG"
  fi
}

run_arm hisrecipe_distill experiments/arms/hisrecipe_distill.sh "$MODELS/hisrecipe_distill_s2026.pt"
run_arm resnet50d_short experiments/arms/resnet50d_short.sh "$MODELS/resnet50d_short_s2026.pt"

echo "after_lean_chain finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
exit 0
