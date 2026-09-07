#!/usr/bin/env bash
# Re-run `hisrecipe_distill` and put `lean_weak_columns` behind it. The card is FREE when this starts —
# this is a queue runner, not a watcher, and it gates on nothing.
#
# WHY IT EXISTS. `after_clean_calib_chain.sh` launched `hisrecipe_distill` at 18:19:40Z and it died in
# under a second:
#     external/raptor/knee-mri-training-the-twelve-finding-model.py:68
#     ModuleNotFoundError: No module named 'sklearn'
# `scikit-learn` is a DECLARED dependency (pyproject.toml:67) and the same
# `uv run --no-sync --project . python -c "import sklearn"` prints 1.9.0 from `.venv` four minutes
# later. Nothing was installed in between. The extra was absent for a window because `uv run` re-syncs
# an env it thinks is stale and `uv sync --extra X` uninstalls every extra not named — and every
# pre-commit hook is a `uv run`, so commits landing beside a launching arm can empty the extras under
# it. The arm's own `uv run` calls therefore run with `UV_NO_SYNC=1` exported here, which tells uv not
# to touch the env at all.
#
# AND THE CHAIN RECORDED `returned 0` OVER IT. `foreign_run` swallowed the child's failure, so the
# watcher printed "both arms launched" and exited, and `queue_check` read OK with an idle card for
# three minutes until a human looked. That is a defect in `foreign_run`, not in the chain (knee-qtjb4's
# neighbour); this script asserts the checkpoint exists instead of trusting an exit code.
#
# ORDER (mastermind 18:20Z): hisrecipe_distill (2.4 h, its read decides a slot) -> lean_weak_columns
# (~75 min, the only queued arm with a board path: MCL + Lateral OA carry 0.0073 of the 0.0097 gap).
# `resnet50d_short` follows by hand once its arm is written — D53 closed the width axis at nine seats,
# so neither of these is a seating operation and neither is expected to move the board tonight.
set -u

cd /d/personal_projects/rsna_knee_kaggle || exit 1

LOG="logs/after_hisrecipe_retry.log"
MODELS="D:/models/rsna_knee_abnormality"
export UV_NO_SYNC=1

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
run_arm lean_weak_columns experiments/arms/lean_weak_columns.sh "$MODELS/lean_weak_columns_s2026.pt"

echo "after_hisrecipe_retry finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
exit 0
