#!/usr/bin/env bash
# Run the small_feat pair the moment the card frees, so it never idles between arms.
#
# WHY A WAITER AND NOT A CHAIN. `raptor_feat_fit` is already running under its own launcher; appending to
# it is not possible without killing it. This waits on the DEVICE rather than on a pid, because the pid
# that matters is the GPU python grandchild and not the shell that started it — a `TaskStop` on the shell
# reports success while the grandchild keeps the card, which is exactly how two identical scans once ran
# on one device for twenty-five minutes.
#
# IT RUNS BOTH STAGES OR NEITHER: stage two requires stage one's checkpoint (`require` refuses without
# it), so a failed pretrain stops the pair rather than silently fitting from cold and reading as a null.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1

# The full pair is ~85 min; the arm ahead of it has ~2.5 h left. Four hours of waiting is a bound that
# cannot outlive the reason for waiting.
DEADLINE=$(( $(date +%s) + 14400 ))
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if ! .venv/Scripts/python.exe -m kneemri.experiments.card_check 2>&1 | grep -q '^  OURS'; then
    echo "card free at $(date -u); starting the small_feat pair"
    bash experiments/arms/small_feat_pre.sh && bash experiments/arms/small_feat_fit.sh
    echo "small_feat pair finished at $(date -u)"
    exit 0
  fi
  sleep 120
done
echo "DEADLINE: card never freed within 4h, the pair was NOT started"
