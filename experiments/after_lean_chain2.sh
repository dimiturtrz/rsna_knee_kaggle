#!/usr/bin/env bash
# Replaces `after_lean_chain.sh` (killed while it was still on the gate, before it launched anything).
# Same gate — `ARM DONE lean_weak_columns` — a new order behind it.
#
# ORDER (mastermind 19:15Z revival sweep): lean_weak_columns -> recal_calib -> resnet50d_short ->
# hisrecipe_distill.
#
# WHY recal_calib JUMPS THE QUEUE. Bead knee-4h1if, 75 min, and it is the only open arm whose mechanism
# can lift the BLOCK rather than buy a seat. D53 closed the width axis at nine seats and prices an
# ours-seat at <= +0.0004; D56 prices the best-case diversity seat at +0.0004; D58 retired his labels as
# a training axis and with them the `mixed_labels_dense` path, which was the ranked list's only other
# block-lift item. What is left is labels v3: priced +0.0045 and +0.0052 on two pairs, NEITHER carrying
# calibration, so the composition with `--calibrated-targets` has never been read. On a hit the six-block
# refits with v3 at w = 2/3 ~= +0.005 board. 75 min ahead of hisrecipe's 2.4 h at a weaker pitch — his
# fold-0 checkpoints read +0.0000 as extra seats (D53 SS5), and 5.7 h of twin refits must not be bought on
# an untested composition.
#
# hisrecipe_distill goes LAST, not dropped: it is still the only 1/3-class object in the queue, and
# `kneemri-hisseat` was tomorrow's slot either way. It reads ~23:20Z instead of ~22:05Z.
#
# THE CHECKPOINT IS THE SUCCESS MARKER, NOT THE EXIT CODE. hisrecipe died twice today in under a second
# (`ModuleNotFoundError: No module named 'sklearn'` — arms run in `.venv-arm`, not `.venv`) and the chain
# recorded `returned 0` both times, because `lib.sh` runs under `set -u` and not `set -e`, so a dead
# trainer fell through to `release_card` and exited with THAT status. Both foreign arms now capture the
# trainer's status by hand (4d54f1e0); this script still asserts the checkpoint and says the card is idle
# when there is none.
set -u

cd /d/personal_projects/rsna_knee_kaggle || exit 1

LOG="logs/after_lean_chain2.log"
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

run_arm recal_calib experiments/arms/recal_calib.sh "$MODELS/recal_calib_s2026.pt"
run_arm resnet50d_short experiments/arms/resnet50d_short.sh "$MODELS/resnet50d_short_s2026.pt"
run_arm hisrecipe_distill experiments/arms/hisrecipe_distill.sh "$MODELS/hisrecipe_distill_s2026.pt"

echo "after_lean_chain2 finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
exit 0
