#!/usr/bin/env bash
# Wait for cell A' to release the card, then run cell C: A' verbatim on the maxspan corpus.
#
# WHY A PID WAIT AND NOT AN `ARM DONE` GREP. A' was not launched through `lib.sh`, so it writes no
# `_mark` line at all — a watcher gated on a marker string would wait forever with the card idle
# behind it, which is the failure `queue_check` was written for. Its pid is the only signal that
# exists, so that is what this waits on.
#
# WHY `cd /c/kaggle`. His `_find` walks `root="/kaggle/input"` (T:86), and on Windows that path is
# drive-relative: it resolves against the drive of the current directory. `D:\kaggle\input` now holds
# BOTH his original corpus and the maxspan one, and `os.walk` would take whichever sorts first —
# `raptor-corpus` — so cell C would silently train on the corpus it is supposed to be compared
# against. `C:\kaggle\input` holds one junction, pointing at maxspan and nothing else. Every other
# path below is absolute for the same reason: the cwd is no longer the repo.
set -u

REPO="D:/personal_projects/rsna_knee_kaggle"
PY="$REPO/.venv/Scripts/python.exe"
SCRIPT="$REPO/external/raptor/knee-mri-training-the-twelve-finding-model.py"
LOG="$REPO/logs/repl2x2_c.full.log"
A_PID="${1:?usage: after_cellc.sh <cell A pid>}"

echo "waiting on cell A pid $A_PID" >> "$LOG"
while kill -0 "$A_PID" 2>/dev/null; do sleep 30; done
echo "cell A released the card at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

cd /c/kaggle || exit 1
export PYTHONPATH="$REPO"
nohup "$PY" -m kneemri.experiments.foreign_run \
  --script "$SCRIPT" \
  --arch coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k \
  --res 384 --epochs 16 --bs 8 --k 12 --k_eval 24 --grad_ckpt --ckpt timm \
  --tag repl2x2_c \
  --labels "D:/data/external/raptor/labels/labels_llm_soft.parquet" \
  --folds 1 --fold 0 --fold_file "$REPO/external/raptor/fold0_909.json" \
  --workers 4 >> "$LOG" 2>&1 &
echo "cell C launched as pid $! at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
wait
echo "ARM DONE repl2x2_c" >> "$LOG"
