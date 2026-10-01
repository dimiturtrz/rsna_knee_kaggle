#!/usr/bin/env bash
# run_noisy_student.sh — build the teacher merged3 read, then run the arm it gates.
#
# The chain in after_seats_done.sh never reached these two steps: its watcher gated on a marker in
# lr4e-5_ws_df.full.log and died with the session that launched it, so noisy_student has sat written,
# beaded P1 and never run since 2026-09-07. Nothing about it is stale — the merged3 cache, the labels
# table and the four fold-0-clean twins are all on disk; only teacher_merged3.npz is missing.
#
# The teacher is the fold-0-clean twins alone. The champion seats are --fold -1 and have seen the
# holdout, so they cannot teach an arm scored on fold 0.
set -u
cd "$(dirname "$0")/.."
PROCESSED="D:/data/volumetric/mri/processed/rsna_knee_abnormality"
MODELS="D:/data/volumetric/mri/models/rsna_knee_abnormality"
CLEAN_TWINS=("$MODELS"/oof_clean_fold[1-4]_s2026.pt)

echo "TEACHER START $(date -u +%FT%TZ)"
.venv/Scripts/python.exe -m kneemri.inference.teacher_readings \
  --checkpoints "${CLEAN_TWINS[@]}" --scheme merged3 --slices 12 --band 0.2 0.8 --cache-size 336 \
  --fold 0 --out "$PROCESSED/teacher_merged3.npz" || {
  echo "QUEUE STOP teacher_merged3.npz failed $(date -u +%FT%TZ)"
  exit 1
}
echo "QUEUE teacher_merged3.npz built $(date -u +%FT%TZ)"

bash experiments/arms/noisy_student.sh || {
  echo "QUEUE STOP noisy_student failed $(date -u +%FT%TZ)"
  exit 1
}
echo "QUEUE noisy_student exited 0 $(date -u +%FT%TZ)"
