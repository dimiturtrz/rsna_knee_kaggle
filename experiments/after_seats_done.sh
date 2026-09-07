#!/usr/bin/env bash
# Chains the warm twin, the merged3 teachers, the noisy student and the composition behind the LAST
# retrain seat.
#
# THE WARM TWIN GOES FIRST OF THE FOUR because it is the only step whose result changes the others'
# SHAPE: `final_ws_wf` against the from-scratch `final_ws_df` prices the warm start itself, and if the
# two land inside the floor every retrain after this one runs at four epochs instead of twelve.
#
# WHY IT IS GATED ON `lr4e-5_ws_df` AND NOT ON `final_ws_df`, WHICH IS WHAT WAS ORDERED. The marker a
# chain waits on is written by the SEQUENCER (`_mark`, lib.sh:642) after the trainer exits, not by the
# trainer. So jumping the queue in front of the four remaining seats means killing the sequencer that is
# running them, and killing it destroys the very `ARM DONE` line the jump-chain would have to wait on:
# the arm keeps training as an orphan and no marker is ever written. The card stays booked either way —
# the only thing the jump buys is an ordering, and it costs the signal. Seats first, then this.
#
# WHY IT REPLACES TWO WATCHERS RATHER THAN JOINING THEM. `after_retrain.sh` owned compose behind the
# seats and `after_noisy_student.sh` owned the full-strength teacher behind the student; both are killed
# by whoever arms this, because both would collide with it on the card. `after_noisy_student.sh` was
# also BROKEN: its teacher pass carries no `--cache-size`, which is the defect that killed the 04:56Z
# pass — a 224 px twin moved onto merged3 resolves `images_224px_12sl_merged3.npy`, a file that has
# never existed, because merged3 was only ever built at 336.
#
# EVERY STEP STOPS THE CHAIN. The 04:56Z failure was silent because a step exited 1 and the queue went
# on regardless while a second watcher on the same marker kept the card busy, so `queue_check` read OK
# throughout. A chain that hands the card on after a failed step is not a queue, it is a lottery.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE lr4e-5_ws_df"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/lr4e-5_ws_df.full.log"
MODELS="D:/models/rsna_knee_abnormality"
PROCESSED="D:/data/volumetric/mri/processed/rsna_knee_abnormality"
CLEAN_TWINS=("$MODELS"/oof_clean_fold[1-4]_s2026.pt)
OOF_TWINS=("$MODELS"/oof_fold[0-4]_s2026.pt)

until grep -q "$MARKER" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen — the card is free $(date -u +%FT%TZ)"

bash experiments/arms/final_ws_wf.sh || {
  echo "QUEUE STOP final_ws_wf failed $(date -u +%FT%TZ)"
  exit 1
}
echo "QUEUE final_ws_wf exited 0 $(date -u +%FT%TZ)"

.venv/Scripts/python.exe -m kneemri.inference.teacher_readings \
  --checkpoints "${CLEAN_TWINS[@]}" --scheme merged3 --slices 12 --band 0.2 0.8 --cache-size 336 \
  --fold 0 --out "$PROCESSED/teacher_merged3.npz" || {
  echo "QUEUE STOP teacher_merged3.npz failed $(date -u +%FT%TZ)"
  exit 1
}
echo "QUEUE teacher_merged3.npz built $(date -u +%FT%TZ)"

.venv/Scripts/python.exe -m kneemri.inference.teacher_readings \
  --checkpoints "${OOF_TWINS[@]}" --scheme merged3 --slices 12 --band 0.2 0.8 --cache-size 336 \
  --teach-all --out "$PROCESSED/teacher_merged3_full.npz" || {
  echo "QUEUE STOP teacher_merged3_full.npz failed $(date -u +%FT%TZ)"
  exit 1
}
echo "QUEUE teacher_merged3_full.npz built $(date -u +%FT%TZ)"

bash experiments/arms/noisy_student.sh || {
  echo "QUEUE STOP noisy_student failed $(date -u +%FT%TZ)"
  exit 1
}
echo "QUEUE noisy_student exited 0 $(date -u +%FT%TZ)"

bash experiments/arms/compose_v4b_clean.sh || {
  echo "QUEUE STOP compose_v4b_clean failed $(date -u +%FT%TZ)"
  exit 1
}
echo "QUEUE compose_v4b_clean exited 0 $(date -u +%FT%TZ)"
