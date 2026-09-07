#!/usr/bin/env bash
# Replaces after_clean_teacher.sh (pid 1462989, killed 01:35Z) with the WITNESS inserted between the
# clean teachers and the student that reads them.
#
# WHY THE SWAP RATHER THAN AN EDIT. `after_clean_teacher.sh` was live and mid-script; bash reads a
# running script by byte offset, so editing it in place rewrites the lines it has not reached yet. The
# teacher loop is a CHILD process and survives its parent, so killing the watcher leaves all four clean
# teachers running and only the chaining is re-armed here.
#
# WHY THE WITNESS GOES HERE. `seat_clean_distill` measures what removing the leak COSTS; it cannot show
# the leak exists. `teacher_witness` correlates the distilled student's fold-0 RESIDUALS against the
# mean fold-0 residuals of the four teachers that trained on fold 0, with `band_dense` (same fold, no
# teacher in its loss) as the control and the four clean teachers as the second control. It needs both
# teacher sets on disk, and it needs the card free — which is exactly the gap between the last teacher
# and the student. Minutes, and it explains the number the next arm produces.
#
# THE MARKER IS SPELLED WITHOUT THE SEED and the LOG IS SPELLED WITH IT — `_mark` writes the run_train
# name while `_arm_log` names the file for the checkpoint basename, and the file lives one directory
# DOWN from the checkpoints, in $MODELS/experiments/.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE oof_clean_fold4"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/oof_clean_fold4_s2026.full.log"
MODELS="D:/models/rsna_knee_abnormality"
PROCESSED="D:/data/volumetric/mri/processed/rsna_knee_abnormality"

until grep -q "$MARKER" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen — the four clean teachers are on disk $(date -u +%FT%TZ)"

.venv/Scripts/python.exe -m kneemri.evaluation.teacher_witness \
  --student "$MODELS/seat_distill_lr4e5_s2026.holdout.npz" \
  --control "$MODELS/band_dense_s2026.holdout.npz" --device cuda
echo "QUEUE teacher_witness exited $? $(date -u +%FT%TZ)"

# No --fill. The four clean blocks cover folds 1-4, which is exactly what a fold-0-scored student trains
# on; a fold-0 block here would be the leak walking back in through the front door.
.venv/Scripts/python.exe -m kneemri.evaluation.oof_teacher \
  --twins 'oof_clean_fold[1-4]_s2026.holdout.npz' --out "$PROCESSED/teacher_clean.npz"
echo "QUEUE teacher_clean.npz built $? $(date -u +%FT%TZ)"

bash experiments/arms/seat_clean_distill.sh
echo "QUEUE seat_clean_distill exited $? $(date -u +%FT%TZ)"
