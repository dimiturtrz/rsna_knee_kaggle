#!/usr/bin/env bash
# Replaces after_seat_fold0.sh and after_retrain.sh, both killed at 01:19Z.
#
# WHY. The board voided the reading those chains were built on: `oofsolo` 0.926 against a family that
# transfers at 0.79, so the +0.0107 distill term is mostly leak (`teacher_clean.sh` carries the
# mechanism). `oof_fold0` was in service of the retrain and the retrain is not licensed; `compose_v4b_oof`
# and `distill_v3base_oof` both read through the leaky teacher and would inflate the same way, so they
# are HELD rather than queued. The card goes to the ruler instead.
#
# THE MARKER IS SPELLED WITHOUT THE SEED and the LOG IS SPELLED WITH IT — `_mark` writes the run_train
# name while `_arm_log` names the file for the checkpoint basename, and the file lives one directory
# DOWN from the checkpoints, in $MODELS/experiments/.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE seat_distill_lr4e5"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/seat_distill_lr4e5_s2026.full.log"
PROCESSED="D:/data/volumetric/mri/processed/rsna_knee_abnormality"

until grep -q "$MARKER" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen — starting the clean teacher $(date -u +%FT%TZ)"

bash experiments/arms/teacher_clean.sh
echo "QUEUE teacher_clean exited $? $(date -u +%FT%TZ)"

# No --fill. The four clean blocks cover folds 1-4, which is exactly what a fold-0-scored student trains
# on; a fold-0 block here would be the leak walking back in through the front door.
.venv/Scripts/python.exe -m kneemri.evaluation.oof_teacher \
  --twins 'oof_clean_fold[1-4]_s2026.holdout.npz' --out "$PROCESSED/teacher_clean.npz"
echo "QUEUE teacher_clean.npz built $? $(date -u +%FT%TZ)"

bash experiments/arms/seat_clean_distill.sh
echo "QUEUE seat_clean_distill exited $? $(date -u +%FT%TZ)"
