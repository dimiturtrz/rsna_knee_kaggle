#!/usr/bin/env bash
# Chains the oof5 branch behind the clean student: fifth twin -> five-block teacher -> fold -1 student.
#
# THE MARKER IS SPELLED WITHOUT THE SEED AND THE LOG IS SPELLED WITH IT. `_mark` writes the `run_train`
# name (`ARM DONE seat_clean_distill`) while `_arm_log` names the FILE for the checkpoint basename
# (`seat_clean_distill_s2026.full.log`), one directory down from the checkpoints. Grepping the
# checkpoint spelling in the marker text waits forever with the card idle behind it, and `queue_check`
# reports OK the whole time.
#
# WHY NO --fill. `oof_teacher --fill` patches uncovered studies from a member's own reading; here every
# one of the 4,406 is covered by exactly one twin that held it out, which is the property the whole
# branch rests on. A fill would put an in-sample row into a teacher whose only claim is that it has none.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE seat_clean_distill"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/seat_clean_distill_s2026.full.log"
PROCESSED="D:/data/volumetric/mri/processed/rsna_knee_abnormality"

until grep -q "$MARKER" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen — the clean student is read $(date -u +%FT%TZ)"

bash experiments/arms/oof_fold0.sh
echo "QUEUE oof_fold0 exited $? $(date -u +%FT%TZ)"

.venv/Scripts/python.exe -m kneemri.evaluation.oof_teacher \
  --twins 'oof_fold[0-4]_s2026.holdout.npz' --out "$PROCESSED/teacher_oof5.npz"
echo "QUEUE teacher_oof5.npz built $? $(date -u +%FT%TZ)"

bash experiments/arms/seat_oof5_distill.sh
echo "QUEUE seat_oof5_distill exited $? $(date -u +%FT%TZ)"
