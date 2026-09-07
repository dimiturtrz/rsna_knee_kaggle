#!/usr/bin/env bash
# Chains the noisy-student branch behind seat_oof5_distill: witness -> teacher -> student.
#
# THE MARKER IS SPELLED WITHOUT THE SEED AND THE LOG IS SPELLED WITH IT. `_mark` writes the `run_train`
# name (`ARM DONE seat_oof5_distill`) while `_arm_log` names the FILE for the checkpoint basename
# (`seat_oof5_distill_s2026.full.log`). Grepping the checkpoint spelling in the marker text waits forever
# with the card idle behind it, and `queue_check` reports OK the whole time.
#
# WHY THE WITNESS GOES FIRST. The four clean twins were fitted at 336/20 over the 0.1-0.9 band and are
# about to teach at 336/12 over 0.2-0.8. Whole-stack plus max pooling makes the depth flexible in
# principle; this measures it instead. Per-member re-grouped AUCs are more diagnostic than one vote
# number — a single degraded member is visible here and invisible in a vote — and the twins' fold-0
# studies decode identically in the canonical and merged3 arrays at the same depth and band, so the
# canonical read answers the question the merged3 read would.
#
# WHY THE TEACHER IS AN INFERENCE PASS RATHER THAN AN `oof_teacher` JOIN. The 1,987 foreign rows have no
# holdout file to join, and a rank is a position within a population: joining a shard ranked over our
# 4,406 to a shard ranked over the foreign 1,987 puts a meaningless separation on every cross-population
# pair, which is exactly the quantity the distill term reads.
set -u
cd "$(dirname "$0")/.."

MARKER="ARM DONE seat_oof5_distill"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/seat_oof5_distill_s2026.full.log"
MODELS="D:/models/rsna_knee_abnormality"
PROCESSED="D:/data/volumetric/mri/processed/rsna_knee_abnormality"
TWINS=("$MODELS"/oof_clean_fold[1-4]_s2026.pt)

until grep -q "$MARKER" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen — the card is free $(date -u +%FT%TZ)"

.venv/Scripts/python.exe -m kneemri.inference.holdout_backfill \
  --checkpoints "${TWINS[@]}" --served-slices 12 --served-band 0.2 0.8
echo "QUEUE teacher witness exited $? $(date -u +%FT%TZ)"

.venv/Scripts/python.exe -m kneemri.inference.teacher_readings \
  --checkpoints "${TWINS[@]}" --scheme merged3 --slices 12 --band 0.2 0.8 --cache-size 336 \
  --fold 0 --out "$PROCESSED/teacher_merged3.npz"
echo "QUEUE teacher_merged3.npz built $? $(date -u +%FT%TZ)"

bash experiments/arms/noisy_student.sh
echo "QUEUE noisy_student exited $? $(date -u +%FT%TZ)"
