#!/usr/bin/env bash
# Builds the FULL-strength merged3 teacher the moment noisy_student frees the card.
#
# Unconditional, and deliberately so: it is ~15 min over a predecoded array, it is a prerequisite of the
# tranche-one arm on EITHER branch that runs it, and building it while the branch is being read costs
# nothing but would otherwise sit in front of 4.2 h of seats. If the branch does not fire, the file is
# 6,393 rows on disk and the canonical retrain_six_distill path is unaffected.
#
# ONE PASS, FIVE TWINS, ONE RANK POPULATION -- see experiments/arms/merged3_tranche_one.sh for why this
# may not be teacher_oof5.npz joined to an external-only pass.
#
# The marker text is the run_train name; the LOG FILE is the checkpoint spelling. They are two different
# variables (lib.sh:580 vs :642) and only one of them carries the seed.
MARKER="ARM DONE noisy_student"
ARM_LOG="D:/models/rsna_knee_abnormality/experiments/noisy_student_s2026.full.log"
MODELS="D:/models/rsna_knee_abnormality"
PROCESSED="D:/data/volumetric/mri/processed/rsna_knee_abnormality"
TWINS=("$MODELS"/oof_fold[0-4]_s2026.pt)

until grep -q "$MARKER" "$ARM_LOG" 2>/dev/null; do sleep 60; done
echo "$MARKER seen — the card is free $(date -u +%FT%TZ)"

.venv/Scripts/python.exe -m kneemri.inference.teacher_readings \
  --checkpoints "${TWINS[@]}" --scheme merged3 --slices 12 --band 0.2 0.8 --teach-all \
  --out "$PROCESSED/teacher_merged3_full.npz"
echo "QUEUE teacher_merged3_full.npz built $? $(date -u +%FT%TZ)"
