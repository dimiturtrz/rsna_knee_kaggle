#!/usr/bin/env bash
# Wait for `deep32_batch6` to release the card, then run the two trunk-axis instruments in cost order.
#
# WHY THESE TWO AND WHY IN THIS ORDER. D118 §4 assigns both. `resnet50d_lr` is seventeen minutes of card
# (its first draw recorded 85-87 s an epoch) and is the cheaper read, so it goes first: if it lands flat,
# `vjepa_cold` is still worth its 3.2 h on a DIFFERENT argument (an unfair draw, D118 §3) rather than the
# same one. Running the expensive arm first would leave the cheap verdict hostage to it.
#
# WHY NO SMOKE. `arm_check --smoke` exists to write a truncated cache for a reading nobody has decoded.
# Both arms reuse a reading that already ran to completion — `resnet50d_short` at 224 px / 20 slices and
# `vjepa_member` at 256 train-size / 20 slices — so the caches are on disk and a smoke would only pay for
# them twice.
#
# WHY THIS GATE PATH. A `run_train` arm re-runs from a snapshot whose `tee` writes
# `$MODELS/experiments/<checkpoint>.full.log`; `logs/` holds the `foreign_run` arms, a different launcher.
# The FILE carries the checkpoint spelling (`deep32_batch6_s2026`) and the MARKER TEXT carries the
# `run_train` name (`deep32_batch6`) -- two different variables, and this greps the marker as an existing
# `ARM DONE` line spells it.
set -u

cd /d/personal_projects/rsna_knee_kaggle || exit 1

MODELS="D:/models/rsna_knee_abnormality"
GATE="$MODELS/experiments/deep32_batch6_s2026.full.log"
LOG="logs/after_instruments.log"

echo "waiting for 'ARM DONE deep32_batch6' in $GATE" >> "$LOG"
until grep -q "ARM DONE deep32_batch6" "$GATE" 2>/dev/null; do sleep 30; done
echo "gate cleared at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

bash experiments/arms/resnet50d_lr.sh >> "$LOG" 2>&1
echo "resnet50d_lr finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

bash experiments/arms/vjepa_cold.sh >> "$LOG" 2>&1
echo "vjepa_cold finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
