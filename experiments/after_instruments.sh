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
#
# WHY `vjepa_cold` STAYS ARMED AFTER D125 §3 PUT IT ON HOLD. The hold is for the `options`-read re-bucket
# (D123 §4): `--variant` defaults to DINOv2-small (`kneemri/training/encoder_flags.py:52-56`, verified
# here), so D117's name-parsed "default" bucket is a mixture and the order-statistic null that priced
# every family has no reference population. That defect changes the NULL this arm is compared against.
# It does not change the arm's own argument, which is D118 §3: the volumetric family's four draws are
# two crashes, one TRUNCATED run and one warm start off its own encoder, so it has never had a matched
# twelve-epoch cold fit. That sentence is true whatever the re-bucket says. Holding it would leave the
# card idle for 3.2 h to protect a comparison that can be recomputed off the checkpoint afterwards,
# and the standing owner directive is that the card cannot be the blocker. Raised to mastermind in the
# same wake; if the re-bucket lands before the gate clears, this file gets rewritten, never edited live.
#
# `retrain_six_distill` IS NOT HERE, and was never queued. D121 ordered it first on a per-seat +0.0107;
# D122/D123 showed that is `oof_teacher_s2026`'s LEAKED number (voided in `comp/board.jsonl:118`), the
# clean student's term is +0.0024-+0.0056 under a 0.0062 floor, and D124 measured its pool term at
# -0.0003. D125 withdrew it in writing. Independently: `teacher_oof5.npz` is not on disk, so the arm
# would have died in epoch one.
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
