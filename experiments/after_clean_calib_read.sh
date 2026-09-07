#!/usr/bin/env bash
# Take `clean_calib`'s pre-registered reading off the disk, then put the card straight onto the
# blend-composition axis. No twin is trained here, and the reason is that the twin already exists.
#
# THE TWIN THE 14:43Z ORDER ASKED FOR IS `final_ws_df`, TRAINED, ON DISK. `arm_check` refused the
# `clean_calib_cf` this script used to launch: "is final_ws_df flag for flag, up to --out" (knee-7uoq).
# `final_ws_df.metrics.jsonl` row 0 reads fold -1, ema, 12 epochs, 336 px, 20 slices, band 0.1/0.9,
# calibrated_targets true, distill_weight 0.7, teacher teacher_oof5.npz, and the file holds 13 lines —
# recipe plus twelve epochs, so it converged. The whole-corpus calibrated-distill seat is available now
# rather than at ~19:30Z, and it is on neither `comp/members.roster` nor `comp/members.declines`.
#
# SO THE READING IS STILL WORTH TAKING AND STILL DECIDES SOMETHING — it just no longer decides whether
# to spend 75 minutes of card. `clean_calib` (fold 0, clean teacher, calibrated) is the only honest
# offline stand-in for `final_ws_df` (fold -1, oof5 teacher, calibrated): a `--fold -1` fit holds
# nothing out and cannot be priced by any offline ruler, so its knob has to be read on the fold-0
# member that differs from it in the teacher file alone. Its delta against `seat_clean_distill_s2026`
# on the 909 under LEDGER75 is therefore what licences SEATING `final_ws_df` in the 23:30Z entry.
#
# THE LINE IS +0.005, NOT THE ARM'S OWN +0.008 (mastermind 15:04Z, on D33). D33 priced +0.008 as a coin
# flip on this ruler — median +0.0073, 95% [+0.001, +0.013] — so a bar set there decides the seat by
# sampling noise rather than by the corner. >= +0.005 licences the seat READ; +0.001..+0.005 is
# directional and the seat read still runs; <= +0.001 is a null corner and colsyn ships at 23:30Z.
# The entry itself is decided by that seat read (>= +0.005 L75mid over the champion proxy 0.9237),
# never by this number. >= +0.0135 would be a separate finding — "distillation amplifies the knob" —
# ledgered if it happens and not planned on.
#
# WHAT THE CARD DOES INSTEAD, and unconditionally. D32 closed the column-specialist route and named
# blend composition as the lever: more independent readers of the same twelve. `retrain_six_distill.sh`
# is that arm and four of its six seats are unfit. It is safe to relaunch as written: `run_train`
# REFUSES an arm whose `--out` already holds a checkpoint (lib.sh:580) and `lib.sh` sets only `-u`, so
# the two fitted seats print a refusal line and the arm walks on to the four that are missing. Seat six
# `merged_blend_df` runs FIRST and is fold-0, so the family's only offline kill-switch reading lands
# ~1.2 h in — and it now trains on `teacher_clean_full.npz`, which no model that saw fold 0 wrote.
#
# THE MARKER AND THE FILE ARE SPELLED DIFFERENTLY, and both spellings here are the ones that exist:
# `_arm_log` names the file for the checkpoint (`clean_calib_s2026.full.log`) and `_mark` writes the
# `run_train` name (`ARM DONE clean_calib`). Grepping the checkpoint spelling in the marker waits forever
# with the card idle behind it, and `queue_check` reports OK the whole time (CLAUDE.md, knee-ket1c).
set -u

cd /d/personal_projects/rsna_knee_kaggle || exit 1

GATE="logs/clean_calib_s2026.full.log"
LOG="logs/after_clean_calib_read.log"
PY=".venv/Scripts/python.exe"
HOLDOUTS="D:/models/rsna_knee_abnormality"
CHILD="clean_calib_s2026"
PARENT="seat_clean_distill_s2026"

echo "waiting for 'ARM DONE clean_calib' in $GATE" >> "$LOG"
until grep -q "ARM DONE clean_calib" "$GATE" 2>/dev/null; do sleep 30; done
echo "gate cleared at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

READING="logs/clean_calib_ledger75.log"
"$PY" -m kneemri.evaluation.ledger75 --holdouts "$HOLDOUTS" --only "$CHILD" "$PARENT" > "$READING" 2>&1
cat "$READING" >> "$LOG"

# Column 3 is LEDGER75 in the table `ledger75.main` prints; the two rows are matched by member stem.
DELTA=$(awk -v c="$CHILD" -v p="$PARENT" '
  $1 == c { child = $3 }
  $1 == p { parent = $3 }
  END { if (child == "" || parent == "") { print "NA" } else { printf "%.4f", child - parent } }
' "$READING")

echo "LEDGER75 $CHILD - $PARENT = $DELTA" >> "$LOG"

if [ "$DELTA" = "NA" ]; then
  echo "one of the two rows is missing from the reading; the seat licence has no number" >> "$LOG"
elif awk -v d="$DELTA" 'BEGIN { exit !(d >= 0.005) }'; then
  echo "SEAT READ LICENSED: >= +0.005 — price final_ws_df in the champion seat via the clean_calib stand-in" >> "$LOG"
elif awk -v d="$DELTA" 'BEGIN { exit !(d > 0.001) }'; then
  echo "DIRECTIONAL: +0.001..+0.005 — seat read still runs; the entry ships only if it clears +0.005" >> "$LOG"
else
  echo "CORNER NULL: <= +0.001 — colsyn ships at 23:30Z" >> "$LOG"
fi
if awk -v d="$DELTA" 'BEGIN { exit !(d >= 0.0135) }' 2>/dev/null; then
  echo "SEPARATE FINDING: >= +0.0135 — distillation amplifies the knob; ledger it" >> "$LOG"
fi

echo "launching retrain_six_distill at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
bash experiments/arms/retrain_six_distill.sh >> "$LOG" 2>&1

echo "after_clean_calib_read finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
