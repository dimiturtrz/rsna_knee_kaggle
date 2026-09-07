#!/usr/bin/env bash
# Take `clean_calib`'s pre-registered reading off the disk, then run the card in the order the 15:55Z
# mastermind letter set. REPLACES `after_clean_calib_read.sh`, which launched the six twins
# unconditionally the moment the reading was printed.
#
# WHY THE ORDER CHANGED. The twins are a SEAT operation on a block already priced: D35 fits an added
# honest seat at 1/6x2/3 to -0.0009 + 0.076*lift + 0.0139*(1-corr), so six retrained-but-similar seats
# buy the block's w = 2/3 only if the treatment moves each seat's SOLO. `foreign_labels_dense` is the
# arm that measures whether such a treatment exists: if his report table lifts OUR recipe's solo by
# +0.01, that lift applies to ALL SIX at once (w = 2/3, ~ +0.005 board) and it is lever 1's shape, not
# a seventh voice. So the label read goes FIRST and the twins go behind a gate.
#
# THE GATE ON THE TWINS IS THE CLEAN_CALIB READ, at +0.004 rather than the seat licence's +0.005: the
# twins are a six-member block (D36 job 1), and "+0.000 per twin" is the right price for ONE twin and
# the wrong one for six together, because the fit's w is the BLOCK's.
#
# HARNESS FLAGS FOR `foreign_labels_dense`, both checked before this was written (15:50Z):
#   (i)  `--calibrated-targets` STAYS. It is `LevelCalibration` (evaluation/level_calibration.py:72-152):
#        one empirical gold rate per unique reader value, then pool-adjacent-violators weighted by the
#        cell COUNT (:112-137), and any level the fit never saw is left untouched (:139-152). His table's
#        14 distinct values are therefore not 14 independent estimates - PAV spends its correction where
#        the evidence is thin and merges a twelve-cell bucket into its neighbour. Nothing in the fit is
#        sized for our five-level vocabulary, so the flag does what it says on a 14-level table too.
#   (ii) `--hold-annotations` STAYS and is IMPLICIT here: his table carries 0 of the 58 gold studies
#        (arm header :29-32), so the flag holds nothing out of training. It is kept because it is what
#        leaves the holdout record beside the checkpoint, without which the arm cannot be priced as a
#        seat at all. The epoch-1 recipe line must still name his path under `labels` and OURS under
#        `eval_labels` - two flags one letter apart, and an arm that trained and scored on one table
#        would print a plausible number for the wrong cell.
#
# MARKER vs FILE: `_arm_log` names the file for the checkpoint (`clean_calib_s2026.full.log`) and
# `_mark` writes the `run_train` name (`ARM DONE clean_calib`). Both spellings below are the ones that
# exist on disk (CLAUDE.md, knee-ket1c).
set -u

cd /d/personal_projects/rsna_knee_kaggle || exit 1

GATE="logs/clean_calib_s2026.full.log"
LOG="logs/after_clean_calib_chain.log"
PY=".venv/Scripts/python.exe"
HOLDOUTS="D:/models/rsna_knee_abnormality"
CHILD="clean_calib_s2026"
PARENT="seat_clean_distill_s2026"
HISRECIPE="experiments/arms/hisrecipe_distill.sh"
ATTACH="inbox/main/attach/hisrecipe_distill.sh"

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
  echo "SOLO GATE HELD: >= +0.005 - researcher runs d34_seat.py clean_calib_s2026; the ENTRY still needs" >> "$LOG"
  echo "  that seat read >= +0.002 out of sample (20 halves, >= 16/20 > 0) before kneemri-cleancalib ships" >> "$LOG"
elif awk -v d="$DELTA" 'BEGIN { exit !(d > 0.001) }'; then
  echo "DIRECTIONAL: +0.001..+0.005 - seat read still runs; the entry ships only if it clears +0.005" >> "$LOG"
else
  echo "CORNER NULL: <= +0.001 - kneemri-cleancalib stays staged and unspent" >> "$LOG"
fi
if awk -v d="$DELTA" 'BEGIN { exit !(d >= 0.0135) }' 2>/dev/null; then
  echo "SEPARATE FINDING: >= +0.0135 - distillation amplifies the knob; ledger it" >> "$LOG"
fi

echo "(a) launching foreign_labels_dense at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
bash experiments/arms/foreign_labels_dense.sh >> "$LOG" 2>&1
echo "(a) foreign_labels_dense returned $? at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

# (b) the researcher's his-recipe port, if it has arrived. It is delivered as a script into the mail
# attachment directory; a letter without a script is not runnable and must not stall the card.
if [ ! -f "$HISRECIPE" ] && [ -f "$ATTACH" ]; then
  cp "$ATTACH" "$HISRECIPE"
  echo "(b) took hisrecipe_distill.sh from the mail attachment" >> "$LOG"
fi
if [ -f "$HISRECIPE" ]; then
  echo "(b) launching hisrecipe_distill at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
  bash "$HISRECIPE" >> "$LOG" 2>&1
  echo "(b) hisrecipe_distill returned $? at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
else
  echo "(b) hisrecipe_distill absent at $(date -u +%Y-%m-%dT%H:%M:%SZ); the twins take its place in line" >> "$LOG"
fi

# (c) the six twins, BEHIND the gate. Six seats retrained under one treatment are the block's w = 2/3,
# so they run only if the treatment moved a solo by an amount the block can carry.
if [ "$DELTA" != "NA" ] && awk -v d="$DELTA" 'BEGIN { exit !(d >= 0.004) }'; then
  echo "(c) twins LICENSED at delta $DELTA >= +0.004; launching retrain_six_distill" >> "$LOG"
  bash experiments/arms/retrain_six_distill.sh >> "$LOG" 2>&1
else
  echo "(c) twins NOT RUN: delta $DELTA < +0.004. Six seats retrained under a treatment that did not" >> "$LOG"
  echo "    move one solo cannot move the block, and the card goes to the next mechanism instead." >> "$LOG"
fi

echo "after_clean_calib_chain finished at $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"
