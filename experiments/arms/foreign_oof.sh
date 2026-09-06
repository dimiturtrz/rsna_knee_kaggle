#!/usr/bin/env bash
# foreign_oof.sh — their five folds over the WHOLE corpus, so every study has a STRONG out-of-sample opinion.
#
# THIS ARM IS AN INSTRUMENT. Nothing here is seated and no score of ours is read from it. What it
# produces is the second instrument the label-correction lever has been missing, at a strength the
# lever actually needs.
#
# WHY IT IS NEEDED, and what went wrong without it. `label_contradiction` asks for cells where the
# report reader AND an out-of-sample model both contradict the training label. `oof_folds` supplied the
# model side for the four fifths of the corpus fold 0 could not speak for — but at 224 px and 8 epochs,
# a deliberately cheap instrument, and trained on the SAME table the reader derived. Run corpus-wide
# today it finds 70 candidate cells of 52872 (0.1%), where the fold-0 census that motivated the whole
# lever read 6.2%. Suspect the instrument before the labels: a model fitted on the reader's own table
# cannot be an independent second opinion about that table — it agrees with it by construction, and the
# 0.1% is a measurement of that circularity, not of the labels.
#
# THEIR FOLDS BREAK THE CIRCLE. mattiaangeli's five checkpoints were fitted on the HOST's labels, by
# somebody else's decode, split by md5(Report_text)[:8] % 5 — so for every one of our 4407 studies there
# is exactly one of their folds that never saw it, and none of the five ever saw our reader's table.
# They are also strong where ours is cheap: 0.8812 fold-matched on the 909, against 0.8695 for the best
# of the four `oof_folds` runs. An independent instrument, out of sample everywhere, at member strength.
#
# THE FALSIFIABLE PART, pre-registered before the run:
#   * The corpus candidate rate lands near the fold-0 census (a few percent of cells, tens of percent of
#     positives in the weak columns) — the 0.1% was the circularity, the labels have the room the lever
#     assumed, and a correction arm is queued on the corrected table.
#   * It lands at the same ~0.1% — the conjunction is genuinely rare and the label route closes for the
#     second time, this time on an instrument that cannot be accused of agreeing with the table it
#     is auditing. That closure is worth the card on its own.
#   * It lands ABOVE ~15% of cells — suspect the join, not the labels: their label order, their study
#     identifiers, or a serving band mismatch. `foreign_price` on the 909 is the control, and it must
#     still read 0.8812.
#
# COST: 4407 studies x five folds. The 909-study serving ran in one pass; this is ~4.8x of it, and it
# is the only card work that unblocks the label axis.
#
# deliberate: --studies train.csv (the WHOLE corpus, not a holdout — that is the point of the arm)
# reading: foreign_oof_corpus.npz
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

FOREIGN_KERNEL="$REPO/external/dinov3folds/bend_the_knee_to_dinov3_ensembled.py"
FOREIGN_WEIGHTS="D:/data/external/kaggle_foreign/mattiaangeli_dinov3"
RAW="$(dirname "$PROCESSED")/../raw/rsna_knee_abnormality"

# A SCAN NEEDS THE OVERWRITE GUARD `run_train` ALREADY HAS. This arm does not train, so nothing on the
# path that refuses a second write to a checkpoint sees it — and a scan relaunched onto a job that had NOT
# failed is exactly the defect `card_check` was written for. It has now happened twice with this arm: two
# pairs of identical processes, same argv, same `--out`, both writing this npz while the other read it.
# Cheapest place to stop it is here, in the arm, before the card is touched.
if [ -e "$MODELS/foreign_oof_corpus.npz" ] && [ -z "$ALLOW_OVERWRITE" ]; then
  echo "REFUSED foreign_oof - '$MODELS/foreign_oof_corpus.npz' already holds a scan. ALLOW_OVERWRITE=1 to replace it." >&2
  exit 1
fi

"$REPO/.venv/Scripts/python.exe" -m kneemri.inference.foreign_folds \
  --kernel "$FOREIGN_KERNEL" \
  --checkpoints "$FOREIGN_WEIGHTS" \
  --series-root "$RAW/train_series" \
  --series "$RAW/train_series.csv" \
  --studies "$RAW/train.csv" \
  --out "$MODELS/foreign_oof_corpus.npz"
