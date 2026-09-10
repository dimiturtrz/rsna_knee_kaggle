#!/usr/bin/env bash
# oof_reader_folds.sh — the v5reader table across all five folds, so the gold ruler is 58 studies wide.
#
# THIS IS D296 §2 SPENT ON THE ONE QUESTION THAT IS WORTH IT.
#
# WHAT D296 ESTABLISHED. The five fold holdouts PARTITION the corpus (909+875+884+849+890 = 4407,
# disjoint, union the whole thing) and the 58 radiologist-annotated studies fall 13/8/16/11/10 across
# them. So a recipe run on all five folds yields 4407 studies each carrying one prediction from a
# member that never saw it, and the gold read over that stitch is 58 studies — every one of them
# honest. `oof_s2026` is already such a family and reads gold@58 macro 0.8927. The price of an honest
# gold read for a NEW table is five arms instead of one.
#
# WHY THE CONTROL IS FREE, AND WHY THAT IS THE WHOLE ARGUMENT. `oof_folds.sh` and `oof_fold0.sh` built
# `oof_fold[0-4]_s2026` at 224 px / 8 epochs / slices 12 / band 0.2-0.8 / backbone-lr 4e-5 on
# `$LABELS` = report_labels_v2. This arm is that geometry to the token with `--labels` moved to
# v5reader and nothing else touched. So the paired comparison costs FIVE arms, not ten — the v2 side
# of it was paid for weeks ago on a different question. That is the only reason the ratio works.
#
# WHY NOT THE CHEAPER ARM. `oof_clean_s2026` is "one fold-0 arm away" from a second family in D296's
# words, and it is 25 minutes rather than two hours — but its fold-0 block would BE `oof_fold0_s2026`
# (README:9117 already stitches it that way), so the pairing against `oof_s2026` is degenerate on
# exactly the 13 fold-0 gold studies and the effective paired n falls to 45. Worse, the question that
# pairing answers — what training on fold 0 does to a teacher — is already measured at -0.0067
# (`compose_v4b_clean.sh:47`). Twenty-five minutes of card for a degenerate ruler on a settled
# question is a worse ratio than two hours for a full ruler on an open one.
#
# THE QUESTION. `reader_through` is on the card now asking whether the new corpus reader's table beats
# v2 as SUPERVISION, and its pre-registration (corrected mid-run, 2026-09-10T11:10Z) had to concede
# that its gold branch cannot read: n=58 single-fold is 78% in-sample because `--hold-annotations` was
# never passed and does not do what its name says anyway, and the honest 13 inside its holdout carry
# unpaired sd 0.034. This arm is what makes that question answerable. It is the SAME moving token at a
# weaker geometry, and what it buys is not a better member — it is the only ruler in this project that
# can resolve a label-table difference at the scale a label-table difference lives at.
#
# THE PRICE, SAID OUT LOUD. Five runs at ~25 min = ~2.1 h of card. Paired gold difference sd is 0.0101
# (D295 §5, at the measured arm-to-arm correlation 0.84), so this instrument resolves ~0.020 at two sd.
# The mechanism it is pointed at is D283's +0.0608 board-minus-offline gap, which `cnnsolo_v1` today
# confirmed out of sample on a family the board had never held: our ruler reads members LOW because
# its reader invents 108 confident false positives in 696 cells against radiologists. Halving that is
# a supervision change touching every member this project will ever train. P(the paired gold clears
# +0.020) ~ 0.25 by my read of the 121-vs-140 cell count; the ratio is 2.1 h against a lever four
# times the significance bar, and no cheaper instrument exists that can see it.
#
# WHAT IT IS NOT. Not a member family. These five checkpoints are 224/8 instruments and are not roster
# candidates, exactly as `oof_folds.sh` says of its own. Not the banned fold sweep either: no score of
# any single fold here is read, and the five hold out disjoint fifths by construction rather than
# repeating one experiment for a second answer. The deliverable is the stitch.
#
# PRE-REGISTERED, before anything runs, on the PAIRED gold@58 against `oof_s2026`'s 0.8927:
#   >= +0.020   THE TABLE IS BETTER against radiologists at the only resolution that can say so.
#               v5reader becomes the supervision table and every label-side seat is re-fit on it.
#   +/-0.020    NULL AT THIS RESOLUTION. Reader quality is not worth more card at this magnitude;
#               keep `reader_through` as a diversity member and stop paying for reader tables.
#   <= -0.020   THE 74 MISSES COST REAL SIGNAL. The under-call is the defect, not the over-call, and
#               the next reader fills silence rather than suppressing calls (Synovitis, Effusion).
#   The corpus-wide stitch (n=4407, v2 truth) is reported BESIDE it and is NOT a selector: it is
#   scored against the very table this arm is arguing with, so it can only measure agreement with v2.
#
# deliberate: --labels (the single moving token vs oof_folds.sh / oof_fold0.sh)
# deliberate: --image-size 224 (an instrument, not a member — the control was built at 224 too)
# bead: knee-07gvn
# reading: oof_reader_fold*_s2026.holdout.npz
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

READER="$PROCESSED/labels/report_labels_v5reader.csv"
require "$READER"
require "$LABELS"

for fold in 0 1 2 3 4; do
  run_train "oof_reader_fold${fold}" \
    $RECIPE --fold "$fold" --epochs 8 --slice-pooling max --whole-stack $THROUGHPUT \
    --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 --image-size 224 \
    --labels "$READER" --eval-labels "$LABELS" --out "$MODELS/oof_reader_fold${fold}_s2026.pt"
done
