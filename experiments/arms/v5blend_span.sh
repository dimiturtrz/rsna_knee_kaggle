#!/usr/bin/env bash
# The label-table arm, re-pointed by D346 from v3 to the BLEND. Same recipe as `band_span` in every
# other respect; the only substitution is `--labels`.
#
# WHY THIS REPLACED `v3_labels_span` TWENTY MINUTES BEFORE IT WOULD HAVE LAUNCHED. D345 reported v3 and
# v5reader as statistically indistinguishable on the blind adjudication, which is what kept v3 queued.
# D346 retracted that: the researcher's merge had concatenated chunks whose answer column is spelled
# `verdict` (chunks 1-2) against chunks spelled `answer` (chunks 3+), producing NaN for 47+ rows that
# the masks then counted as WRONG. The clean rates on 235 verdicts (216 over-call / 19 under-call):
#
#   table       over-call right (n=216)   under-call right (n=19)
#   v2          0.051                     0.737
#   v3          0.685                     0.737
#   v5reader    0.810                     0.737
#
# McNemar on the discordant pairs: v5-only-right 43 vs v3-only-right 16, chi2 11.46, p=7.1e-04. That is
# not a coin flip, and it is the same axis the arm exists to test, so the arm takes the better table.
#
# WHY A BLEND AND NOT v5reader NEAT. The v5-over-v3 win is a TEAR-COLUMN result. On the 57 gold studies
# (34 effusion positives) the Effusion column runs the other way: v3 AUC 0.8299, v5reader 0.7449. So
# `report_labels_v5blend.csv` is v5reader everywhere with v3's Effusion triple substituted in. Nothing
# else in the table moves, and the construction is supported by exactly those two measurements.
#
# WHAT THE GOLD DID NOT CONFIRM. The Effusion criterion-line reading ("a moderate or large amount of
# fluid distending the joint") is applied literally by v3, which trades 16 false positives for 13 false
# negatives at IDENTICAL AUC against the host's own annotators. The host labels small effusions positive
# anyway - 34 of 57 - so the 1,599 flipped cells are correct-by-criterion and wrong-by-host-label. The
# deletion is priced as a recall-for-precision swap, not vindicated. That is why Effusion is the one
# column this arm does not let the newer reader touch.
#
# THE RULER, AND THE TRAP IN IT. `--eval-labels` stays pinned to v2 ON PURPOSE, exactly as in
# `v3_labels_span`: it keeps the holdout truth identical to `band_span`'s so the ledger number is
# comparable, and it makes the eval-on-own-generator loop explicit. A blend-trained member scored by a
# table whose errors it was trained not to reproduce reads a FLOOR by construction. The read is the
# corrected T1 rescore off this run's holdout npz.
#
# control  band_span 0.8565 midpoint / 0.8592 T1, same recipe, same fold, v2 labels.
# cost     12 epochs, ~70 min on the 5090. No decode, no new artefact.
#
# PREDICTION, written before the run, unchanged from `v3_labels_span` except that the table is stronger.
# On the v2 ruler this reads AT OR BELOW band_span - the ruler rewards reproducing the 12.1% it was
# built from. On the corrected T1 ruler, T1 >= 0.8592 + 0.015 makes the lexicon table the single largest
# measured defect in the project and re-opens every member in the pool as mis-trained. +0.005 to +0.015
# is directional and licenses a re-fit of the top seats. At or below zero says the extraction layer is
# not the ceiling and D330's report-silence bias underneath it is.
#
# The refutation I am most alert to is PLUMBING: assert the recipe's `labels` field names v5blend and
# its `eval_labels` names v2 before any number here is written down.
#
# reading: D:/data/volumetric/mri/models/rsna_knee_abnormality/v5blend_span_s2026*.holdout.npz
# bead: knee-clzzs
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

BLEND="$PROCESSED/labels/report_labels_v5blend.csv"
require "$BLEND"
require "$LABELS"

run_train v5blend_span \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 \
  --labels "$BLEND" --eval-labels "$LABELS" --out "$MODELS/v5blend_span_s2026.pt"
