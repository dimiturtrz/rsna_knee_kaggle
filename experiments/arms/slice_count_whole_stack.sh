#!/usr/bin/env bash
# 12 cached slices -> 16, read whole-stack. The mechanism that already paid, pushed one step further.
#
# WHY THIS IS NOT THE ARM WE ALREADY REFUTED. slice_count_maxpool tested 20 cached slices against 12
# and found nothing (0.8309 vs 0.8310) - but both arms read ONE RANDOM 3-SLICE WINDOW per step. At a
# fixed group of 3, the window covers 3 slices however many are cached, so caching more could only
# change which window is drawable, never what a step sees. The null was structural.
#
# Whole-stack removes exactly that: every cached slice goes through the encoder every step, so the
# cached count IS the anatomy a step sees. 12 slices is 72 images per study, 16 is 96. The refutation
# on record does not reach this case, and the +0.0147 that whole-stack bought is the same mechanism -
# show the model more of the stack - which is the reason to expect this one to move at all.
#
# SUPPORT: the occlusion depth profile spans 0.7976..0.8332 across windows, so depth carries real and
# unevenly-used signal; the 0.20-0.80 band holds ~18 slices on the median study and we keep 12; MRNet
# maxes over ALL slices of a series rather than a fixed count.
#
# WHY 16 AND NOT 20, which is the count already cached and would cost no decode: whole-stack at batch
# 12 with 12 slices holds 19.2 GB. Twenty slices is 1440 images per step, ~33 GB, over the 32 GB card.
# Sixteen is ~1150 images, ~26 GB, and fits. The batch is what is being protected: the ranking term
# scores pairs WITHIN a batch (66 at batch 12, 28 at batch 8), so dropping the batch to afford 20
# slices would change the objective and stop this being a single-variable arm.
#
# control  wsfold0_e6 = 0.8457 - the same recipe, same fold, 12 cached slices.
# cost     ~1.35x the step time of the control; 12 epochs.
#
# PREDICTION, written before the run: if depth coverage is the live axis, this lands above 0.846 and
# the gain shows up broadly rather than in one label - unlike the resolution arm, which predicts a
# gain CONCENTRATED in MCL, ACL and the menisci. The two arms are therefore distinguishable by their
# per-label tables even if their means move alike, and kneemri.evaluation.scores already reports per-label.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
SLICES="${1:-16}"
require "$LABELS"

# Keyed on the mask, which build() writes with a single np.save after the decode loop returns. The
# images .npy is allocated zeroed at build START, so gating on it would train on an empty array and
# return a plausible wrong answer rather than crash.
wait_for_cache 336 "$SLICES"

run_train "slices${SLICES}_ws" \
  $RECIPE --epochs 12 \
  --slice-pooling max --whole-stack --slices "$SLICES" \
  --recompute-activations \
  --labels "$LABELS" --out "$PROCESSED/slices${SLICES}_ws_s2026.pt"
