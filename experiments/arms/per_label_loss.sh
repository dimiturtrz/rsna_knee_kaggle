#!/usr/bin/env bash
# per_label_loss.sh — make the loss weigh all twelve findings equally, because the METRIC does.
#
# THE MISMATCH, measured over all 4,407 studies: Effusion is positive in 2,666 of them and Fracture in
# 314. Pooled cross-entropy therefore spends about ten times more gradient on one than the other, while
# the competition score is the UNWEIGHTED mean of twelve per-label AUCs and counts them the same.
# Normalising the loss per column removes the mismatch exactly: each label contributes one twelfth,
# whatever its prevalence and whatever the report parser's confidence in it.
#
# THIS IS NOT THE PER-LABEL TARGETING THAT WAS REFUTED. That plan (knee-1dw) proposed aiming at MCL and
# Fracture BECAUSE they scored worst, and died when their weakness turned out to be a fold artefact —
# Fracture moves +0.119 between folds. This changes no target and knows nothing about which labels are
# hard; it removes prevalence from the objective, which is a property of the DATA, not of a score.
#
# WHY IT MIGHT DO NOTHING, stated first: AUC is rank-based and invariant to the positive rate, so
# balancing helps a ranking metric far less than it helps a thresholded one. The mechanism that could
# still pay is gradient SHARE — a label the objective barely looks at is a label whose ordering the
# encoder has little reason to learn. That is plausible and it is not guaranteed.
#
# THE FALSIFIABLE PART: if this works, the gain concentrates in the RARE labels (Fracture 314 positives,
# Synovitis 580, MCL 767) and the common ones (Effusion 2,666, Medial Meniscus 2,347) hold or dip
# slightly. A uniform shift means something else changed and the mechanism is not what moved it.
#
# control  wsfold0_e6 = 0.8457, same recipe, pooled loss.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train per_label_loss \
  $RECIPE --epochs 12 \
  --slice-pooling max --whole-stack --per-label-loss \
  $THROUGHPUT --vram-gb 26 \
  --labels "$LABELS" --out "$PROCESSED/per_label_loss_s2026.pt"
