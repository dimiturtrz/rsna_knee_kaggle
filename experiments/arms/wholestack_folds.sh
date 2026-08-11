#!/usr/bin/env bash
# The submission candidate: the winning recipe across folds, for a rank-averaged ensemble.
#
# WHY THIS RECIPE. Whole-stack reading measured 0.8413 on fold 0 against a 0.8310 control (+0.0103,
# the largest single change this project has measured) and it converges by epoch 6-9, so the budget is
# 12 rather than 24. It carries the two other survivors with it: encoder rate 4e-5 (+0.0051 isolated)
# and max slice pooling (+0.0038 isolated).
#
# WHY FOLDS RATHER THAN SEEDS. Our 0.888 leaderboard entry was a five-fold ensemble whose members
# averaged 0.8269 on their own holdouts. Folds also cover the corpus between them, which a bag of
# fold-0 members cannot: every member here trains on a different 3,498 studies.
#
# COST ~58 min per fold. Folds are run one at a time because the card is already compute-saturated at
# 96% of its power limit - a second concurrent arm would split throughput, not add it.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
# An arm asked for no folds has nothing to do, and saying so beats exiting 0 with an empty log.
[ "$#" -gt 0 ] || { echo "ARM FAILED wholestack_folds (no folds given)"; exit 1; }

for fold in "$@"; do
  run_train "wholestack_fold${fold}" \
    --fold "$fold" --epochs 12 --one-cycle --weight-floor --rank-weight 1.0 \
    --slice-pooling max --whole-stack --recompute-activations --vram-gb 20 \
    --labels "$LABELS" --out "$PROCESSED/wholestack_fold${fold}.pt"
done
