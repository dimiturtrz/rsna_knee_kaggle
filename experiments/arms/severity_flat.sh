#!/usr/bin/env bash
# Is the severity gradation we already have worth anything? Collapse it and see what it costs.
#
# report_labels_v2 carries three YES levels (0.68 / 0.82 / 0.94, 9,702 pairs). The flat table collapses
# all three to 0.82 - the middle, so the average pull on a YES pair is unchanged - and changes nothing
# else. The EVAL table is held at the graded original so the ruler cannot move under the arm.
#
# Ranking is on because it is the only component that consumes the gradation (pair weight is
# relu(s_i - s_j)); BCE pulls both toward present and the local evaluation binarises at 0.5.
#
# control  rank24_s2026 (graded)   0.8272
# P  flat YES                      0.8228   -0.0044
#
# So the gradation is live signal the model can use, not assertion-strength it ignores. That is the
# gate a reader emitting true ordinal severity was waiting on, and it passed.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
FLAT="$PROCESSED/labels/flatyes_v2.csv"
require "$LABELS" "$FLAT"

run_train severity_flat \
  $RECIPE --epochs 24 \
  --labels "$FLAT" --eval-labels "$LABELS" --out "$MODELS/flatyes_s2026.pt"
