#!/usr/bin/env bash
# Supervise the slot embedding with the whole report, not just twelve numbers - REFUTED.
#
# The twelve labels are a lossy read of a free-text report; the report itself has hundreds of tokens.
# The arm adds a cosine term pulling a study-level readout toward a sentence-transformer embedding of
# the report. Two weights, to separate 'the idea is wrong' from 'the weight is wrong'.
#
# control  rank24_s2026            0.8272
# N  report term at 0.3            0.8219   -0.0053
# O  report term at 1.0            0.8209   -0.0063
#
# MONOTONE in the weight, which is what makes this a kill rather than a tuning miss. The report
# describes the whole study - technique, history, incidental findings - so the auxiliary target pulls
# capacity toward text the metric never scores. Kept opt-in and default off.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
VECTORS="$PROCESSED/labels/report_embeddings.npz"
require "$LABELS" "$VECTORS"

for weight in 0.3 1.0; do
  run_train "report_${weight}" \
    --fold 0 --epochs 24 --one-cycle --weight-floor --rank-weight 1.0 \
    --report-weight "$weight" --report-vectors "$VECTORS" \
    --labels "$LABELS" --out "$PROCESSED/report${weight}_s2026.pt"
done
