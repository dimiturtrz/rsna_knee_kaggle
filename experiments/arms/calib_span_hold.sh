#!/usr/bin/env bash
# bead: knee-bs4l
# knee-bs4l, the ARM. calib_span with --hold-annotations, against band_span_hold and nothing else moved.
#
# THE FIRST ARM HERE WHERE A NEGATIVE PROXY NUMBER WOULD BE EVIDENCE FOR THE CHANGE. calib_span measured
# -0.0020 against band_span and that closed the lever, killing calib_dense (knee-ep74) unlaunched. It was
# scored on the report-derived proxy. The board's metric is a per-label RANK AUC, and report labels and the
# radiologists' annotations rank the same studies differently — holdannot_s2026 reads 0.8487 against reports
# and 0.8932 against the annotation cells on the SAME weights. So a change that moves a model toward
# annotation truth mechanically LOSES proxy AUC, bounded by that same +0.045. The calibration replaces each
# reader level with the rate the radiologist actually agreed at: moving away from the report is its entire
# purpose, and a small negative on the report ruler is what a correct move is supposed to look like.
#
# PRE-REGISTRATION, both rulers, against band_span_hold:
#   calibration real  -> proxy stays negative or flat AND the annotated_readings aggregate RISES.
#   calibration dead  -> both fall. The lever stays closed, on the objective ruler this time.
#   proxy rises too   -> the -0.0020 was noise, and knee-p9oc's pooling account is what was wrong.
#
# WHAT THIS DOES NOT ASSUME. knee-p9oc found the remap's constants are POOLED across findings that do not
# agree (the 0.68 level assigned 0.378 while the twelve findings run 0.000 to 0.778) — a defect in the
# ESTIMATE that no ruler change repairs. And the per-label pattern was too structured for ruler bias alone
# (Effusion -0.0212* against PF OA +0.0170*). Ruler bias is a partial account of the mean, not a
# replacement for the mechanism. This arm asks whether the lever was ever MEASURED, not whether it works.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train calib_span_hold \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 --calibrated-targets --hold-annotations \
  --labels "$LABELS" --out "$MODELS/calib_span_hold_s2026.pt"
