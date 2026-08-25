#!/usr/bin/env bash
# bead: knee-bs4l
# knee-bs4l, the CONTROL half. band_span with --hold-annotations and nothing else changed.
#
# WHY IT EXISTS AND WHY holdannot_s2026 WILL NOT DO. The arm under test is calib_span + --hold-annotations,
# and calib_span IS band_span plus --calibrated-targets. holdannot is a different recipe in three places at
# once — band 0.2-0.8 against 0.1-0.9, --image-size 224 against the default, and no calibration — so a
# calib_span_hold-vs-holdannot comparison would move three knobs and attribute the whole difference to one.
# This is the control that makes the pair single-variable.
#
# BOTH halves hold the annotations, which is what buys the second ruler: only a checkpoint carrying
# hold_annotations=True is readable by `python -m kneemri.evaluation.annotated_readings`, and that reading
# is the point of the pair. The flag itself is priced at -0.0015 on the proxy (knee-9198).
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"
run_train band_span_hold \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --band-low 0.1 --band-high 0.9 --hold-annotations \
  --labels "$LABELS" --out "$PROCESSED/band_span_hold_s2026.pt"
