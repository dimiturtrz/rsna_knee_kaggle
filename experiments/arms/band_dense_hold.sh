#!/usr/bin/env bash
# band_dense_hold.sh — the held control at the DENSE corner, which the repo does not own.
#
# WHY THIS ARM EXISTS. knee-9198 is a standing P0: a label-side arm reports BOTH rulers, and only a
# checkpoint carrying hold_annotations=True is readable by `kneemri.evaluation.annotated_readings` —
# it REFUSES a checkpoint that trained on the 58 annotation cells rather than scoring it. So an unheld
# label-side arm cannot produce the second number at all; it is not that the number is worse, it is
# that the tool declines to compute it.
#
# The held pair exists only at the SPAN corner (`band_span_hold` / `calib_span_hold`, knee-bs4l). At the
# DENSE corner — 20 slices over 0.10-0.90, the recipe of the best single member this project owns — the
# only held control on disk is `holdannot_s2026`, which is the native224 recipe. Reading a dense arm
# against it would move three knobs and attribute the difference to one. This is the control that makes
# the dense pair single-variable, exactly as band_span_hold does for the span pair.
#
# control  band_dense = 0.8578 (336 px, 20 slices, 0.10-0.90, whole-stack, 4e-5, 12 epochs). SINGLE
#          VARIABLE against it: --hold-annotations and nothing else.
# cost     identical to band_dense. One 12-epoch fit.
#
# PREDICTION, written before the run. The flag is priced at -0.0015 on the proxy (measured at native224:
# 0.8502 -> 0.8487). If the dense corner reads within the 0.0062 floor of that, the hold cost is
# recipe-independent and this control is reusable for every future dense label-side arm. A hold cost
# materially larger than -0.0015 here is itself a finding: it would mean the 58 annotation cells carry
# more of the dense recipe's fit than of native224's, and it would re-price knee-9198's flat -0.0015.
#
# REUSE IS THE POINT. This run is not a member and is not seated. It is infrastructure: the dense family
# is where our best member lives, and every label-side question asked there from now on needs this
# checkpoint as its held control. One 40-minute run buys the control for the whole family.
#
# bead: knee-ep74
#
# NO --vram-gb: DeviceBudget derives the cap from --slices — 20 slices reserve 28.26 GB.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band_dense_hold \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --hold-annotations \
  --labels "$LABELS" --out "$PROCESSED/band_dense_hold_s2026.pt"
