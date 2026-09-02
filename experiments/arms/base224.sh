#!/usr/bin/env bash
# base224.sh — the bigger encoder at ITS OWN pretraining resolution, with the batch handicap removed.
#
# bead: knee-j798
#
# It stands on knee-w84q (the 224 finding) and serves knee-uj15 (the single-model frontier).
#
# WHY THIS ARM. `base_dense` is the only measurement of dinov2-base at the settled operating point and
# it read 0.8512 against band_dense's 0.8578 — minus 0.0066, OUTSIDE the floor — which the champion's
# principles record as an ENCODER reading. That reading carried two handicaps its own header names:
# `--batch 6`, forced because base at 336 px and 20 slices needs 56.3 GB on a 32 GB card, and small's
# learning rate, deliberately not rescaled so the encoder would not be confounded with its schedule.
# `batch_teardown` later measured the batch at +0.0020 on small and concluded the batch was not the
# lever at that operating point — but it measured it on SMALL, at 336, where 12 is affordable. It did
# not measure base at 6.
#
# WHAT MOVED THAT MAKES THIS WORTH RE-ASKING. `native224` read 0.8502 against 0.8525 at 336 — flat
# inside the floor, at 94 s/epoch instead of ~290 and 8.1 GB instead of 18.0. DINOv2 was pretrained at
# 224; we had been interpolating its position embeddings to 336 for nothing measurable. At 224 with 12
# slices the memory that forced base down to batch 6 is roughly halved twice over — 0.44 the patches and
# 0.6 the slices — so base runs at the CONTROL'S OWN BATCH of 12 for the first time.
#
# THE MECHANISM, stated so it can fail. Base is 768-wide against small's 384. The recipe freezes blocks
# 0-5 and fits 6-11, so what doubles is the width of the representation the head reads and the capacity
# of the half that adapts. Our worst finding is Lateral Meniscus at 0.7981 — a focal, small-structure
# call — and a wider representation is the one change that plausibly buys small structure without
# touching sampling geometry that eleven arms already showed is not the constraint. Nothing here is a
# knob: this is the architecture axis, which is the only axis left whose plausible size clears the bar.
#
# ONE VARIABLE against the control: `--variant base`. The control is `native224` = 0.8502, which is
# byte-identical to this except for the encoder. The rate is again NOT rescaled, for the same reason as
# base_dense: rescaling confounds encoder with schedule and there is still no isolated base-appropriate
# rate to borrow. If this reads flat-to-down, the rate is the first suspect and the follow-on is the
# rate, not the abandonment of the encoder.
#
# COST — SMOKE IT BEFORE QUEUEING. base's arithmetic per image is roughly 3-4x small's, so the honest
# range is 300-400 s/epoch and 12 epochs is 60-80 minutes. That is an estimate and not a measurement;
# `python -m kneemri.experiments.arm_check --smoke` has already written the truncated 224 cache, so
# `--studies 60 --epochs 1` prices the epoch and the peak memory in under two minutes. Do that first.
# Derived memory: base_dense peaked 27.14 GB at 336/20sl/batch 6; this is 0.44 the patches, 0.6 the
# slices and 2x the batch, so about 14.5 GB — comfortably inside 32 GB, and the smoke is what confirms it.
#
# THE FALSIFIABLE PART, written before the run.
#   * >= +0.015 over 0.8502. The encoder is the axis, base_dense's minus 0.0066 was the handicaps and
#     not the encoder, and the next arms are the base rate sweep and base under band_dense's geometry.
#   * +0.006 to +0.015. Directional. Base earns a seat on the PIXEL-and-WIDTH axis regardless — it would
#     be the only 768-wide member in a pool seven small members deep, which is the error-structure
#     argument that bought base_dense its seat in the champion despite losing its own control.
#   * within +/- 0.006. Base at small's rate ties small. That is not a refutation of the encoder; it
#     retires the BATCH suspect that base_dense's header left open and hands the question to the rate.
#   * <= -0.015, or a collapse below 0.83. The larger encoder is being mis-fit rather than being worse,
#     and the suspects in order are the rate and `--unfreeze-last 6`, a constant chosen against small's
#     width and never re-derived for twice it.
#
# HARNESS CHECK. The printed recipe must name "variant": "base" and "image_size": "224". A run that
# silently fell back to small would read as a flat result and close the axis on a null that never fired.
#
# control  native224 = 0.8502 (best epoch 7 of 12, converged)
#
# deliberate: --variant base (the arm's thesis)
# deliberate: --image-size 224 (inherited from the control, which IS the 224 read)
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train base224 \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant base --backbone-lr 4e-5 --slices 12 --band-low 0.2 --band-high 0.8 --image-size 224 \
  --labels "$LABELS" --out "$MODELS/base224_s2026.pt"
