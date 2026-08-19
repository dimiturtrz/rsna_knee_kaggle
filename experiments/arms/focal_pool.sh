#!/usr/bin/env bash
# focal_pool.sh — keep each channel's upper tail alongside the patch mean, for findings that occupy a
# small part of the field.
#
# THE ARITHMETIC THIS RESTS ON. At 336 px the patch grid is 24x24 = 576 tokens. A meniscal tear is 1-3 mm
# and lands in a handful of them, so a plain mean over the grid divides its response by roughly two
# orders of magnitude before the head ever sees it. `cls_mean_focal` concatenates a third part — the mean
# of the top eighth of each channel over the grid — which keeps that tail instead of averaging it away.
# It is the same argument the resolution choice was made on, applied to the pooling rather than the pixels.
#
# PER CHANNEL, NOT PER PATCH, which is the part that makes it more than a max-pool: selecting whole
# patches would commit every channel to one location, and the twelve findings do not share a location.
#
# AN ABLATION, NOT A COPY. Every member on disk and every published member uses plain `cls_mean`, so this
# axis is unvaried for us AND for them. `StudyModel._pool` has implemented it since the head was written
# and nothing has ever run it.
#
# WHY IT MIGHT DO NOTHING, stated first: the head is a linear read over the pooled vector, so a third
# part widens its input by 50% on the same number of studies, and the tail statistic is noisier than the
# mean it sits beside. If the findings are already carried by the class token this buys variance and
# nothing else.
#
# THE FALSIFIABLE PART: if this works, the gain concentrates in the SMALL focal findings — meniscus, ACL,
# MCL, Fracture — and the diffuse ones (Effusion, the three OA columns) hold or dip. Those are the
# structure-scale groups the ledger already reads per-label results against. A uniform lift means the
# extra width helped, not the tail, and the mechanism is not what moved it.
#
# control  wsfold0_e6 = 0.8457, identical recipe pooling `cls_mean`.
# cost     one extra dim-wide part in the head's input; no extra forward, no extra epoch time.
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train focal_pool \
  $RECIPE --epochs 12 --patience 3 \
  --slice-pooling max --whole-stack --pool cls_mean_focal \
  $THROUGHPUT \
  --labels "$LABELS" --out "$PROCESSED/focal_pool_s2026.pt"
