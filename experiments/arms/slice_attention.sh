#!/usr/bin/env bash
# Pool a slot's slices by learned attention instead of by their maximum.
#
# PRIOR ART: the post-MRNet knee-MRI literature replaces exactly this max with exactly this attention.
# Medical Slice Transformer (Sci Rep 2025) does it on DINOv2, the backbone we already run.
#
# WHY NOW: whole-stack reading is our largest change (+0.0147) and it works by giving a step more of
# the stack - then aggregates it with an elementwise max, which keeps whichever slice shouted loudest
# per feature and discards which slice it was. The occlusion depth profile spans 0.7976..0.8332 across
# windows, so which slice matters, and max throws that away.
#
# control  tonight's whole-stack fold 0 = 0.8457. Single variable: the aggregation.
# risk     it adds parameters to a supervision-limited model, and supplying correct slot attention via
#          the anatomical prior already measured NEUTRAL. The gate is one 64-wide layer with a single
#          shared query, and its output layer is zero-initialised so training STARTS at the mean and
#          has to earn any deviation.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train slice_attention \
  --fold 0 --epochs 12 --one-cycle --weight-floor --rank-weight 1.0 \
  --slice-pooling attend --whole-stack --recompute-activations --vram-gb 20 \
  --labels "$LABELS" --out "$PROCESSED/sliceattn_s2026.pt"
