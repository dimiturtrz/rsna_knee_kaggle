#!/usr/bin/env bash
# bead: knee-5jfix
# reading: labelwise_ms_s2026.holdout.npz
#
# Multiscale slots under labelwise pooling. `multiscale` was gated out under max pooling (crop80_tissue,
# +0.0016); max keeps one slot per study, labelwise keeps one per label, so scale-specific evidence
# (fine for menisci/ACL, coarse for effusion/OA) is the thing max could not use and labelwise can.
#
# PRE-REGISTRATION (delta = this arm minus labelwise_ms_ctrl_s2026, ledger75)
#   >= +0.0150   lever: port to the 24-slice shipping line, then a slot.
#   +0.0062..+0.0149  directional; the tissue centre is confounded in (alone it read -0.0048).
#   <= +0.0061   multiscale closed on labelwise too.
#
# deliberate: --scheme multiscale --centre tissue — the single change vs labelwise_ms_ctrl; the cache
#   only exists tissue-centred.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train labelwise_ms \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 12 --batch 10 --vram-gb 22 --band-low 0.2 --band-high 0.8 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --scheme multiscale --centre tissue \
  --labels "$LABELS" --out "$MODELS/labelwise_ms_s2026.pt"
