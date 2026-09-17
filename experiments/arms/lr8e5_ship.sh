#!/usr/bin/env bash
# bead: knee-e9f83
# reading: lr8e5_ship_s2026.holdout.npz
#
# WHAT THIS ARM IS
#   `labelwise_depth24_drop40_long`'s byte-identical line with `--backbone-lr 4e-5 -> 8e-5`. One flag.
#   The control is that arm's checkpoint, already on disk at 0.9053 ledger75, so no matched baseline
#   is owed and the whole cost of the question is this run.
#
# THE CONSTRAINT THIS ARM EXISTS TO LIFT
#   `--backbone-lr 4e-5` is fixed across all 18 arms at this corner while labels, slices, band, batch,
#   image-size, vram-gb and scheme all vary there. Corpus-wide, every arm carrying any other rate runs
#   TWELVE epochs; at 24 the corpus is 36 arms at 4e-5 against one warm start at 1e-5. The rate is
#   inherited from a 12-epoch schedule with no slot dropout, and this corner has neither property.
#
# WHY 8e-5 AND NOT 1e-4
#   The four pairs that isolate the rate all read higher at about 2x, and the two on OUR DINOv2
#   whole-stack recipe are exactly 2.05x (llrd224, +0.0043) and 2.45x (band_deep_matched, +0.0016).
#   1e-4 is the Swin pairs' value and sits nearer the collapse measured between 1e-4 and 4e-4
#   (mricore 4e-5 0.8658 -> 4e-4 0.8034). 8e-5 is the size our own stack measured.
#
# PRE-REGISTRATION lives in knee-e9f83 and is not restated here. Record the PEAK EPOCH and the
#   peak-to-finish drop: a higher rate peaking EARLIER than the control's e18 at a similar value says
#   the schedule is binding and the follow-up is budget, not rate.
#
# READ PEAK-TO-PEAK. $RECIPE carries --one-cycle and the rate scales the whole schedule, so the two
#   runs differ from epoch 1 and an epoch-by-epoch comparison means nothing.
#
# deliberate: --backbone-lr 8e-5 — the single variable.
# deliberate: --epochs 24 --slot-dropout 0.40 --slices 24 --batch 10 --vram-gb 28 --band-low 0.1
#             --band-high 0.9 --calibrated-targets --hold-annotations — inherited verbatim from the
#             control. This arm changes one flag.
#
# CARD. --vram-gb 28 of 32, inherited; nothing else of ours may share the card (`card_check`).
#
# control  labelwise_depth24_drop40_long_s2026, 0.9053 ledger75 @909, peak e18. No control run is owed.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train lr8e5_ship \
  $RECIPE --epochs 24 --slice-pooling labelwise --whole-stack $THROUGHPUT \
  --backbone-lr 8e-5 --slices 24 --batch 10 --vram-gb 28 --band-low 0.1 --band-high 0.9 \
  --calibrated-targets --hold-annotations --slot-dropout 0.40 \
  --labels "$LABELS" --out "$MODELS/lr8e5_ship_s2026.pt"
