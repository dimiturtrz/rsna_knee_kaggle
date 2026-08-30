#!/usr/bin/env bash
# swin_toprate.sh — the top is open, so now it is trained at a rate the top can take.
#
# WHY THIS ARM EXISTS, and it is written from a curve rather than from a hope. `swin_supervised` ran with
# stage 3 FROZEN (the knee-94ec discovery defect) and peaked 0.8331. `swin_wholetop` re-ran it
# byte-identical on the fixed discovery, which doubled trainable parameters to 37,869,696 and opened both
# 1024-wide stage-3 blocks — and it did not merely fail to gain, it drifted AWAY from the handicapped run:
#
#   e0 +0.0003   e1 +0.0013   e2 -0.0001   e3 -0.0028   e4 -0.0041
#
# Monotone since e1. Twice the capacity, and the extra capacity is costing. That is not a capacity result;
# it is a RATE result, and it is the mechanism `swin_wholetop.sh` pre-registered under REFUTES before it
# ran: "the 1e-4 rate is too high for stage 3's wider blocks and the follow-up is a rate arm, not another
# family."
#
# WHY THE RATE AND NOT `--layer-decay`. Layer decay was the obvious reach and it is the wrong tool here.
# `EncoderSurgery.backbone_groups` scales a group by `backbone_lr * layer_decay ** (len(opened) - 1 -
# depth)`, so the LAST opened block carries exponent zero and keeps the full rate. Decay lowers the rate of
# the blocks BELOW the top, which are the 512-wide stage-2 blocks that were already training fine at 1e-4.
# It cannot touch the two blocks this arm is about. One flag moves.
#
# WHAT THE THREE POINTS BOUND. The trunk has now been read at two settings of its top: frozen (0.8331) and
# open at 1e-4 (tracking ~0.004 below it). Those bracket the question rather than answer it — "open at a
# lower rate" is the interpolation neither has measured, and it is the only setting under which the extra
# capacity could pay. 3e-5 is chosen as a third of the control's, which is also the order `cnx_dense`
# inherited as the project's ViT adapter rate (4e-5) — so it is not a fresh guess, it is the rate this
# harness used everywhere before 1e-4 was picked for a frozen-top Swin that never trained its widest blocks.
#
# control  swin_supervised = 0.8331 (converged, peak e6, e7-11 declining — not a truncation floor).
#          swin_wholetop = the same recipe at 1e-4 with the top open; its finished peak is the second
#          control and this arm may NOT be queued before that number exists, because a rate arm against
#          a curve read at e4 is fitting the follow-up to a partial run.
# cost     ~3 h, priced off swin_wholetop's own epochs (894-923 s after the first). Identical memory:
#          the same blocks train, only their rate changes.
# seat     `seat_value --candidates ... --against greedyseat_v1`, card-free, after.
#
# PRE-REGISTERED, against 0.8331 and the 0.0062 floor.
#   LANDS — 0.8560 or above. The rate was the whole story and a supervised trunk sits in the dense band.
#     Roster, mount, entry, slot.
#   SEAT — 0.8393 to 0.8559. A real gain over the frozen-top control that does not reach the dense band.
#     Price the seat; mount only at +0.002 or better offline.
#   COSTS — 0.8269 to 0.8392. Inside a floor of the frozen-top control. Then the top of this trunk does
#     not want to be trained on this corpus at any rate the arm has tried, and the finding is that
#     `swin_supervised`'s 0.8331 was not a handicap at all — it was the right amount of trunk to move.
#     Do NOT run a third rate. The ConvNeXt re-open (knee-uk4f) dies with it, since it is the same
#     correction on the same flag.
#   REFUTES — below 0.8269. Worse than both settings of the top, which would say the deficit is not in the
#     rate at all and the trunk is being asked the wrong question by this recipe. Then the axis to move is
#     the recipe (band, slice pooling, read size), not the optimiser, and that is a different arm.
#
# THE REFUTATION THIS CANNOT MAKE: it cannot separate "3e-5 is right" from "less trunk movement is right".
# A flat read here and a flat read at COSTS look the same from the number alone, and only the frozen-top
# control distinguishes them — which is why 0.8331 and not 0.8460 is the bar every branch above is set
# against.
#
# bead: knee-tild
# The arm it follows is knee-tsiz; the defect that produced both is knee-94ec.
#
# deliberate: --backbone-lr
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train swin_toprate \
  $RECIPE --epochs 12 --batch 1 --accumulate 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant swinbase --backbone-lr 3e-5 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 384 \
  --vram-gb 8 \
  --labels "$LABELS" --out "$PROCESSED/swin_toprate_s2026.pt"
