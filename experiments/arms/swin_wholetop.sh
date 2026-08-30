#!/usr/bin/env bash
# swin_wholetop.sh — the same arm as swin_supervised, on a trunk whose top is no longer frozen.
#
# WHY THIS ARM EXISTS. `swin_supervised` read 0.8331 and that number is a fact about a handicapped run.
# `EncoderSurgery.blocks()` took the longest `ModuleList` in the tree as the encoder's blocks, which is
# right for a flat trunk and wrong for a staged one: Swin holds its 24 `SwinLayer` as 2/2/18/2, so the
# longest list was stage 2's eighteen. `--unfreeze-last 6` therefore opened `layers.2.blocks.12-17` and
# `layers.3` — the final two blocks, 1024 wide, the highest-level features in the network — stayed FROZEN
# for the entire run, along with stages 0 and 1. knee-94ec fixed discovery to group by element type, so
# the same flag now opens `layers.2.blocks.14-17` plus both of stage 3.
#
# THE SINGLE VARIABLE IS WHICH SIX BLOCKS TRAIN, and it is a code change rather than a flag. Every line
# of the invocation below is byte-identical to `swin_supervised.sh` — rate, size, batch, accumulate,
# epochs, band, pooling, vram cap. So the comparison against 0.8331 is clean, and it is the only
# comparison this arm makes.
#
# WHY IT IS WORTH THE CARD RATHER THAN A CHEAPER QUESTION. Swin's nearest-neighbour correlation to any
# seated member is 0.8346 against a seated mean of 0.8772-0.9225 — by a distance, the most decorrelated
# member this project has produced. Its seat value was -0.0012 anyway, because a 0.020 solo deficit is
# more than maximal diversity can pay for. The deficit is what this arm attacks, on the read that the
# handicap plausibly accounts for a good part of it: what was frozen was not six arbitrary blocks but the
# whole top of the network, which is the part a fine-tune moves most.
#
# AND IT IS THE ARM THE GOAL ASKS FOR. A non-DINO trunk with a seat is the stated objective, and this is
# the cheapest live path to one: the alternative is a family nobody has fitted, at a full recipe search,
# where this is one re-run of a recipe already smoked.
#
# control  swin_supervised = 0.8331 (best epoch 6; epochs 7-11 declined, so it converged and overfit
#          rather than being truncated). Its seat value against greedyseat_v1 was -0.0012, band
#          [-0.0020, -0.0003], best possible swap -0.0013.
# cost     ~3 h, priced off swin_supervised's own wall-clock at the identical recipe. Stage 3 is 1024
#          wide against stage 2's 512, so the opened parameter count roughly doubles; activation memory
#          is unchanged because those blocks ran in the forward pass either way.
# seat     priced with `seat_value --candidates ... --against greedyseat_v1`, never against
#          raptorweight_v1, five of whose six members carry no holdout row.
#
# PRE-REGISTERED, against 0.8331 and the 0.0062 floor.
#   LANDS — 0.8560 or above. The handicap was the whole story, and a supervised trunk sits in the dense
#     band. Mirror to Kaggle (Apache-2.0), mount, build an entry, spend a slot.
#   SEAT — 0.8393 to 0.8559. A real gain (a floor or more) that does not reach the dense band. Price the
#     seat; mount only at +0.002 or better offline, which transfers roughly one-for-one.
#   COSTS — 0.8269 to 0.8392. Inside a floor of the control either way. The handicap did not account for
#     the deficit, and the honest reading becomes that this trunk at this recipe is not competitive here.
#     Do NOT re-run the ConvNeXt pair on the same hope; re-price that separately.
#   REFUTES — below 0.8269. Opening the top made it worse, which would say the 1e-4 rate is too high for
#     stage 3's wider blocks and the follow-up is a rate arm, not another family.
#
# bead: knee-tsiz
# The defect this arm answers is knee-94ec; the original arm is knee-ryzy.
#
# deliberate: every flag — see above, they are the control's and may not be touched in this arm.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"

require "$LABELS"

run_train swin_wholetop \
  $RECIPE --epochs 12 --batch 1 --accumulate 6 --slice-pooling max --whole-stack $THROUGHPUT \
  --variant swinbase --backbone-lr 1e-4 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 384 \
  --vram-gb 8 \
  --labels "$LABELS" --out "$PROCESSED/swin_wholetop_s2026.pt"
