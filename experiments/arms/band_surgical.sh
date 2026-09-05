#!/usr/bin/env bash
set -euo pipefail
#
# band_surgical - the SAME SIX BLOCKS, opened somewhere else. Two cells, one variable: WHERE.
#
# WHAT THIS CHANGES. Both runs are `band_dense` with one flag added: `--unfreeze-offset`. Same trunk,
# same 20 slices, same 0.1-0.9 band, same max slice pooling, same whole stack, same 4e-5 backbone rate,
# same 12 epochs, same six open blocks. `band_dense` opens blocks 6-11 (the default suffix). `band_early`
# opens 0-5. `band_mid` opens 3-8. Nothing else moves, so the pair reads one thing: does it matter which
# six blocks of a DINOv2 trunk carry the gradient.
#
# WHY THIS HAS NEVER BEEN ASKED HERE. `open_blocks` was a strict SUFFIX until 2026-09-05 - the count was
# the only thing a recipe could say, so the whole depth sweep on record (0, 2, 6, 12) moved the SIZE of
# the window and never its POSITION. Every number in the ledger is a trailing-block fit. The early and
# interior cells are not "another rung of a closed sweep"; they are an axis with no measurement on it.
#
# THE MECHANISM. Lee et al. (ICLR 2023, openreview APuPRxjHvZ) fine-tune a single CONTIGUOUS BLOCK of a
# pretrained network and show the best block is decided by the TYPE of distribution shift, not by the
# dataset's size or the network's depth: input-level shift (corruption, a different sensor, a different
# stain) is best fixed at the EARLIEST layers, feature-level shift in the middle, output-level shift at
# the head. Their WILDS Camelyon17 result is the input-level cell and its best choice is the earliest
# embedding block; FMoW's geographic shift is feature-level and lands on later attention blocks. On
# several of their cells surgical fine-tuning beats full fine-tuning outright, which is the part that
# makes this worth a card: it is not a regularisation trick, it is a claim that the trailing blocks are
# the WRONG place to spend the gradient when the shift is at the input.
#
# WHICH CELL WE ARE IN, argued rather than assumed. Our trunk is pretrained on natural RGB photographs
# and serves greyscale knee MRI: the pixel statistics, the channel structure, the noise process and the
# spatial frequency content all differ at the INPUT. That is Lee's input-level cell almost by definition,
# and it is the cell their result says the trailing blocks are the wrong choice for - which is the only
# choice this project has ever made. `band_mid` is on the ticket because the argument is not airtight:
# an MRI sequence also differs from a photograph in what the mid-level features MEAN, which is the
# feature-level cell, and one arm cannot separate "early is right" from "not-the-suffix is right".
#
# THE PRICE, before it runs. `band_dense` is 361 s x 12 = 1.20 h; both cells open six blocks, so each
# run costs the same and the pair is ~2.4 h of card. The gain Lee's mechanism could plausibly deliver is
# large by this project's standards - their surgical-over-full margins on the shifted cells run to
# several points - and this is the only open lever whose mechanism could clear the 1.5% bar. Call it one
# chance in four of clearing it, which at 2.4 h is the best ratio on the board; the fallback outcome is
# not nothing, because a flat pair CLOSES a whole axis that has never been measured.
#
# control  band_dense = 0.8578 (best epoch 8), byte-identical flags except `--unfreeze-offset`. The
#          control is already on disk, which is why this arm costs two runs and not three.
#
# PRE-REGISTERED against band_dense = 0.8578, written before the run, with the 0.0062 holdout floor
# setting every boundary. Read the BETTER of the two cells:
#
#   s >= 0.8728            WHERE BEATS HOW MANY, by more than the bar. The position of the window is a
#              first-class recipe axis and every member this project fits is currently fitting the wrong
#              blocks. Re-fit the champion's seats at the winning offset and price the reseat offline
#              before any slot.
#   0.8728 > s >= 0.8640   THE MECHANISM REPRODUCES, sub-bar. Directional. Keep the offset for new
#              members where it is free, and spend the next arm on the DEPTH of the winning window
#              (six blocks may not be the right count once the position is free to move) rather than on
#              a confirmation seed.
#   0.8640 > s >= 0.8516   FLAT. Six blocks are six blocks wherever they sit, on this trunk at this rate.
#              THE POSITION AXIS CLOSES for DINOv2 trailing-block recipes. It does NOT close for the
#              foreign trunks - Lee's effect is about the pretraining-to-target gap, and a RadImageNet or
#              ImageNet-supervised CNN has a different one - but this project has no live CNN lever, so
#              that is a written-off branch, not a queued arm.
#   s < 0.8516             Worse than the control by more than the floor. THE EXPECTED DIRECTION FOR THE
#              EARLY CELL IF LEE'S CLAIM DOES NOT TRANSFER, and it is still a reading: it says the
#              trailing blocks are load-bearing here, which is a fact about the trunk nobody has measured.
#              Before believing it, check the two ways this arm can lie to itself - see the next block.
#
# VERIFY THE FLAG LANDED BEFORE READING ANY SCORE. `unfreeze_last` alone cannot see WHERE, so an offset
# that reached nothing would open the same NUMBER of blocks in the usual place and read as a flat on a
# mechanism that never ran. The recipe line must carry `unfreeze_offset` 6 (early) / 3 (mid), and the
# landed check must report `frozen_top_blocks` equal to it - that field is read back off the WEIGHTS,
# not off the config, which is the whole reason it exists. A run whose `frozen_top_blocks` is 0 is
# `band_dense` under another name and must be thrown away rather than read.
#
# deliberate: --unfreeze-offset is the ONE variable this arm moves off band_dense. The count stays at
# the default six and the rate stays at 4e-5, because moving either alongside makes a gain unattributable
# - which is exactly how `ortho_lpft` lost its own question.
#
# reading: logs/band_surgical_ledger.log
# bead: knee-ep6a4
# also: knee-6hfi, knee-cj40f
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train band_early \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --unfreeze-last 6 --unfreeze-offset 6 \
  --labels "$LABELS" --out "$MODELS/band_early_s2026.pt"

run_train band_mid \
  $RECIPE --epochs 12 --slice-pooling max --whole-stack $THROUGHPUT \
  --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 \
  --unfreeze-last 6 --unfreeze-offset 3 \
  --labels "$LABELS" --out "$MODELS/band_mid_s2026.pt"
