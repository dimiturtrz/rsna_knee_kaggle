#!/usr/bin/env bash
# cnx_dense_448.sh — the retry `cnx_dense` pre-registered for itself, and ONLY that.
#
# ARM THIS ONLY IF `cnx_dense` READ OUT UNDER 0.80 SOLO. That is its own third clause, written before it
# ran: "the head, the band and the slice pooling were all settled on token sequences from a patch-14 ViT;
# a 121-token grid may simply be too coarse for the slot head's spatial pooling, in which case the reading
# is about RESOLUTION and the retry is 448 px (a 14x14 grid) before the trunk is judged." If the parent
# instead landed in 0.82-0.85 the axis moves to roster diversity (knee-ewi7), and if its seat value came
# out at or below zero the axis moves off the encoder entirely (knee-ec0p). This script answers one
# question and it is not the question those two ask.
#
# THE ONE VARIABLE. 336 -> 448 is the SAME 336 px cache read through `--train-size`, exactly as the parent
# reached 352, so the decode, the band, the slice count and the pixels on disk are byte-identical between
# the two runs. An upsample adds NO INFORMATION — and that is the point rather than a caveat. It moves the
# token grid from 11x11 to 14x14, 121 tokens to 196, while holding the image content fixed, which isolates
# GRID COARSENESS from pixel detail. A gain here is the slot head wanting more spatial slots; a flat
# result says the trunk was judged fairly at 352 and the family, not the resolution, is what lost.
#
# 14x14 is chosen because it is the ViT incumbents' own grid at 336/patch-24-equivalent — the geometry the
# head, the band and the max slice pooling were all settled on. So this is the ConvNeXt evaluated on the
# spatial budget its competitors were tuned for, not merely a bigger one.
#
# MEMORY IS THE RISK AND IT IS PRICED, NOT ASSUMED. Activations in a convolutional trunk live in the early
# high-resolution stages, so they scale with PIXELS and not with the final token count: (448/352)^2 = 1.62,
# against the parent's measured 10.49 GB reserved, projects ~17 GB. `--batch 1 --accumulate 6` holds the
# effective batch at the control's 6 while halving the resident stack, which is the substitution
# `small_accum` already measured to be free (0.8579 against batch 6's 0.8598 and the control's 0.8578,
# three points inside a 0.0020 spread). `--vram-gb 24` is that projection with headroom, and it still
# leaves the co-tenant room on a 32 GB card. SMOKE IT FIRST — `python -m kneemri.experiments.arm_check
# --smoke` is what prices the real bill before forty minutes are spent discovering it, and the parent's
# own smoke is the reason its 16 GB cap was a measurement rather than a guess.
#
# Everything else is `cnx_dense`'s, which is `base_dense`'s: 4e-5, 20 slices, band 0.10-0.90, whole-stack,
# max slice pooling, 12 epochs one-cycle, the standing $RECIPE.
#
# control  cnx_dense itself — same trunk, same data, same recipe, one geometry apart. base_dense = 0.8512
#          is the ViT this family is competing with, band_dense = 0.8578 the recipe's reference point.
# seat     comp/champion.json basedense_v1 = 0.8690 over the 909-study fold-0 holdout, five members.
# cost     ~1.6x the parent per step at half the batch, so expect the epoch to run longer than 663 s.
#
# PRE-REGISTERED, three outcomes. Floor 0.0062, board resolution 0.0015.
#   THE GRID WAS THE PROBLEM — solo clears the parent by more than 0.0062. Then the ConvNeXt was never
#     judged at 352: the slot head needs spatial slots, the trunk is viable, and the family axis reopens
#     at the resolution the incumbents were tuned on.
#   THE TRUNK WAS THE PROBLEM — solo within the floor of the parent. Then 62% more pixels bought nothing,
#     resolution is exonerated as the explanation, and ConvNeXtV2 under THIS head is finished as a member
#     at 87.7M parameters. The encoder-family axis has then read out negative twice (v3base_dense, this)
#     and the next member comes from the DATA side, knee-ec0p.
#   VRAM — if it will not fit at batch 1, the retry is 392 px (a 12x12 grid, 1.24x the parent's pixels)
#     rather than fewer slices or a narrower band. Slices and band are load-bearing — whole-stack reading
#     is the +0.0147 lever — and the batch is measured not to be.
#
# ARM_LIB is set when this arm runs from a snapshot copy (see run_snapshot); $0 moves, the lib does not.
source "${ARM_LIB:-$(dirname "$0")/../lib.sh}"
require "$LABELS"

run_train cnx_dense_448 \
  $RECIPE --epochs 12 --batch 1 --accumulate 6 --slice-pooling max --whole-stack \
  --variant cnxbase --backbone-lr 4e-5 --slices 20 --band-low 0.1 --band-high 0.9 --train-size 448 \
  --vram-gb 24 \
  --labels "$LABELS" --out "$PROCESSED/cnx_dense_448_s2026.pt"
